import os
import time
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset_utils.bigbench import get_bb_dataset
from study_utils.log_utils import Logger
from transformers import DistilBertForMaskedLM
from laser.LaserWrapper import LaserWrapper
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress


class DistilBertBigBenchExperiment:
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        self.progress = Progress(logger=logger)
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        self.dataset_metric = DatasetMetrics(logger=logger)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def intervene(self, model, tokenizer, dataset, args, logger):
        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")
        time_edit_start = time.time()

        if args.rate == -1:
            model_edit = model
            logger.log("Skipping intervention. Using original model.")
        else:
            model_edit = LaserWrapper.get_edited_model(model=model,
                                                       lname=args.lname,
                                                       lnum=args.lnum,
                                                       rate=args.rate,
                                                       intervention=args.intervention,
                                                       logger=logger,
                                                       in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []
        self.dataset_metric.reset()
        self.progress.start()

        # 获取所有可能答案的 token ID
        choices = args.choices
        choice_token_ids = []

        for choice in choices:
            tokenized = tokenizer(f" {choice}", add_special_tokens=False)["input_ids"]
            assert len(tokenized) == 1, f"Expected single token for answer: {choice}"
            choice_token_ids.append(int(tokenized[0]))

        num_choices = len(choices)
        choice_tensor = torch.LongTensor(choice_token_ids).to(self.device)

        for i in tqdm(range(0, dataset_size, args.batch_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            my_batch_size = min(args.batch_size, dataset_size - i)
            batch = dataset[i: i + my_batch_size]

            # 构造输入并插入 mask token
            input_texts = [f"{question} {tokenizer.mask_token}" for question, _ in batch]

            # 截断输入文本以保留 mask token
            truncated_input_texts = []
            for text in input_texts:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > 490:
                    tokens = tokens[-490:]  # 保留最后 490 个 token
                truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                truncated_input_texts.append(f"{truncated_text} {tokenizer.mask_token}")

            # Tokenize
            batch_token_ids_and_mask = tokenizer(
                truncated_input_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512
            ).to(self.device)

            # 提取 mask token 的位置
            mask_token_flag = (batch_token_ids_and_mask["input_ids"] == tokenizer.mask_token_id)
            mask_token_ids_list = []

            for j in range(mask_token_flag.shape[0]):
                indices = torch.where(mask_token_flag[j])[0]
                if len(indices) == 0:
                    raise ValueError(f"No mask token found in sample {i + j}")
                mask_token_ids_list.append(indices[0].unsqueeze(0))
            mask_token_ids = torch.cat(mask_token_ids_list)

            gold_answers = [gold_answer for _, gold_answer in batch]

            with torch.no_grad():
                logits = model_edit(**batch_token_ids_and_mask).logits
                logprob = torch.log_softmax(logits, dim=2)

            vocab_size = logprob.shape[2]
            mask_token_ids = mask_token_ids.view(my_batch_size, 1, 1)
            mask_token_ids = mask_token_ids.expand([my_batch_size, 1, vocab_size])
            predicted_logprob = torch.gather(logprob, index=mask_token_ids, dim=1)
            predicted_logprob = predicted_logprob[:, 0, :]

            sorted_logprob, sorted_indices = torch.sort(predicted_logprob, descending=True)
            sorted_logprob = sorted_logprob[:, :args.k].detach().cpu().numpy()
            sorted_indices = sorted_indices[:, :args.k].detach().cpu().numpy()

            batch_top_k_tokens = [
                [tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(args.k)]
                for j in range(my_batch_size)
            ]

            batch_top_1_accuracy = [gold_answers[j].lower().strip() in batch_top_k_tokens[j][:1]
                                    for j in range(my_batch_size)]
            batch_top_5_accuracy = [gold_answers[j].lower().strip() in batch_top_k_tokens[j][:5]
                                    for j in range(my_batch_size)]
            batch_top_10_accuracy = [gold_answers[j].lower().strip() in batch_top_k_tokens[j][:10]
                                     for j in range(my_batch_size)]

            # 收集每个样本中各个选项的 log prob
            choices_token_logprobs = []
            for choice_token_id in choice_token_ids:
                batch_choice_token_ids = torch.LongTensor([choice_token_id] * my_batch_size).unsqueeze(1).to(self.device)
                choice_log_prob = torch.gather(predicted_logprob, index=batch_choice_token_ids, dim=1)[:, 0]
                choices_token_logprobs.append(choice_log_prob)
            choices_token_logprobs = torch.vstack(choices_token_logprobs)  # num_choices x batch

            # 获取预测的类别索引
            predicted_choice_ix = choices_token_logprobs.argmax(dim=0)  # batch

            # 记录结果
            for j in range(my_batch_size):
                pred_choice = choices[predicted_choice_ix[j]]
                is_correct = pred_choice.lower().strip() == gold_answers[j].lower().strip()
                answer_len = 1
                logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                      answer_log_prob=choices_token_logprobs[predicted_choice_ix[j], j],
                                                      answer_len=answer_len)

                self.dataset_metric.accept(is_correct=is_correct,
                                          f1pr_score=None,
                                          log_prob_results=logprob_results,
                                          top_k_acc={1: batch_top_1_accuracy[j],
                                                     5: batch_top_5_accuracy[j],
                                                     10: batch_top_10_accuracy[j]})
                predictions.append({
                    "ix": i + j,
                    "question": batch[j][0],
                    "gold-answer": batch[j][1],
                    "predicted-answer": pred_choice,
                    "correct": is_correct,
                    "case-sensitive": False,
                    "white-space-strip": True,
                    "predicted-topk-logprob": sorted_logprob[j],
                    "predicted-topk-token-id": sorted_indices[j],
                    "predicted-topk-tokens": batch_top_k_tokens[j],
                    "answer_length": answer_len
                })

        self.terminate_and_save(predictions)

    def terminate_and_save(self, predictions):
        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()
        time_start = time.time()

        save_pred_fname = f"{self.save_dir}/distilbert-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        save_summary_fname = f"{self.save_dir}/distilbert-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"
        results = self.dataset_metric.agg_to_dict()
        for k, v in vars(args).items():
            results["args/%s" % k] = v
        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with DistilBERT LLM on BigBench')
    parser.add_argument('--st', type=int, default=0, help='Start point of data samples')
    parser.add_argument('--rate', type=float, default=1, help='Rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--k', type=int, default=10, help='Top-k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'],
                        help="Type of intervention")
    parser.add_argument('--lname', type=str, default="fc_in",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None', 'dont'],
                        help="Which parameter to affect")
    parser.add_argument('--lnum', type=int, default=3, help='Layer number to edit', choices=list(range(0, 6)))
    parser.add_argument('--split', type=str, default="causal_judgement", help='big bench split to run on')
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/bigbench/distilbert_results",
                        help='Directory where the data is')

    args = parser.parse_args()

    llm_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = DistilBertForMaskedLM.from_pretrained(llm_name)

    home_dir = args.home_dir
    split = args.split
    save_dir = f"{home_dir}/{split}/{llm_name}/{args.intervention}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log.txt")

    experiment = DistilBertBigBenchExperiment(save_dir=save_dir, logger=logger)
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for k, v in vars(args).items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset, choices = get_bb_dataset(split)
    args.choices = choices  # 保存 choices 到 args 中用于后续评估

    processed_data = []
    for dp in dataset:
        question = dp[0]
        answer = dp[1]
        processed_data.append((question, answer))

    experiment.intervene(model=model, tokenizer=tokenizer, dataset=processed_data, args=args, logger=logger)