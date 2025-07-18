import os
import time
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset_utils.truthfulqa import get_truthfulqa_pointwise_data
from study_utils.log_utils import Logger
from transformers import DistilBertForMaskedLM
from laser.LaserWrapper import LaserWrapper
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress


class DistilBertTruthfulQAExperiment:
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

        # 获取 true/false 的 token ID
        true_token_id = tokenizer(" true", add_special_tokens=False)["input_ids"]
        false_token_id = tokenizer(" false", add_special_tokens=False)["input_ids"]

        assert len(true_token_id) == 1 and len(false_token_id) == 1, \
            "Expected single token for 'true' and 'false'"
        true_token_id = true_token_id[0]
        false_token_id = false_token_id[0]

        for i in tqdm(range(0, dataset_size, args.batch_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            my_batch_size = min(args.batch_size, dataset_size - i)
            batch = dataset[i: i + my_batch_size]

            # 构造输入并插入 mask token
            input_texts = [question + " " + tokenizer.mask_token for question, _ in batch]
            batch_token_ids_and_mask = tokenizer(input_texts,
                                                 return_tensors="pt",
                                                 padding="longest").to(self.device)

            # 容错提取 mask token 位置（取第一个）
            mask_token_flag = (batch_token_ids_and_mask["input_ids"] == tokenizer.mask_token_id)
            mask_token_ids_list = []

            for j in range(mask_token_flag.shape[0]):
                indices = torch.where(mask_token_flag[j])[0]
                if len(indices) == 0:
                    raise ValueError(f"No mask token found in sample {i + j}")
                mask_token_ids_list.append(indices[0].unsqueeze(0))
            mask_token_ids = torch.cat(mask_token_ids_list)

            # 确保 gold_answers 是字符串类型（兼容原始数据中的整数标签）
            gold_answers = []
            for _, gold_answer in batch:
                if isinstance(gold_answer, str):
                    gold_answers.append(gold_answer.lower().strip())
                elif isinstance(gold_answer, int):
                    gold_answers.append("true" if gold_answer == 1 else "false")
                else:
                    raise ValueError(f"Unexpected gold answer type: {type(gold_answer)}")

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

            batch_top_10_tokens = [
                [tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(10)]
                for j in range(my_batch_size)
            ]

            batch_top_1_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:1]
                                    for j in range(my_batch_size)]
            batch_top_5_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:5]
                                    for j in range(my_batch_size)]
            batch_top_10_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:10]
                                     for j in range(my_batch_size)]

            batch_true_token_ids = torch.LongTensor([true_token_id] * my_batch_size).unsqueeze(1).to(self.device)
            true_log_prob = torch.gather(predicted_logprob, index=batch_true_token_ids, dim=1)[:, 0]

            batch_false_token_ids = torch.LongTensor([false_token_id] * my_batch_size).unsqueeze(1).to(self.device)
            false_log_prob = torch.gather(predicted_logprob, index=batch_false_token_ids, dim=1)[:, 0]

            for j in range(my_batch_size):
                if gold_answers[j] == "true":
                    is_correct = true_log_prob[j].item() > false_log_prob[j].item()
                    answer_log_prob = true_log_prob[j].item()
                    answer_token_id = int(true_token_id)
                else:
                    is_correct = false_log_prob[j].item() > true_log_prob[j].item()
                    answer_log_prob = false_log_prob[j].item()
                    answer_token_id = int(false_token_id)

                answer_len = 1
                logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                      answer_log_prob=answer_log_prob,
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
                    "answer_token_id": answer_token_id,
                    "correct": is_correct,
                    "case-sensitive": False,
                    "white-space-strip": True,
                    "predicted-topk-logprob": sorted_logprob[j],
                    "predicted-topk-token-id": sorted_indices[j],
                    "predicted-topk-tokens": batch_top_10_tokens[j],
                    "true_log_prob": true_log_prob[j].item(),
                    "false_log_prob": false_log_prob[j].item(),
                    "answer_logprob": answer_log_prob,
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
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with DistilBERT LLM on TruthfulQA')
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
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/counterfact/distilbert_results",
                        help='Directory where the data is')

    args = parser.parse_args()

    llm_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = DistilBertForMaskedLM.from_pretrained(llm_name)

    home_dir = args.home_dir
    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    dataset = get_truthfulqa_pointwise_data(logger)

    processed_dataset = []
    for prompt, label in dataset:
        if isinstance(label, int):
            label_str = "true" if label == 1 else "false"
        elif isinstance(label, str):
            label_str = label.lower().strip()
            assert label_str in {"true", "false"}, f"Invalid label string: {label_str}"
        else:
            raise ValueError(f"Unsupported label type: {type(label)}")
        processed_dataset.append((prompt + " <mask>", label_str))

    experiment = DistilBertTruthfulQAExperiment(save_dir=save_dir, logger=logger)
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    experiment.intervene(model=model, tokenizer=tokenizer, dataset=processed_dataset, args=args, logger=logger)