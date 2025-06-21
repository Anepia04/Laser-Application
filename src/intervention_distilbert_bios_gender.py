import os
import time
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from study_utils.log_utils import Logger
from transformers import DistilBertForMaskedLM
from laser.LaserWrapper import LaserWrapper
from dataset_utils.bias_in_bios import BiasBiosGender
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, beautify, Progress


class DistilBertBiosGenderExperiment:
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

        # 获取 male 和 female 对应的 token ID
        male_token_id = tokenizer(" male", add_special_tokens=False)["input_ids"]
        female_token_id = tokenizer(" female", add_special_tokens=False)["input_ids"]

        assert len(male_token_id) == 1 and len(female_token_id) == 1, \
            "Expected single token for 'male' and 'female'"
        male_token_id = male_token_id[0]
        female_token_id = female_token_id[0]

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

            # 确保 gold_answers 是字符串形式
            gold_answers = []
            for _, gold_answer in batch:
                if isinstance(gold_answer, str):
                    gold_answers.append(gold_answer.lower().strip())
                elif isinstance(gold_answer, int):
                    gold_answers.append("male" if gold_answer == 0 else "female")
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

            batch_top_1_accuracy = [gold_answers[j] in batch_top_10_tokens[j][:1]
                                    for j in range(my_batch_size)]
            batch_top_5_accuracy = [gold_answers[j] in batch_top_10_tokens[j][:5]
                                    for j in range(my_batch_size)]
            batch_top_10_accuracy = [gold_answers[j] in batch_top_10_tokens[j][:10]
                                     for j in range(my_batch_size)]

            male_log_probs = predicted_logprob[:, male_token_id]
            female_log_probs = predicted_logprob[:, female_token_id]

            batch_predictions = []
            for j in range(my_batch_size):
                is_correct = batch_top_1_accuracy[j]
                prediction = "male" if male_log_probs[j] > female_log_probs[j] else "female"
                batch_predictions.append(prediction)

                answer_len = 1
                logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                      answer_log_prob=max(male_log_probs[j], female_log_probs[j]),
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
                    "gold-answer": gold_answers[j],
                    "predicted-answer": prediction,
                    "correct": is_correct,
                    "case-sensitive": False,
                    "white-space-strip": True,
                    "predicted-topk-logprob": sorted_logprob[j],
                    "predicted-topk-token-id": sorted_indices[j],
                    "predicted-topk-tokens": batch_top_10_tokens[j],
                    "male_log_prob": male_log_probs[j].item(),
                    "female_log_prob": female_log_probs[j].item(),
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
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with DistilBERT LLM on Bios-Gender')
    parser.add_argument('--st', type=int, default=0, help='Start point of data samples')
    parser.add_argument('--rate', type=float, default=1, help='Rates for intervention')
    parser.add_argument('--dtpts', type=int, default=39642, help='# samples per instruction')
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
                        default="/mnt/data/iclr2024/bios_gender/distilbert_results",
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

    dataset_util = BiasBiosGender()
    dataset = dataset_util.get_dataset(logger)

    processed_data = []
    for dp in dataset:
        bio = dp["hard_text"]

        gender = "male" if dp["answer"] == 0 else "female"

        max_len = 50
        bio_token_ids = tokenizer(bio, add_special_tokens=False)["input_ids"][-max_len:]
        assert len(bio_token_ids) <= max_len
        bio = tokenizer.decode(bio_token_ids, skip_special_tokens=True)


        # 构造 prompt
        if bio.strip().endswith(".") or bio.strip().endswith("?"):
            prompted_bio = f"Consider the following text: {bio.strip()} Is the person in this text male or female? This person is <mask>."
        else:
            prompted_bio = f"Consider the following text: {bio.strip()}. Is the person in this text male or female? This person is <mask>."

        processed_data.append((prompted_bio, gender))

    experiment = DistilBertBiosGenderExperiment(save_dir=save_dir, logger=logger)
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for k, v in vars(args).items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    experiment.intervene(model=model, tokenizer=tokenizer, dataset=processed_data, args=args, logger=logger)