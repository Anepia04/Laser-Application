import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset_utils.bigbench import get_bb_dataset
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress


class Results:
    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_dict(self):
        return {
            "val_acc": self.val_acc,
            "val_logloss": self.val_logloss,
            "test_acc": self.test_acc,
            "test_logloss": self.test_logloss
        }

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"


class DistilGPT2Experiment:
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        self.progress = Progress(logger=logger)
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        self.dataset_metric = DatasetMetrics(logger=logger)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def intervene(self, model, tokenizer, dataset, args, llm_name):
        dataset_size = len(dataset)
        self.logger.log(f"Starting intervention for layer {args.lnum}, type {args.lname}, rate {args.rate}. Dataset size: {dataset_size}")

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(
            model=model,
            lname=args.lname,
            lnum=args.lnum,
            rate=args.rate,
            intervention=args.intervention,
            logger=self.logger,
            in_place=True
        )
        model_edit.to(self.device)
        self.logger.log(f"Edited model on {model_edit.device} in {elapsed_from_str(time_edit_start)}")

        predictions = []
        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=dataset_size - i)

            prompt, answer = dataset[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            input_and_answer = tokenizer(prompt + " " + answer, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generate_ids = model_edit.generate(
                    inputs.input_ids,
                    max_new_tokens=args.max_len,
                    min_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                generation = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

                # 计算对数概率
                results = model_edit(input_and_answer.input_ids)
                logits = results.logits[0]
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                log_prob_results = self.metrics.answer_log_prob(
                    log_prob=log_prob,
                    question_answer_token_ids=input_and_answer.input_ids[0],
                    answer=answer,
                    llm_tokenizer=tokenizer
                )

            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

            self.dataset_metric.accept(
                is_correct=is_correct,
                f1pr_score=f1pr_score,
                log_prob_results=log_prob_results
            )

            predictions.append({
                "ix": i,
                "prompt": prompt,
                "gold_answer": answer,
                "generation": generation,
                "correct": is_correct,
                "f1_score": f1pr_score.f1,
                "precision": f1pr_score.precision,
                "recall": f1pr_score.recall,
                "total_logprob": log_prob_results.total_log_prob,
                "answer_logprob": log_prob_results.answer_log_prob,
                "answer_length": log_prob_results.answer_len
            })

        self.terminate_and_save(predictions)
        return predictions

    def terminate_and_save(self, predictions):
        self.logger.log("Saving results. Final Performance:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.lnum}-{args.lname}-{args.rate}.p"
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.lnum}-{args.lname}-{args.rate}.pkl"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results[f"args/{k}"] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        self.logger.log(f"Results saved in {elapsed_from_str(time_start)}")

    @staticmethod
    def get_acc_log_loss(predictions):
        acc = np.mean([1.0 if pred["correct"] else 0.0 for pred in predictions]) * 100.0
        log_loss = np.mean([-pred["answer_logprob"] / pred["answer_length"] for pred in predictions])
        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):
        val_size = int(split * len(predictions))
        val_pred = predictions[:val_size]
        test_pred = predictions[val_size:]
        val_acc, val_loss = DistilGPT2Experiment.get_acc_log_loss(val_pred)
        test_acc, test_loss = DistilGPT2Experiment.get_acc_log_loss(test_pred)
        return Results(val_acc, val_loss, test_acc, test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilGPT2 on BigBench Hard QA')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--split', type=str, default="qa_wikidata", help='BBH split (e.g., qa_wikidata)')
    parser.add_argument('--max_len', type=int, default=10, help='Max generation length')
    parser.add_argument('--intervention', type=str, default="rank-reduction", choices=['dropout', 'rank-reduction', 'zero'])
    parser.add_argument('--lname', type=str, default="None", choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'])
    parser.add_argument('--lnum', type=int, default=5, help='Layer number (0-5 for DistilGPT2)', choices=list(range(-1, 6)))
    parser.add_argument('--model_path', type=str, default="distilgpt2", help='Model path')
    parser.add_argument('--home_dir', type=str, default="/mnt/data/iclr2024/bbh_qa/distilgpt2_results", help='Save directory')
    parser.add_argument('--save_path', type=str, default="/mnt/data/iclr2024/bbh_qa/distilgpt2_results", help='Results save path')

    args = parser.parse_args()

    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(model.device)

    save_dir = f"{args.home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    logger.log("=" * 50)
    logger.log(f"DistilGPT2 BBH QA Experiment: Layer {args.lname}-{args.lnum}, Rate {args.rate}")
    logger.log("=" * 50)
    for k, v in args.__dict__.items():
        logger.log(f">>> Argument {k}: {v}")
    logger.log("=" * 50)

    dataset, _ = get_bb_dataset(args.split)
    logger.log(f"Loaded {len(dataset)} samples for split {args.split}")

    experiment = DistilGPT2Experiment(save_dir=save_dir, logger=logger)
    predictions = experiment.intervene(model, tokenizer, dataset, args, llm_name)

    results = experiment.validate(predictions, split=0.2)
    logger.log(f"Results: {results.to_str()}")
    logger.log("Experiment completed.")

    summary = results.to_dict()
    for k, v in vars(args).items():
        summary[f"args/{k}"] = v

    with open(f"{args.save_path}/results_{args.split}_{args.lnum}_{args.lname}_{args.rate}.pkl", "wb") as f:
        pickle.dump(summary, f)