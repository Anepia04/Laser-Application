import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as opt

from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset_utils.truthfulqa import get_truthfulqa_pointwise_data
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class GPT2Experiment:
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
        self.logger.log(f"Starting intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

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
        self.logger.log(f"Edited and placed model on {model_edit.device} in {elapsed_from_str(time_edit_start)}")

        predictions = []

        # 获取true/false的token_id（确保包含前置空格）
        true_token_id = tokenizer(" true")["input_ids"][0]
        false_token_id = tokenizer(" false")["input_ids"][0]

        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=dataset_size - i)

            prompt, label = dataset[i]

            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                logits = model_edit(**inputs).logits[0, -1, :]  # 获取最后一个token的预测
                log_probs = torch.nn.functional.log_softmax(logits, dim=0)

                true_log_prob = log_probs[true_token_id].item()
                false_log_prob = log_probs[false_token_id].item()

                if label == 0:  # False
                    is_correct = false_log_prob > true_log_prob
                    answer_log_prob = false_log_prob
                else:
                    assert label == 1, f"Label must be 0 or 1. Found {label}"
                    is_correct = true_log_prob > false_log_prob
                    answer_log_prob = true_log_prob

                # 修正：使用正确的total_log_prob计算方式
                total_log_prob = answer_log_prob  # 简化版本，实际应计算整个序列的log_prob
                log_prob_results = ContextAnswerLogProb(
                    total_log_prob=total_log_prob,
                    answer_log_prob=answer_log_prob,
                    answer_len=1
                )

            self.dataset_metric.accept(
                is_correct=is_correct,
                f1pr_score=None,
                log_prob_results=log_prob_results
            )

            predictions.append({
                "ix": i,
                "question": prompt,
                "log_losses": -answer_log_prob,
                "gold-answer": label,
                "correct": is_correct,
                "case-sensitive": self.case_sensitive,
                "white-space-strip": self.strip,
                "total_logprob": total_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "true_false_log_probs": {"true": true_log_prob, "false": false_log_prob},
                "question_answer_length": inputs.input_ids.shape[1]
            })

        self.terminate_and_save(predictions)
        return predictions

    def terminate_and_save(self, predictions):
        self.logger.log("Saving results. Final performance:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results[f"args/{k}"] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        self.logger.log(f"Results saved in {elapsed_from_str(time_start)}")

    def evaluate(self, test_logits, temp):
        mean_log_prob = 0.0
        for indices, logit in test_logits:
            indices = torch.from_numpy(indices).to(self.device)
            logit = torch.from_numpy(logit).to(self.device)
            log_prob = torch.nn.functional.log_softmax(logit / temp, dim=1)
            selected_log_prob = torch.gather(log_prob, index=indices.view(-1, 1), dim=1)
            mean_log_prob += selected_log_prob.sum().item() / indices.shape[0]
        mean_log_prob /= len(test_logits)
        self.logger.log(f"Temperature {temp}: Mean log prob {mean_log_prob} on {len(test_logits)} samples")

    def temperature_tuning(self, predictions, val=0.2):
        val_size = int(val * len(predictions))
        val_logits = [item["answer_logits"] for item in predictions[:val_size]]
        test_logits = [item["answer_logits"] for item in predictions[val_size:]]

        self.logger.log(f"Starting temperature tuning with {len(val_logits)} val and {len(test_logits)} test samples")
        self.evaluate(test_logits, 1.0)

        temp_logit = nn.Parameter(torch.FloatTensor([1.0]))
        optimizer = opt.Adam([temp_logit], lr=0.001)

        for epoch in range(1000):
            total_loss = 0.0
            for indices, logit in val_logits:
                indices = torch.from_numpy(indices).to(self.device)
                logit = torch.from_numpy(logit).to(self.device)
                temp = torch.nn.functional.sigmoid(temp_logit)
                log_prob = torch.nn.functional.log_softmax(logit / temp, dim=1)
                loss = -torch.gather(log_prob, index=indices.view(-1, 1), dim=1).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            temp_value = torch.nn.functional.sigmoid(temp_logit).item()
            self.logger.log(f"Epoch {epoch+1}, loss: {total_loss/len(val_logits):.3f}, temperature: {temp_value}")

            if epoch % 100 == 0:
                self.evaluate(test_logits, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT-2 Intervention on TruthfulQA')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--dtpts', type=int, default=817, help='Number of data points')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 for GPT-2)')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'], help='Intervention type')
    parser.add_argument('--lname', type=str, default="q_proj",
                        choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out','dont'],
                        help='Layer name to intervene')
    parser.add_argument('--lnum', type=int, default=11, help='Layer number to intervene (0-11 for GPT-2)')
    parser.add_argument('--model_path', type=str, default="gpt2", help='Model path')
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/truthfulqa/gpt2_pointwise_results",
                        help='Directory to save results')
    parser.add_argument('--dataset_file', type=str, default="None", help='Dataset file')

    args = parser.parse_args()

    # 加载模型和分词器
    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = GPT2LMHeadModel.from_pretrained(llm_name, torch_dtype=torch.float16)

    # 创建保存目录和日志
    save_dir = f"{args.home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # 初始化实验
    experiment = GPT2Experiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Experiment: {llm_name}, Layer: {args.lname}-{args.lnum}, Rate: {args.rate}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f"Argument {k}: {v}")
    logger.log("=" * 50)

    # 加载数据集并执行干预实验
    dataset = get_truthfulqa_pointwise_data(logger)
    predictions = experiment.intervene(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=args,
        llm_name=llm_name
    )

    # 可选：温度调优
    # experiment.temperature_tuning(predictions)

    logger.log("Experiment completed.")