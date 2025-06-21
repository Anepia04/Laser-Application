import os
import time
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from dataset_utils.bias_in_bios import BiasBiosOccupation
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


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
        self.logger.log(f"Starting intervention with rate {args.rate}. Dataset size: {dataset_size}")

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

        # 职业标签及Token ID映射
        choice_tokens = BiasBiosOccupation.occupations
        choice_token_ids = [
            tokenizer(f" {token.strip()}")["input_ids"][0] 
            for token in choice_tokens
        ]

        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=dataset_size - i)

            text = dataset[i]["hard_text"]
            answer_ix = dataset[i]["answer"]
            answer = choice_tokens[answer_ix]

            # 构建提示模板
            if text.strip().endswith((".", "?")):
                prompt = f"Consider the following text: {text.strip()}\nWhat is the profession of the person in this text? The profession of this person is"
            else:
                prompt = f"Consider the following text: {text.strip()}.\nWhat is the profession of the person in this text? The profession of this person is"

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model_edit(inputs.input_ids)
                logits = outputs.logits[0, -1, :]  # 取最后一个Token的Logits
                log_prob = torch.nn.functional.log_softmax(logits, dim=0)

                # 计算各职业标签的Log概率
                choices_logprob = torch.stack([
                    log_prob[tid] for tid in choice_token_ids
                ]).numpy()

                # 评估指标
                is_correct = np.argmax(choices_logprob) == answer_ix
                answer_log_prob = choices_logprob[answer_ix]

                # 计算Top-k准确率
                sorted_indices = torch.argsort(log_prob, descending=True)[:10]
                top_tokens = tokenizer.batch_decode(sorted_indices)
                top_tokens = [t.strip().lower() for t in top_tokens]
                target = answer.lower().strip()

                top_1_acc = 1.0 if target in top_tokens[:1] else 0.0
                top_5_acc = 1.0 if target in top_tokens[:5] else 0.0
                top_10_acc = 1.0 if target in top_tokens[:10] else 0.0

                # 计算总对数概率
                question_log_prob = self.calculate_question_logprob(outputs.logits, inputs.input_ids)
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(
                    total_log_prob=total_log_prob,
                    answer_log_prob=answer_log_prob,
                    answer_len=1
                )

            self.dataset_metric.accept(
                is_correct=is_correct,
                top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc},
                log_prob_results=logprob_results
            )

            predictions.append({
                "ix": i,
                "text": text,
                "prompt": prompt,
                "gold_answer": answer,
                "gold_answer_ix": answer_ix,
                "generation": top_tokens[0] if top_tokens else "",
                "correct": is_correct,
                "choices_logprob": choices_logprob.tolist(),
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_tokens": top_tokens,
                "case-sensitive": self.case_sensitive,
                "white-space-strip": self.strip,
                "total_logprob": total_log_prob,
                "question_logprob": question_log_prob,
                "answer_logprob": answer_log_prob
            })

            if i % 100 == 0:
                self.logger.log(f"Example {i}: Text='{text[:50]}...', Gold={answer}, Correct={is_correct}")

        self.terminate_and_save(predictions, llm_name)

    def calculate_question_logprob(self, logits, input_ids):
        """计算提示文本的对数概率（不包含答案部分）"""
        input_ids = input_ids[0, :-1]  # 排除最后一个Token（答案位置）
        log_prob = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=2)
        selected_logprob = log_prob.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        return selected_logprob.sum().item()

    def terminate_and_save(self, predictions, llm_name):
        self.logger.log("Saving results. Final Performance:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        save_dir = f"{self.save_dir}/{llm_name}/{args.intervention}/{args.lname}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存预测结果
        pred_path = f"{save_dir}/{llm_name}-predictions-{args.rate}-{args.lnum}.p"
        with open(pred_path, "wb") as f:
            pickle.dump(predictions, f)

        # 保存评估摘要
        summary = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            summary[f"args/{k}"] = v
        summary_path = f"{save_dir}/{llm_name}-summary-{args.rate}-{args.lnum}.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump(summary, f)

        self.logger.log(f"Results saved in {elapsed_from_str(time_start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilGPT2 on Bias in Bios (Profession)')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (DistilGPT2 prefers batch=1)')
    parser.add_argument('--intervention', type=str, default="rank-reduction", choices=['dropout', 'rank-reduction'])
    parser.add_argument('--lname', type=str, default="None", choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'])
    parser.add_argument('--lnum', type=int, default=5, help='Layer number (0-5 for DistilGPT2)', choices=list(range(-1, 6)))
    parser.add_argument('--model_path', type=str, default="distilgpt2", help='Model path')
    parser.add_argument('--home_dir', type=str, default="/mnt/data/iclr2024/bios_profession/distilgpt2_results", help='Save directory')
    parser.add_argument('--dataset_file', type=str, default="/mnt/data/bios_profession_dataset.pkl", help='Dataset path')

    args = parser.parse_args()

    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(args.device)

    save_dir = args.home_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    logger.log("=" * 50)
    logger.log(f"DistilGPT2 Bios Profession Experiment: Layer {args.lname}-{args.lnum}, Rate {args.rate}")
    logger.log("=" * 50)
    for k, v in args.__dict__.items():
        logger.log(f">>> Argument {k}: {v}")
    logger.log("=" * 50)

    dataset_util = BiasBiosOccupation()
    dataset = dataset_util.get_dataset(logger)
    logger.log(f"Loaded {len(dataset)} profession bias samples")

    experiment = DistilGPT2Experiment(save_dir=save_dir, logger=logger)
    experiment.intervene(model=model, tokenizer=tokenizer, dataset=dataset, args=args, llm_name=llm_name)

    logger.log("Experiment completed.")