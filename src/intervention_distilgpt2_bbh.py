import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset_utils.bigbench import get_bb_dataset
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class Results:
    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

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

    def get_choice_tokens(self, choices, tokenizer):
        """获取选项的Token ID，单Token返回列表，多Token返回None"""
        choice_token_ids = []
        for choice in choices:
            tokenized = tokenizer(f" {choice}", add_special_tokens=False)["input_ids"]
            if len(tokenized) != 1:
                return None  # 多Token选项
            choice_token_ids.append(tokenized[0])
        return choice_token_ids

    def single_token_eval(self, prompt, label, model_edit, choices, choice_token_ids):
        """单Token选项评估：通过最后一个Token的Logits判断"""
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = model_edit(inputs.input_ids)[0][:, -1, :]  # [batch_size, vocab_size]
        log_prob = torch.nn.functional.log_softmax(logits, dim=1).squeeze()
        choice_logprobs = [log_prob[tid].item() for tid in choice_token_ids]
        pred_id = np.argmax(choice_logprobs)
        label_id = choices.index(label)
        is_correct = (pred_id == label_id)
        answer_log_prob = choice_logprobs[label_id]
        return is_correct, ContextAnswerLogProb(
            total_log_prob=answer_log_prob,
            answer_log_prob=answer_log_prob,
            answer_len=1
        )

    def multi_token_eval(self, prompt, label, model_edit, choices):
        """多Token选项评估：计算完整序列的对数概率"""
        all_logprobs = []
        for choice in choices:
            full_prompt = f"{prompt} {choice}"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(self.device)
            outputs = model_edit(inputs.input_ids)
            logits = outputs.logits[0]
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)
            # 计算答案部分的对数概率（假设答案是最后一个Token或序列）
            answer_token_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
            answer_start = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            answer_log_prob = sum(log_prob[answer_start + i, tok].item() 
                                 for i, tok in enumerate(answer_token_ids))
            all_logprobs.append(answer_log_prob)
        pred_id = np.argmax(all_logprobs)
        label_id = choices.index(label)
        is_correct = (pred_id == label_id)
        return is_correct, all_logprobs[label_id]

    def intervene(self, model, tokenizer, dataset, args, llm_name, choices):
        """执行干预实验并评估"""
        dataset_size = len(dataset)
        self.logger.log(f"=== Starting Intervention ===")
        self.logger.log(f"Layer: {args.lnum}, Type: {args.lname}, Rate: {args.rate}, Dataset Size: {dataset_size}")

        # 复制原始模型并应用干预
        model_edit = deepcopy(model)
        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(
            model=model_edit,
            lname=args.lname,
            lnum=args.lnum,
            rate=args.rate,
            intervention=args.intervention,
            logger=self.logger,
            in_place=True
        )
        model_edit.to(self.device).eval()
        self.logger.log(f"Model edited in {elapsed_from_str(time_edit_start)}")

        # 准备评估工具
        choice_token_ids = self.get_choice_tokens(choices, tokenizer)
        single_token = choice_token_ids is not None
        self.logger.log(f"Choices are single-token: {single_token}, Tokens: {choice_token_ids}")

        predictions = []
        self.dataset_metric.reset()

        for i in tqdm(range(dataset_size), desc="Evaluating"):
            prompt, label = dataset[i]
            with torch.no_grad():
                if single_token:
                    is_correct, logprob = self.single_token_eval(
                        prompt, label, model_edit, choices, choice_token_ids
                    )
                else:
                    is_correct, logprob = self.multi_token_eval(
                        prompt, label, model_edit, choices
                    )
            self.dataset_metric.accept(is_correct=is_correct, log_prob_results=logprob)
            predictions.append({
                "ix": i,
                "prompt": prompt,
                "gold_label": label,
                "correct": is_correct,
                "answer_logprob": logprob.answer_log_prob,
                "answer_length": logprob.answer_len
            })

        self.terminate_and_save(predictions, args)
        return predictions

    def terminate_and_save(self, predictions, args):
        """保存结果并输出摘要"""
        self.dataset_metric.terminate()
        self.logger.log("\n=== Final Performance ===")
        self.dataset_metric.print()

        # 保存预测结果和摘要
        save_dir = f"{self.save_dir}/{args.lnum}-{args.lname}-{args.rate:.2f}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存预测结果
        pred_path = os.path.join(save_dir, "predictions.pkl")
        with open(pred_path, "wb") as f:
            pickle.dump(predictions, f)
        
        # 保存评估指标
        results = self.dataset_metric.agg_to_dict()
        results["args"] = vars(args)
        summary_path = os.path.join(save_dir, "summary.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(results, f)
        
        self.logger.log(f"Results saved to {save_dir}")

    @staticmethod
    def get_acc_log_loss(predictions):
        """计算准确率和对数损失"""
        acc = np.mean([p["correct"] for p in predictions]) * 100.0
        log_loss = np.mean([-p["answer_logprob"] / max(p["answer_length"], 1) for p in predictions])
        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):
        """划分验证集和测试集"""
        val_size = int(len(predictions) * split)
        val_pred, test_pred = predictions[:val_size], predictions[val_size:]
        val_acc, val_loss = DistilGPT2Experiment.get_acc_log_loss(val_pred)
        test_acc, test_loss = DistilGPT2Experiment.get_acc_log_loss(test_pred)
        return Results(val_acc, val_loss, test_acc, test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilGPT2 on BigBench Hard (BBH)')
    parser.add_argument('--split', type=str, default="causal_judgement", help='BBH task split (e.g., causal_judgement)')
    parser.add_argument('--intervention', type=str, default="rank-reduction", choices=['dropout', 'rank-reduction', 'zero'])
    parser.add_argument('--lname', type=str, default="fc_in", choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'])
    parser.add_argument('--lnum', type=int, default=5, help='Layer number (0-5 for DistilGPT2, -1 for all layers)')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate (e.g., 0.1 to 9.9)')
    parser.add_argument('--model_path', type=str, default="distilgpt2", help='Model path or name')
    parser.add_argument('--home_dir', type=str, default="./iclr2024/bbh/distilgpt2", help='Base save directory')

    args = parser.parse_args()
    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(args.device)
    base_model.eval()

    # 创建保存目录
    save_root = os.path.join(args.home_dir, args.split, llm_name, args.intervention)
    os.makedirs(save_root, exist_ok=True)
    logger = Logger(save_dir=save_root, fname="experiment.log")

    # 打印实验配置
    logger.log("=" * 50)
    logger.log(f"DistilGPT2 BBH Experiment: {args.split}")
    logger.log(f"Intervention: {args.intervention}, Layer: {args.lname}-{args.lnum}, Rate: {args.rate}")
    logger.log("=" * 50)

    # 加载数据集和选项
    dataset, choices = get_bb_dataset(args.split)
    logger.log(f"Loaded {len(dataset)} samples. Choices: {choices}")

    # 运行基线模型（不干预）
    if args.lnum == -1 and args.lname == "dont":
        logger.log("=== Baseline Model (No Intervention) ===")
        predictions = DistilGPT2Experiment(save_root, logger).intervene(
            model=base_model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            llm_name=llm_name,
            choices=choices
        )
        results = DistilGPT2Experiment.validate(predictions)
        logger.log(f"Baseline Results: {results.to_str()}")
    else:
        # 运行干预实验
        predictions = DistilGPT2Experiment(save_root, logger).intervene(
            model=base_model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            llm_name=llm_name,
            choices=choices
        )
        results = DistilGPT2Experiment.validate(predictions)
        logger.log(f"Intervention Results: {results.to_str()}")

    logger.log("Experiment Completed.")