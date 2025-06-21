import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from dataset_utils.bias_in_bios import BiasBiosGender
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class DistilGPT2BiosExperiment:
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
        self.logger.log(f"Starting intervention with rate {args.rate}. Dataset size {dataset_size}")

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

        # 定义性别标签对应的Token ID
        gender_tokens = {
            "female": tokenizer(" female")["input_ids"][0],
            "male": tokenizer(" male")["input_ids"][0]
        }

        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=dataset_size - i)

            text = dataset[i]["hard_text"]
            gold_gender = "female" if dataset[i]["answer"] == 1 else "male"

            # 构建提示模板
            prompted_text = f"Consider the following text: {text.strip()}.\nIs the person in this text male or female?\nThe gender of this person is"
            inputs = tokenizer(prompted_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # 获取最后一个Token的Logits
                outputs = model_edit(inputs.input_ids)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)

                # 提取性别标签的Log概率
                female_logprob = log_prob[0, gender_tokens["female"]].item()
                male_logprob = log_prob[0, gender_tokens["male"]].item()

                # 判断预测结果
                if gold_gender == "female":
                    is_correct = female_logprob > male_logprob
                    answer_log_prob = female_logprob
                else:
                    is_correct = male_logprob > female_logprob
                    answer_log_prob = male_logprob

                # 计算Top-k准确率
                sorted_logprob, sorted_indices = torch.sort(log_prob, descending=True)
                top_tokens = tokenizer.batch_decode(sorted_indices[0, :10])
                top_1_acc = gold_gender in [t.strip().lower() for t in top_tokens[:1]]
                top_5_acc = gold_gender in [t.strip().lower() for t in top_tokens[:5]]
                top_10_acc = gold_gender in [t.strip().lower() for t in top_tokens[:10]]

                # 计算总对数概率（问题+答案）
                total_log_prob = log_prob.sum().item()
                logprob_results = ContextAnswerLogProb(
                    total_log_prob=total_log_prob,
                    answer_log_prob=answer_log_prob,
                    answer_len=1
                )

            self.dataset_metric.accept(
                is_correct=is_correct,
                f1pr_score=None,
                log_prob_results=logprob_results,
                top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc}
            )

            predictions.append({
                "ix": i,
                "text": text,
                "prompted_text": prompted_text,
                "gold_gender": gold_gender,
                "female_logprob": female_logprob,
                "male_logprob": male_logprob,
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_tokens": top_tokens,
                "correct": is_correct,
                "case-sensitive": self.case_sensitive,
                "white-space-strip": self.strip,
                "total_logprob": total_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "prompt_length": inputs.input_ids.shape[1]
            })

            if i % 100 == 0:
                self.logger.log(f"Example {i}: Text='{text[:50]}...', Gold Gender='{gold_gender}', Correct={is_correct}")

        self.terminate_and_save(predictions, llm_name)

    def terminate_and_save(self, predictions, llm_name):
        self.logger.log("Saving results. Final Performance:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        save_pred_fname = f"{self.save_dir}/{llm_name}-bios-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        save_summary_fname = f"{self.save_dir}/{llm_name}-bios-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results[f"args/{k}"] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        self.logger.log(f"Results saved in {elapsed_from_str(time_start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilGPT2 on Bias in Bios Dataset')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (DistilGPT2 prefers batch=1)')
    parser.add_argument('--k', type=int, default=10, help='Top-k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help='Intervention type')
    parser.add_argument('--lname', type=str, default="None",
                        choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None', 'dont',
                                 "all", "mlp", "attn"],
                        help='Layer name to intervene')
    parser.add_argument('--lnum', type=int, default=5, help='Layer number (0-5 for DistilGPT2)',
                        choices=list(range(-1, 6)))
    parser.add_argument('--model_path', type=str, default="distilgpt2", help='Model path')
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/bios_gender/distilgpt2_results",
                        help='Directory to save results')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/bios_gender/bios_gender_dataset.pkl",
                        help='Path to Bias in Bios dataset file')

    args = parser.parse_args()

    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(model.device)

    save_dir = f"{args.home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-bios-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    logger.log("=" * 50)
    logger.log(f"DistilGPT2 Bias in Bios Experiment: Layer {args.lname}-{args.lnum}, Rate {args.rate}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>> Argument {k}: {v}")
    logger.log("=" * 50)

    # 加载Bias in Bios数据集
    dataset_util = BiasBiosGender()
    dataset = dataset_util.get_dataset(logger)
    logger.log(f"Loaded Bias in Bios dataset with {len(dataset)} samples")

    experiment = DistilGPT2BiosExperiment(save_dir=save_dir, logger=logger)
    experiment.intervene(model=model, tokenizer=tokenizer, dataset=dataset, args=args, llm_name=llm_name)

    logger.log("Experiment completed.")