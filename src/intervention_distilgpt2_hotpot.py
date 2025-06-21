import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from dataset_utils.hotpot import Hotpot
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class DistilGPT2HotpotExperiment:
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

        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=dataset_size - i)

            question = dataset[i]["question"]
            answer = dataset[i]["answer"]
            
            # 构建提示模板
            if not question.endswith("?") and not question.endswith("."):
                prompted_question = f"{question}? The answer is"
            else:
                prompted_question = f"{question} The answer is"

            # 准备输入
            inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)
            input_and_answer = tokenizer(prompted_question + " " + answer, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # 生成回答
                if args.beam > 1:
                    generate_ids = model_edit.generate(
                        inputs.input_ids,
                        max_new_tokens=args.max_len,
                        min_new_tokens=1,
                        num_beams=args.beam,
                        do_sample=False
                    )
                else:
                    generate_ids = model_edit.generate(
                        inputs.input_ids,
                        max_new_tokens=args.max_len,
                        min_new_tokens=1
                    )

                generation = tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].replace(prompted_question, "").strip()

                # 计算对数概率
                results = model_edit(input_and_answer.input_ids)
                logits = results.logits[0]
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)

                # 提取回答部分的对数概率
                answer_start_idx = inputs.input_ids.shape[1]
                answer_token_ids = input_and_answer.input_ids[0, answer_start_idx:]
                answer_log_prob = log_prob[answer_start_idx-1:-1, answer_token_ids].diag().sum().item()
                total_log_prob = log_prob.diag().sum().item()

                logprob_results = ContextAnswerLogProb(
                    total_log_prob=total_log_prob,
                    answer_log_prob=answer_log_prob,
                    answer_len=len(answer_token_ids)
                )

            # 计算评估指标
            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)

            self.dataset_metric.accept(
                is_correct=is_correct,
                f1pr_score=f1pr_score,
                log_prob_results=logprob_results
            )

            if i % 10 == 0:
                self.logger.log(f"Question: {prompted_question}")
                self.logger.log(f"Gold Answer: {answer}")
                self.logger.log(f"Generated: {generation}")
                self.logger.log(f"Correct: {is_correct}, F1: {f1pr_score.f1:.4f}")

            predictions.append({
                "ix": i,
                "question": question,
                "prompted_question": prompted_question,
                "gold-answer": answer,
                "generation": generation,
                "correct": is_correct,
                "f1_score": f1pr_score.f1,
                "precision": f1pr_score.precision,
                "recall": f1pr_score.recall,
                "case-sensitive": self.case_sensitive,
                "white-space-strip": self.strip,
                "total_logprob": total_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": len(answer_token_ids),
                "question_answer_length": input_and_answer.input_ids.shape[1]
            })

        self.terminate_and_save(predictions, llm_name)

    def terminate_and_save(self, predictions, llm_name):
        self.logger.log("Saving results. Final Performance:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        save_pred_fname = f"{self.save_dir}/{llm_name}-hotpot-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        save_summary_fname = f"{self.save_dir}/{llm_name}-hotpot-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results[f"args/{k}"] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        self.logger.log(f"Results saved in {elapsed_from_str(time_start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistilGPT2 on HotpotQA Dataset')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--beam', type=int, default=1, help='Beam size for generation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (DistilGPT2 prefers batch=1)')
    parser.add_argument('--max_len', type=int, default=15, help='Maximum length for generation')
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
                        default="/mnt/data/iclr2024/hotpot/distilgpt2_results",
                        help='Directory to save results')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/hotpot/hotpot_dataset.pkl",
                        help='Path to HotpotQA dataset file')

    args = parser.parse_args()

    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(model.device)

    save_dir = f"{args.home_dir}_{args.beam}/{llm_name}/{args.intervention}/{args.lname}"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-hotpot-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    logger.log("=" * 50)
    logger.log(f"DistilGPT2 HotpotQA Experiment: Layer {args.lname}-{args.lnum}, Rate {args.rate}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>> Argument {k}: {v}")
    logger.log("=" * 50)

    # 加载HotpotQA数据集
    dataset_util = Hotpot()
    dataset = dataset_util.get_dataset(logger)
    logger.log(f"Loaded HotpotQA dataset with {len(dataset)} samples")

    experiment = DistilGPT2HotpotExperiment(save_dir=save_dir, logger=logger)
    experiment.intervene(model=model, tokenizer=tokenizer, dataset=dataset, args=args, llm_name=llm_name)

    logger.log("Experiment completed.")