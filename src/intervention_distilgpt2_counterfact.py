import os
import time
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
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

            question, answer = dataset[i]
            gold_answer_token_ids = tokenizer(answer)["input_ids"]
            assert len(gold_answer_token_ids) == 1, "Answer must be a single token"
            gold_token_id = gold_answer_token_ids[0]

            inputs = tokenizer(question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model_edit(inputs.input_ids)
                logits = outputs.logits[:, -1, :]  # 取最后一个token的logits
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                answer_log_prob = log_prob[0, gold_token_id].item()

                sorted_logprob, sorted_indices = torch.sort(log_prob, descending=True)
                top_k = args.k
                top_logprobs = sorted_logprob[0, :top_k].cpu().numpy()
                top_tokens = tokenizer.batch_decode(sorted_indices[0, :top_k])

                is_correct = (top_tokens[0].lower().strip() == answer.lower().strip())
                top_1_acc = float(is_correct)
                top_5_acc = float(answer.lower().strip() in [t.lower().strip() for t in top_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [t.lower().strip() for t in top_tokens[:10]])

                # 计算问题和答案的总对数概率（简化为仅答案部分，因问题已编码在输入中）
                total_log_prob = answer_log_prob
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
                "question": question,
                "gold-answer": answer,
                "generation": top_tokens[0],
                "correct": is_correct,
                "top_1_acc": top_1_acc,
                "top_5_acc": top_5_acc,
                "top_10_acc": top_10_acc,
                "top_10_logprob": top_logprobs,
                "top_10_tokens": top_tokens,
                "case-sensitive": self.case_sensitive,
                "white-space-strip": self.strip,
                "total_logprob": total_log_prob,
                "answer_logprob": answer_log_prob,
                "answer_length": 1,
                "question_answer_length": inputs.input_ids.shape[1] + 1
            })

        self.terminate_and_save(predictions, llm_name)

    def terminate_and_save(self, predictions, llm_name):
        self.logger.log("Saving results. Final Performance:")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-2 Intervention on CounterFact Dataset')
    parser.add_argument('--rate', type=float, default=1.0, help='Intervention rate')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (GPT-2 prefers batch=1)')
    parser.add_argument('--k', type=int, default=10, help='Top-k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help='Intervention type')
    parser.add_argument('--lname', type=str, default="None",
                        choices=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None', 'dont',
                                 "all", "mlp", "attn"],
                        help='Layer name to intervene')
    parser.add_argument('--lnum', type=int, default=8, help='Layer number (0-11 for GPT-2)',
                        choices=list(range(-1, 12)))
    parser.add_argument('--model_path', type=str, default="gpt2", help='Model path (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/counterfact/gpt2_results",
                        help='Directory to save results')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/counterfact/counterfact_dataset.pkl",
                        help='Path to dataset file')

    args = parser.parse_args()

    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(model.device)

    save_dir = f"{args.home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    logger.log("=" * 50)
    logger.log(f"GPT-2 Experiment: Layer {args.lname}-{args.lnum}, Rate {args.rate}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>> Argument {k}: {v}")
    logger.log("=" * 50)

    with open(args.dataset_file, "rb") as f:
        data = pickle.load(f)

    dataset = [(item["question"], item["gold-answer"]) for item in data]
    logger.log(f"Loaded dataset with {len(dataset)} samples")

    experiment = GPT2Experiment(save_dir=save_dir, logger=logger)
    experiment.intervene(model=model, tokenizer=tokenizer, dataset=dataset, args=args, llm_name=llm_name)

    logger.log("Experiment completed.")