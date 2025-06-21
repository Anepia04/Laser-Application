## What is Layer-Selective Rank Reduction?

**LA**yer-**SE**lective **R**ank-Reduction, abbreviated as LASER, is an intervention where we replace a selected weight matrix in the transformer architecture of an LLM with its low-rank approximation. A single LASER transformation consists of 3 hyperparameters: the layer number to modify (&ell;) such as 16th layer, the parameter type (&tau;) such as the first MLP layer, and the fraction of the maximum rank to retain (&rho;) such as 0.01 fraction of the rank. We can write this transformation as (&ell;, &tau;, &rho;) and we can compose these transformations and apply them in parallel. The low-rank approximation is performed using SVD. Figure below from our paper shows an illustration.

![LASER illustration](https://pratyushasharma.github.io/laser/images/main.png)

LASER can give significant performance improvements on question-answerting tasks without additional model training. Our paper presents various results related to evaluating LASER on 3 different LLMs and several LLM benchmarks. This repository contains the code to reproduce these results.

## How to run a sample code

We first discuss installing the code and then discuss how to run an experiment.

### Installation

To install the experiment, please install the pip file. We chiefly just need pytorch and the datasets and transformers package from huggingface. It might be a good idea to create a conda environment.

```bash
pip install -r requirements.txt
```

### Run a sample code

At the moment, each setup is its own file. To run an experiment that performs a single LASER transformer to GPTJ on the Fever dataset, you can run:

```bash
python intervention_distilgpt2_fever.py --lname fc_in --rate 9.9 --lnum 26 ----intervention rank-reduction
```

here _lnum_ is &ell;, _lname_ is &tau;, and _rate_ is related to &rho; by &rho; = 1 - 0.1 * rate. The rate is a value between [0, 10.0] and measures how many components to throw away with 10 means all components are thrown away and we get a 0 matrix and 0 means all components are retained and we retain the original matrix. The use of rate is for legacy reasons and we will refactor the code to directly use &rho; in the future. The mapping for _lname_ that we use is:

**lname** | **description**| 
--- | --- |
dont | use the base model and dont perform intervention |
fc_in | first layer of MLP |
fc_out | second layer of MLP | 
fc_up | a third MLP weight matrix in some LLM, used for Hadamard multiplication | 
mlp | all MLP weight matrices {fc_in, fc_up, fc_out} | 
k_proj | key matrix in self attention | 
v_proj | value matrix in self attention | 
q_proj | query matrix in self attention | 
out_proj | output matrix in self attention |
attn | all attention weight matrices |

## Code Organization

Code is inside the `src` folder. The main experiment files are top-level inside the `src`. The filename convention is `intervention_<llm-name>_<dataset-name>.py` where `<llm-name>` is the name of the LLM and `<dataset-name>` is the name of the dataset. For BigBench, the dataset split is often specified with an additional flag --split. Please see the codebase for details of command line arguments. We will provide a comprehensive tutorial later.

The code for performing laser is inside the `laser` package. We use PyTorch to do SVD and compute low-rank approximation. The code for low-rank approximation happens [here](https://github.com/Anepia04/Laser-Application/blob/master/src/laser/matrix_utils.py#L39). The code for reading and processing dataset is inside `dataset_util`. Finally, metrics and logging are done using the `study_utils`.  
