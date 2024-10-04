# VoteEnsemble

This repository provides the official code for the paper "Subsampled Ensemble Can Improve Generalization Tail Exponentially".

## Repository Structure

<!-- ### Major Code File: ParallelSolve.py

- Implements algorithms from the paper:
  - `majority_vote` and `majority_vote_LP`: Correspond to $\mathsf{MoVE}$ (Algorithm 1).
  - `baggingTwoPhase_woSplit` and `baggingTwoPhase_woSplit_LP`: Implement $\mathsf{ROVE}$ (Algorithm 2 without data splitting).
  - `baggingTwoPhase_wSplit` and `baggingTwoPhase_wSplit_LP`: Implement $\mathsf{ROVEs}$ (Algorithm 2 with data splitting).
  - The subscript `LP` indicates functions dealing with continuous problems (because we need to handle rounding issues differently for discrete and continuous problems). -->

### Core Algorithm File: VoteEnsemble.py
- Implements algorithms from the paper:
  - The class `MoVE` defines our $\mathsf{MoVE}$ algorithm.
  - The class `ROVE` defines our $\mathsf{ROVE}$ and $\mathsf{ROVEs}$ algorithms.
  - The abstract class `BaseLearner` provides a template for any base learning algorithm.

### Problem Definition File: BaseLearners.py
- Provides definitions of base learning algorithms for various problems by subclassing `BaseLearner`:
  - The class `BaseLR`: Least squares for linear regression.
  - The class `BaseRidge`: Ridge regression for linear regression.
  - The class `BaseNN`: Adam with early stopping for regression with multilayer perceptrons.
  - The class `BasePortfolio`: SAA for the mean-variance portfolio optimization problem.
  - The class `BaseLP`: SAA for the stochastic linear program.
  - The class `BaseMatching`: SAA and Wasserstein DRO for the maximum weight matching problem.
  - The class `BaseNetwork`: SAA for the supply chain network design problem.
  - The functions `gurobi_SSKP` and `gurobi_SSKP_DRO_wasserstein` in `ParallelSolve.py` define SAA and Wasserstein DRO for the resource allocation problem.

### Utility Files
<!-- - Core functions for various problems:
  - `SSKP`: Resource allocation -->
  <!-- - `network`: Supply chain network design -->
  <!-- - `portfolio`: Portfolio optimization -->
  <!-- - `LASSO`: Model selection -->
  <!-- - `matching`: Maximum weight matching -->
  <!-- - `LP`: Linear programs -->
- **utils/generateSamples.py**: Generates samples under specified distributions/dimensions for various problems.
- **utils/plotting.py**: Plots figures after experiments are completed.
- **dataProcessing.ipynb**: Preprocesses real datasets.
- Remark: Note that in these files, name `alg3` refers to $\mathsf{ROVE}$ and `alg4` refers to $\mathsf{ROVEs}$.

### Experiment Scripts
<!-- - Scripts ending with `_dro_comparison.py`: Compare performance between bagging-enhanced SAA and DRO with the Wasserstein metric. -->
- Scripts ending with `_SAA_comparison.py`: Compare algorithms using SAA as the base model; profile the effects of subsample size ($k,k_1,k_2$) and ensemble sizes ($B, B_1, B_2$).
- Scripts ending with `_epsilon.py`: Profile the effect of threshold $\epsilon$.
- Scripts ending with `_dro_ensemble.py`: Algorithm comparison using DRO with Wasserstein metric as the base model.
- **SSKP_probability.py**: Draws the probability figure in Figure 7.
- **LP_tail_influence.py**: Evaluates the influence of tail heaviness; used to draw Figure 4a.
- **LP_eta.py**: Compares the value $\eta_{k,\delta}$ for different methods; used to draw Figure 17.
- **runExpLP.py**: Run experiments for the stochastic linear program problem.
- **runExpLR.py**: Run experiments for linear regression with least squares.
- **runExpRidge.py**: Run experiments for linear regression with Ridge regression.
- **runExpMatching.py**: Run experiments for the maximum weight matching problem.
- **runExpNetwork.py** and **runExpNetworkHeavy.py**: Run experiments for the supply chain network design problem.
- **runExpPortfolio.py**: Run experiments for the mean-variance portfolio optimization problem.
- **runExpNN.py**: Run experiments on synthetic data for regression with multilayer perceptrons of varying numbers of hidden layers.
- **runRealDataExpNN.py**: Run experiments on real data for regression with multilayer perceptrons of four hidden layers.


## Installation and Usage

- The Gurobi solver is required for problems other than model selection.
- Run the scripts like the following command in the terminal (use `nohup` due to verbose output from Gurobi):

  ```bash
  nohup python SSKP_SAA_comparison.py &
  ```

## Data and Figures

- **plot_results_iclr/**: Contains data and parameters corresponding to the figures in the paper.
  - **plotting_iclr.ipynb**: Contains all scripts for generating figures.
  - Subfolders store the specific data for each figure.