# Bagging

This repository provides the official code for the paper "Bagging Improves Generalization Exponentially."

## Repository Structure

### Major Code File: ParallelSolve.py

- Implements algorithms from the paper:
  - `majority_vote` and `majority_vote_LP`: Correspond to BAG (Algorithm 1).
  - `baggingTwoPhase_woSplit` and `baggingTwoPhase_woSplit_LP`: Implement ReBAG (Algorithm 2 without data splitting).
  - `baggingTwoPhase_wSplit` and `baggingTwoPhase_wSplit_LP`: Implement ReBAG-S (Algorithm 2 with data splitting).
  - The subscript `LP` indicates functions dealing with continuous problems (because we need to handle rounding issues differently for discrete and continuous problems).

### Utility Files in `utils/`

- Core functions for various problems:
  - `SSKP`: Resource allocation
  - `network`: Supply chain network design
  - `portfolio`: Portfolio optimization
  - `LASSO`: Model selection
  - `matching`: Maximum weight matching
  - `LP`: Linear programs
- **generateSamples.py**: Generates samples under specified distributions/dimensions for various problems.
- **plotting.py**: Plots figures after experiments are completed.
- Remark: Note that in these files, name `alg3` refers to ReBAG and `alg4` refers to ReBAG-S

### Experiment Scripts

- Scripts ending with `_SAA_comparison.py`: Compare algorithms using SAA as the base model; profile the effects of subsample size ($k$) and number of subsamples ($B, B_1, B_2$).
- Scripts ending with `_epsilon.py`: Profile the effect of threshold $\epsilon$.
- Scripts ending with `_dro_comparison.py`: Compare performance between bagging-enhanced SAA and DRO with the Wasserstein metric.
- Scripts ending with `_dro_bagging.py`: Algorithm comparison using DRO with Wasserstein metric as the base model.
- **SSKP_probability.py**: Draws the probability figure in Figure 9.
- **LP_tail_influence.py**: Evaluates the influence of tail heaviness; used to draw Figure 5a.
- **LP_eta.py**: Compares the value $\eta_{k,\delta}$ for different methods; used to draw Figure 10.

## Installation and Usage

- The Gurobi solver is required for problems other than model selection.
- Run the scripts like the following command in the terminal (use `nohup` due to verbose output from Gurobi):

  ```bash
  nohup python SSKP_SAA_comparison.py &
  ```

## Data and Figures

- **plot_results/**: Contains data and parameters corresponding to the figures in the paper.
  - **plotting.ipynb**: Contains all scripts for generating figures.
  - Subfolders store the specific data for each figure.
