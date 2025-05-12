# CCT Midterm

This project uses PyMC to apply **Cultural Consensus Theory (CCT)** to a dataset about local plant knowledge. It aims to figure out what the group agrees on (the consensus answers) and how knowledgeable each person is (their competence) based on yes/no responses.


## Description

Cultural Consensus Theory (Romney, Weller & Batchelder, 1986) assumes that agreement between informants indicates shared knowledge. This model estimates:

* **Informant Competence (D):** Probability each informant knows the correct answer
* **Consensus Answers (Z):** Most likely culturally agreed-upon answer for each item

Model details:

* `D_i ∼ Uniform(0.5, 1)` — prior for competence
* `Z_j ∼ Bernoulli(0.5)` — prior for consensus answers
* `X_ij ∼ Bernoulli(p_ij)` with
  `p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)`

Inference is done via MCMC sampling using PyMC.


## Requirements

To install the necessary packages:

```bash
pip install pymc numpy pandas arviz
```


## How to Run

1. Ensure the following structure exists:

   * `code/cct.py`
   * `data/plant_knowledge.csv`
2. From the root folder, run:

```bash
cd code
python cct.py
```

The script will:

* Print summary statistics of the posterior
* Plot posterior distributions for competence and consensus answers
* Compare consensus estimates with a naive majority vote

## Outputs

* Posterior summaries of informant competence (`D`)
* Posterior summaries of consensus answers (`Z`)
* Naive majority vote vs CCT consensus key
* Plots of posterior distributions

## Report
For this project, I used PyMC to apply Cultural Consensus Theory (CCT) to a small dataset about local plant knowledge. The model estimates two things: how competent each informant is (how likely they are to know the right answers) and what the consensus answers are for a set of questions. I used a Uniform(0.5, 1) prior for informant competence because it assumes every informant has at least a minimal level of knowledge. For the consensus answers, I used a Bernoulli(0.5) prior to reflect no prior knowledge about which answer (0 or 1) is correct.
I ran the model using MCMC sampling with 4 chains and 2000 draws per chain. The diagnostics looked good, so the model seemed to work well. Based on the results, I estimated how competent each informant was and figured out the most likely answers the group agreed on. To get a final consensus answer key, I just rounded the average predicted value for each question. I also compared these results with a simple majority vote. Most of the time they matched, but sometimes the CCT model gave different answers because it pays more attention to which informants are more reliable. I ran into various different errors and problems including with the PyMC model. I had debugged and stepped away, but then came back to it with fresh eyes. I was able to look up and find much of the information needed. 


## Credits

Thank you to **Professor Joachim Vandekerckhove** for teaching the concepts behind Cultural Consensus Theory and providing the dataset for this assignment.

This README and various supporting code and comments were edited and debugged with the assistance of **ChatGPT by OpenAI**.

## Reference

Romney, A. K., Weller, S. C., & Batchelder, W. H. (1986). Culture as Consensus: A Theory of Culture and Informant Accuracy. *American Anthropologist*, 88(2), 313–338.