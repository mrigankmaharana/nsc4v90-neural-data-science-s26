# Neural Data Science (NSC 4V90.001) – Spring 2026

Course repo for **Neural Data Science (NSC 4V90.001)** at UT Dallas, taught by **Dr. Kaushik Lakshminarasimhan**.

This repo contains:

* Python labs for exploratory analysis of neural data (rasters, histograms, tuning curves, event-aligned PSTHs)
* Modeling notebooks: Poisson & exponential models, linear / logistic / multiclass regression, and generalized linear models (GLMs)
* Code for evaluating model fits (cross-validation, overfitting checks)
* Scripts for hypothesis testing on neural activity (t-tests, rank tests, permutation tests, correlation tests)
* Final project code, figures, and slides

Focus: learn to organize, visualize, model, and test hypotheses on real neural datasets while building practical Python skills for scientific computing.

---

## Project Structure

```
notebooks/   interactive analyses
src/         reusable modules (loading, plotting, models, utils)
data/        input datasets (not tracked)
figures/     output plots and figures
```

---

## Data

Datasets are placed in `/data`.
Large or proprietary datasets are excluded from version control via `.gitignore`.

---

## Environment

* Running via Google Colab A100 kernel from VS Code
* Python 3.10+
* Optional: JAX, PyTorch

Dependencies tracked via:

```
pip freeze > requirements.txt
```

---

## Compute Strategy

* **CPU** for exploratory analysis and plotting
* **GPU (A100)** for heavy compute workloads (training loops, large JAX ops, accelerated linear algebra)

This avoids unnecessary compute unit burn on Colab.

---

## Final Project Deliverables

* Reproducible notebook(s)
* Modular code in `/src`
* Figures in `/figures`
* Report and/or slide deck
