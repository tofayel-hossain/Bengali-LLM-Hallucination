# Bengali LLM Hallucination Detection

> Solution for the **Bengali LLM Hallucination Detection Competition**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg">
  <img src="https://img.shields.io/badge/Language-Bengali-green.svg">
</p>

## 🏆 Competition Result

- **Competition:** Bengali LLM Hallucination Detection
- **Final Rank:** **80th / 222 Teams**
- **Task:** Detect whether a Bengali Large Language Model (LLM) generated answer is **faithful** or **hallucinated**.


# Problem Statement

Large Language Models often generate responses that appear fluent but contain incorrect or fabricated information. This competition focused on automatically identifying hallucinated answers in Bengali.

Given a question, answer, and optionally supporting context, the objective is to classify whether the generated response is:

- **0 → Hallucinated**
- **1 → Faithful**

# Our Solution

Instead of relying on a single classifier, we designed a **hybrid multi-stage pipeline** that routes samples based on the availability of supporting context.

The overall architecture consists of two specialized branches.

## Context Branch

For samples containing context, we trained a lexical and semantic context alignment model that evaluates whether the generated answer is supported by the provided evidence.

Features include:

- Text normalization
- Lexical overlap features
- Semantic similarity
- Bangla-specific preprocessing
- Context-aware classification

## Null Context Branch

When no supporting context is available, the model follows a dedicated pipeline:

1. BanglaBERT-based classifier
2. Answer retrieval module
3. Evidence verification
4. Hybrid prediction model

This branch is specifically designed for cases where external evidence is unavailable.

## Fusion Strategy

The outputs from both branches are combined using:

- Soft Voting
- Logistic Regression Stacking
- Threshold Optimization
- Out-of-Fold (OOF) Validation

The original hybrid prediction is preserved whenever fusion does not improve validation performance.

# Project Pipeline

```
                Input Sample
                      │
         ┌────────────┴────────────┐
         │                         │
  Context Available?           No Context
         │                         │
         ▼                         ▼
  Context Branch            BanglaBERT Pipeline
         │                         │
         ▼                         ▼
  Context Classifier       Retrieval + Verifier
         │                         │
         └────────────┬────────────┘
                      │
              Soft Voting Fusion
                      │
		   Logistic Regression Stack
                      │
                Final Prediction
```

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- BanglaBERT
- Scikit-learn
- Pandas
- NumPy

# Model Components

- BanglaBERT
- Logistic Regression
- Context Alignment Classifier
- Retrieval-based Verification
- Soft Voting Ensemble
- Stacking Ensemble

# Validation Strategy

- 5-Fold Cross Validation
- Out-of-Fold Predictions
- Threshold Optimization
- Macro F1 Score Evaluation

# Dataset

You can download the dataset from Kaggle:

**🔗 Kaggle Dataset:**  
`https://www.kaggle.com/datasets/tofayelhossain/hallucinate-dataset`

# How to Run

## Clone Repository

```bash
git clone https://github.com/tofayel-hossain/Bengali-LLM-Hallucination.git
cd Bengali-LLM-Hallucination
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Notebook

Open:

```
bengali_hallucination_final.ipynb
```

Run all cells on a GPU-enabled environment (Kaggle or Google Colab).

# Results

| Metric | Value |
|---------|------:|
| Competition Rank | **80 / 222 Teams** |
| Validation | 5-Fold Cross Validation |
| Ensemble | Logistic Regression Stacking |
| Language | Bengali |

# Future Improvements

- Integrate Wikipedia-based retrieval for external fact verification.
- Explore Retrieval-Augmented Generation (RAG).
- Experiment with larger Bengali language models.
- Improve ensemble learning and threshold optimization.
- Expand the training dataset with more diverse examples.

# Team

**Tofayel Hossain**

Computer Science & Engineering

American International University-Bangladesh (AIUB)

GitHub: https://github.com/tofayel-hossain
