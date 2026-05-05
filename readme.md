# Data Analysis Agent

An AI-powered agent that takes your data and your question, then figures out the rest.

## The Problem

Running data analysis usually means deciding upfront whether you need regression, clustering, PCA, or something else — and then writing the code to do it. Most people either don't know which method fits their question, or they know but don't want to spend time on boilerplate.

## What This Does

You give it a dataset and describe what you want to understand. The agent profiles your data, decides whether supervised or unsupervised learning makes sense, runs the analysis, and explains what it found — in plain English.

No method selection. No preprocessing code. No interpreting metric outputs manually.

## Example

```
$ python main.py --data customers.csv

Loaded 4,820 rows × 11 columns.

What do you want to analyze?
> I want to know which customers are likely to churn next month

Detected target: churn (binary, 18.3% positive rate)
Approach: supervised → classification
Running: Logistic Regression, Random Forest

Results:
  Random Forest — AUC: 0.91, F1: 0.84
  Top predictors: days_since_login, monthly_spend, support_tickets

Insight: Customers who haven't logged in for 30+ days and have filed
at least one support ticket in the past 60 days account for 71% of
churned users in the test set. There are 284 customers currently
matching this profile.
```

## Supported Analysis Types

| User Intent | Method |
|---|---|
| Predict a number (revenue, price, score) | Regression |
| Predict a category (churn, fraud, label) | Classification |
| Find natural groups in the data | Clustering |
| Understand which variables drive variance | PCA / Dimensionality Reduction |

The agent picks based on your data and your goal — you don't have to specify.

## Goals for This Project

- [ ] Accept CSV, Excel, and JSON inputs
- [ ] Profile data automatically (types, missingness, distributions)
- [ ] Use Claude to interpret user intent and select the right method
- [ ] Run preprocessing and model training without user intervention
- [ ] Return results in natural language with actionable insights
- [ ] Save outputs (charts, summary reports) to a local folder

## Tech Stack

- **Decision layer**: Claude via Anthropic API
- **Data**: pandas, numpy
- **ML**: scikit-learn
- **Viz**: matplotlib, seaborn

## Status

Early development. Framework being built out.
