# AI-Powered Financial Institution Risk Evaluation

This repository features a scalable, data-driven pipeline designed to automate the risk assessment of financial institutions by analyzing the Consumer Financial Protection Bureau (CFPB) complaint database. The project demonstrates how Large Language Models (LLMs) can be leveraged to transform thousands of unstructured consumer narratives into actionable regulatory intelligence.

---

## Core Objective

Regulators currently face an overwhelming volume of consumer complaints, making it difficult to identify systemic failures manually. This project proposes a framework that uses Zero-Shot LLM scoring to evaluate complaints based on their qualitative impact, enabling consistent and scalable risk monitoring across the industry.

---

## Technical Implementation

* **Model Selection:** To prioritize data privacy and eliminate API costs, the pipeline uses Llama-3-8B-Instruct deployed locally via Ollama.
* **Zero-Shot Learning:** Because manual labeling for fine-tuning is resource-intensive, the system utilizes advanced prompt engineering to score narratives without prior training.
* **Weighted Risk Scorecard:** The framework integrates LLM-derived scores (Severity, Complexity, Actionability) with regulatory metrics (Response Delay) to produce a comprehensive Risk Scorecard.

---

## Dataset Summary

The analysis was performed on a refined sample of the CFPB database, demonstrating a workflow that is fully compatible with large-scale big data processing.

* **CFPB_sample.csv:** A raw subset of the CFPB database containing 100 consumer complaint narratives from 2024.
* **Metrics.csv:** The processed version of the sample, including LLM-generated scores for each individual record.
* **ScoreCard_Company Level.csv:** Aggregated risk scores grouped by financial institution (e.g., Chase, Wells Fargo, Capital One).
* **ScoreCard_Company & Product Level.csv:** Granular risk analysis grouped by both company and specific financial products (e.g., Credit Cards, Mortgages).

---

## Scalability and Professional Application

While implemented as a student project on a standard laptop, this pipeline is architected for enterprise-grade scalability. The local 8B parameter model can be seamlessly replaced with higher-performance LLMs in a secure environment to analyze millions of records. This modular approach makes the tool ideal for regulatory agencies or consultancy firms seeking to automate institutional oversight without compromising data security.

**Main Script:** CFPB FI Risk Assessment Pipeline.ipynb