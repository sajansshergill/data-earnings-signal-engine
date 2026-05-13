# Alternative Data Earnings Signal Engine
**Causal inference + LLM-Native Workflow on Transaction Panel Data**

Simulates a YiptData-style research pipeline: engineers signals from synthetic transaction-level panel data, applies Difference-in-Differences causal modeling to isolate revenue infections, predicts earnigs suprise direction, and deploys an LLM agent to auto-generate investor-facting white paper sections —— compete with figures, equations, and narrative farming.

## Table of Contents
1. Overview
2. Architecture
3. Project Structure
4. Components
5. Tech Stack
6. Setup & Installation
7. Usage
8. LLM-Native Worflow
9. Eval Harness
10. Results & Key Findings
11. White Paper Output Sample
12. Design Decsions
13. Limitations & Future Work

## Overview
Alternative data —— transaction records, invoice feeds, web-scraped panels –– reaches institutional investors weeks before quarterly earnings. The question is never just <em>"is there a spike?"</em> but <em>"is this spike causally attributable to the company's performance, or is it a market-wide trend?</em>
This project build that full pipeline end-to-end:
1. **Panel Dtaa Simulation** —— Generates realistic transaction-level data for treatment and control firms across 12 quarters, with injected revenue inflection events
2. **Feature Engineering** –— Computes cohort retention, spend velocity, new-vs-returning customer ratio, and rolling single momentum using PySpark-style vectorized operations
3. **Causal Inference (DiD)** –— Applies Difference-in-Differences with two-way fixed effects (form + time) to isolate the causal lift attributable to the treatment event, not macro trends
4. **Earnings Surprise Classifier** —— XGBoost that maps engineered signals to binary earnings surprise direction (beat / miss)
5. **LLM White Paper Agent** —— Claude API agent that ingests model outputs and generates an investor-facing white paper section with equation citations, figure callouts, and confidence-qualified narrative
