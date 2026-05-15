# Alternative Data Earnings Signal Engine
**Causal inference + LLM-Native Workflow on Transaction Panel Data**

Simulates a YipitData-style research pipeline: engineers signals from synthetic transaction-level panel data, applies Difference-in-Differences causal modeling to isolate revenue inflections, predicts earnings surprise direction, and deploys an LLM agent to auto-generate investor-facing white paper sections complete with figures, equations, and narrative framing.

## Table of Contents
1. Overview
2. Architecture
3. Project Structure
4. Components
5. Tech Stack
6. Setup & Installation
7. Usage
8. LLM-Native Workflow
9. Eval Harness
10. Results & Key Findings
11. White Paper Output Sample
12. Design Decisions
13. Limitations & Future Work

## Overview
Alternative data — transaction records, invoice feeds, and web-scraped panels — reaches institutional investors weeks before quarterly earnings. The question is never just <em>"is there a spike?"</em> but <em>"is this spike causally attributable to the company's performance, or is it a market-wide trend?"</em>
This project builds that full pipeline end-to-end:
1. **Panel Data Simulation** — Generates realistic transaction-level data for treatment and control firms across 12 quarters, with injected revenue inflection events
2. **Feature Engineering** — Computes cohort retention, spend velocity, new-vs-returning customer ratio, and rolling signal momentum using vectorized operations
3. **Causal Inference (DiD)** — Applies Difference-in-Differences with two-way fixed effects (firm + time) to isolate the causal lift attributable to the treatment event, not macro trends
4. **Earnings Surprise Classifier** — XGBoost maps engineered signals to binary earnings surprise direction (beat / miss)
5. **LLM White Paper Agent** — Claude API agent that ingests model outputs and generates an investor-facing white paper section with equation citations, figure callouts, and confidence-qualified narrative
6. **Eval Harness** — Rubric-based scoring system that audits LLM output across six dimensions: factual accuracy, citation discipline, causal language precision, hedging appropriateness, figure consistency, and readability
7. **Streamlit Dashboard** — Dual-stakeholder interface showing signal diagnostics for analysts and white paper output for clients

## Architecture
<img width="372" height="551" alt="Screenshot 2026-05-13 at 2 24 00 PM" src="https://github.com/user-attachments/assets/7386370e-7c1a-4348-9dc4-55a0fbb531d0" />

## Project Structure
<img width="744" height="1720" alt="image" src="https://github.com/user-attachments/assets/c174b49c-3a42-489a-92e7-8fff7d6171ba" />

## Components
**1. Panel Data Simulation (src/simulation/panel_generator.py)**
Generates a balanced panel of 50 firms x 12 quarters:
- 10 treatment firms, 40 control firms
- Transaction volume, average order value, cohort size per quarter
- Revenue inflection event injected at Q7 for treatment group (effect size: +15% GMV)
- Gaussian noise calibrated to real e-commerce volatility benchmarks

**2. Feature Engineering (src/features/engineer.py)**
All operations vectorized (no Python loops on DataFrames):
<img width="1168" height="460" alt="image" src="https://github.com/user-attachments/assets/8d238dfe-eb93-4518-ac5b-514dacd74b3a" />

**3. Causal Inference (src/causal/)**
Y_{it} = α_i + λ_t + β·(Treat_i × Post_t) + ε_{it}
- α_i — firm fixed effects (absorbs time-invariant confounders)
- λ_t — time fixed effects (absorbs macro trends)
- β — causal estimate of the treatment effect
- Pre-treatment parallel trends assumption formally tested and plotted

**4. Earnings Surprise Classifier (src/predictive)**
- **Model**: XGBoost with walk-forward cross-validation (no data leakage)
- **Target**: Binary — earnings beat (+1) or miss (0) vs. analyst consensus
- **Features**: All engineered signals + DiD residuals as a meta-feature
- **Explainability**: SHAP waterfall plots per prediction, global feature importance
- **Threshold tuning**: Recall-prioritized (false negatives are more costly for investors than false positives)

**5. LLM White Paper Agent (src/llm/)**
Claude API agent that receives structured model outputs and generates a white paper section formatted for institutional investor readers:
#### Input content passed to agent:
- DiD β estimate with 95% confidence interval
- Top 3 SHAP features driving the earnings surprise signal
- XGBoost model AUC and precision/recall at operating threshold
- Pre-treatment parallel trends test result
- Figure file references (DiD event study plot, SHAP waterfall)

**Output:** 400-600 word white paper section with:
- Methodology paragraph (reference DiD literature, cites Callaway & Sant'Anna 2021)
- Results paragraph with inline equation and confidence framing
- Limitations paragraph (honest about synthetic control assumptions)
- Figure callouts with interpretive captions

**6. Eval Harness (src/eval/)**
Rubric-based scoring of every LLM output before it surfaces to users — mirrors the citation discipline and methodological rigor the JD explicitly calls out:
<img width="1264" height="546" alt="image" src="https://github.com/user-attachments/assets/f9d582cb-1060-49de-b05b-852e194ec01a" />

Scores are logged to DuckDB. Outputs below 0.75 composite are flagged for human review and not surfaced on the dashboard.

## Tech Stack
<img width="1104" height="778" alt="image" src="https://github.com/user-attachments/assets/87322f83-2a18-46da-ba03-253166b0db70" />

## Setup & Installation
#### Clone the repo
git clone https://github.com/sajanshergill/alternative-data-earnings-signal-engine.git
cd alternative-data-earnings-signal-engine

#### Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

#### Install dependencies
pip install -r requirements.txt

#### Configure environment
cp .env.example .env
##### Add your ANTHROPIC_API_KEY to .env

#### Run the full pipeline
python src/simulation/panel_generator.py
python src/features/engineer.py
python src/causal/did_model.py
python src/predictive/xgb_classifier.py
python src/llm/white_paper_agent.py
python src/eval/harness.py

#### Launch dashboard
streamlit run src/dashboard/app.py

## Usage
### Run end-to-end pipeline
```python
from src.simulation.panel_generator import PanelConfig, generate_panel
from src.features.engineer import engineer_features
from src.causal.did_model import run_did
from src.predictive.xgb_classifier import train_and_evaluate
from src.llm.white_paper_agent import generate_from_results
from src.eval.harness import evaluate_output

#### 1. Generate panel
panel = generate_panel(PanelConfig(n_firms=50, n_quarters=12, treatment_effect=0.15))

#### 2. Engineer features
features = engineer_features(panel)

#### 3. Causal estimate
did_results = run_did(features)
print(f"DiD β: {did_results.beta:.3f} (95% CI: {did_results.ci_lower:.3f}, {did_results.ci_upper:.3f})")

#### 4. Predictive model
model_metrics = train_and_evaluate(features)
print(f"AUC: {model_metrics['auc']:.3f} | Recall@threshold: {model_metrics['recall']:.3f}")

#### 5. Generate white paper section
wp_section = generate_from_results(did_results, model_metrics)

#### 6. Evaluate output
eval_score = evaluate_output(wp_section, did_results, model_metrics)
print(f"Eval composite score: {eval_score.composite:.2f}")
```

## LLM-Native Workflow
This project was built with LLM coding assistants as a primary collaborator across every phase — consistent with how the role expects you to work. Specific patterns used:
### Where the LLM multiplies output:
- Scaffolded the DiD fixed-effects model setup (fixed-effects APIs are verbose; assistant got it right on first pass)
- Generated the rubric scoring logic from a natural language description of each dimension
- First draft of the white paper prompt template, iterated 3 times based on output quality

### Where I overrode the assitant: 
- Initial SHAP integration used "TreeExplainer" with background data — the assistant defaulted to "summary_plot" which did not match the waterfall format needed; replaced with a waterfall pipeline manually
- Walk-forward CV logic had a subtle look-ahead leak in the first generated version; caught in code review and corrected
- Causal language in the first white paper draft used "causes" without DiD qualification — flagged by the eval harness and corrected in prompt v2

## Eval Harness
Every LLM-generated white paper section is scored before surfacing. Sample output:
Run ID: wp_run_20240315_143022
─────────────────────────────────────────────
Factual Accuracy       0.92  ✓
Citation Discipline    0.85  ✓
Causal Language        0.78  ✓
Hedging                0.90  ✓
Figure Consistency     0.80  ✓
Readability            0.88  ✓
─────────────────────────────────────────────
Composite Score        0.86  ✓ APPROVED
─────────────────────────────────────────────

## Results & Key Findings
<img width="1104" height="714" alt="image" src="https://github.com/user-attachments/assets/0c710347-023a-49cf-b8fa-701a500e2345" />

## White Paper Output Sample
>> <em> The Difference-in-Differences specification with two-way fixed effects isolates a statistically significant revenue inflection attributable to the treatment event (β = 0.147, 95% CI: [0.091, 0.203], p < 0.01), controlling for firm-level heterogeneity and quarter-specific macroeconomic trends. The pre-treatment parallel trends assumption was formally tested and not rejected (F-test p = 0.61), lending credibility to the causal interpretation (Callaway & Sant'Anna, 2021). Signal momentum — the exponentially weighted growth rate of transaction volume — emerges as the dominant predictive feature (mean |SHAP| = 0.34), consistent with prior literature linking GMV acceleration to positive earnings surprise (see Figure 1)...</em>

## Design Decisions
- **Why DiD over regression discontinuity?**
The treatment event (product launch / pricing change) affects firms at the same calendar time, making RD inappropriate. DiD with firm fixed effects is the standard approach for this panel structure.
- **Why recall-prioritized threshold tuning?**
For institutional investors, missing a true earnings beat (false negative) has asymmetric downside compared to a false positive — the cost of acting on a miss signal is recoverable; the cost of missing a major beat is not.
- **Why DuckDB for eval logging?**
Zero-config, embedded, columnar — right tool for append-heavy eval logs queried analytically. No infrastructure overhead for a research pipeline.
- **Why audit the LLM output rather than trust it?**
The eval harness exists precisely because LLMs produce plausible-sounding text that can misstate confidence intervals, use unhedged causal language, or cite figures that don't exist. The rubric is the guardrail.

## Limitations & Future Work
- **Synthetic data:** Panel is simulated; real alternative data panels (transaction, invoice, web-scraped) would require licensing and PII handling not in scope here
- **Single treatment event:** Production systems would handle staggered adoption DiD (Callaway & Sant'Anna estimator) for firms treated at different times
- **LLM hallucination floor:** Even with the eval harness, composite scores below 0.75 require human review — full automation is not the goal
- **Earnings consensus proxy:** Beat/miss labels are simulated; live deployment would integrate IBES or Bloomberg consensus estimates


## References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2), 200–230.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. Review of Economic Studies, 72(1), 1–19.
