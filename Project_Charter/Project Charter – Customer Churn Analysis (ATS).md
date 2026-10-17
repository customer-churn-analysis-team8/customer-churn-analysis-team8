# Project Charter – Customer Churn Analysis

Work Integrated Learning – Data Analytics (WILDA26\_07A) · Client: Advanced Telecommunications Solutions (ATS)

## Project overview

Advanced Telecommunications Solutions (ATS) is losing about one in four customers: 1,869 of 7,043 customers in the supplied dataset (26.5%) have churned. This project will find who churns and why, predict churn before it happens, and recommend retention actions ATS can target.

**Business problem.** Telecommunications is consolidating around digital services, and high churn erodes revenue and raises acquisition costs. ATS needs evidence on which customer factors drive churn, not intuition.

**Purpose.** Apply Python (Pandas, NumPy, Matplotlib, Scikit-learn) to the ATS customer dataset to segment customers with K-Means clustering, predict churn with an artificial neural network (ANN), and turn both into retention recommendations for stakeholders.

## Objectives and success criteria

The project succeeds when ATS receives a validated churn model, interpretable customer segments and a ranked set of retention actions.

| Objective | Success criterion |
| --- | --- |
| Prepare a clean, model-ready dataset | No missing or inconsistent values; categoricals encoded; numeric features scaled; reproducible 80/20 stratified train/test split |
| Segment customers | Optimal cluster count chosen by the elbow method (backed by silhouette score); each cluster profiled, labelled and visualised |
| Predict churn | ANN beats a simple baseline (logistic regression) on held-out data; reports accuracy, precision, recall, F1 and ROC-AUC; churn-class recall is the priority metric |
| Explain churn drivers | Top factors identified and ranked, consistent across clustering, model and exploratory analysis |
| Recommend retention strategies | At least one targeted action per high-risk segment, tied to evidence in the analysis |
| Communicate outcomes | Final report and stakeholder presentation delivered on time |

Target metric thresholds (for example, recall of 0.70 or better on churners) are proposed and will be confirmed once the baseline is known.

## Scope

The project covers analysis of the supplied ATS dataset only, from preparation through to recommendations.

**In scope**

- Data cleaning, validation, encoding, scaling and train/test splitting of Dataset
- Exploratory analysis of churn by customer attribute
- K-Means customer segmentation with the elbow method
- ANN churn prediction and model evaluation
- Identification of churn and retention factors
- Retention recommendations, limitations, final report and presentation

**Out of scope**

- Collecting new data or joining external sources (billing history, support tickets, network quality)
- Deploying the model into ATS production systems or building a live dashboard
- Implementing or running retention campaigns
- Financial modelling of customer lifetime value (no revenue history is provided)

## Dataset and initial assessment

The Dataset holds 7,043 customers and 10 columns, with no blank cells, but it has two integrity issues to resolve in preprocessing.

| Column | Type | Values |
| --- | --- | --- |
| gender | Categorical | Male 3,555 · Female 3,488 |
| SeniorCitizen | Binary (0/1) | 1 = senior, 16.2% of customers |
| Dependents | Categorical | Yes 2,110 · No 4,933 |
| tenure | Numeric (months) | 0–72, median 29 |
| Phone Service | Categorical | Yes 6,361 · No 682 |
| Multiple Lines | Categorical | Yes 2,971 · No 4,072 |
| Internet Service | Categorical | DSL 3,947 · Fiber optic 3,096 |
| Contract | Categorical | Month-to-month 3,875 · One year 1,473 · Two year 1,695 |
| Monthly Charges | Numeric (whole dollars) | 18–119, median 70 |
| Churn (target) | Categorical | Yes 1,869 (26.5%) · No 5,174 |

**Integrity issues found**

- 260 customers have PhoneService = No but MultipleLines = Yes, which is logically impossible.
- 302 rows are exact duplicates. With no customer ID they may be distinct customers with identical profiles, so the Data Engineer will decide whether to keep or drop them and document the choice.
- 11 customers have tenure 0 (new sign-ups); they stay in but are flagged.
- The target is imbalanced (26.5% churn), so the split must be stratified and the model judged on recall and F1, not accuracy alone.

**Early signals.** Churn falls steadily with tenure: 47.4% in the first 12 months versus 9.5% after 48 months. Seniors churn at 41.7% versus 23.6%, and customers without dependents at 31.3% versus 15.5%. Contract type and internet service differ by less than 2 points, so they look like weak drivers in this version of the data.

## Roles and responsibilities

Five roles share the work; each owns one stage deliverable and reviews the next. Team member names are to be filled in.

| Role | Team member | Responsibilities | Owns deliverable |
| --- | --- | --- | --- |
| Project Manager | Ridham Rakeshkumar Champaneria | Charter, schedule, task tracking, meetings, risk log, quality checks, final submission | 1-1 Project Charter |
| Data Engineer | Sheetal Talreja | Load and validate data, resolve integrity issues, encode categoricals, scale features, stratified train/test split | 2-1 Data Preparation |
| Data Analyst – Clustering | Ruyu Liu | Elbow method and silhouette check, train K-Means, profile, label and visualise clusters | 2-2 Clustering Analysis |
| Data Analyst – Predictive Modeling | Sabrina Islam | Design ANN architecture, train and tune, baseline comparison, evaluate and interpret predictions | 3-1 Predictive Modeling |
| Business Analyst | Maksud Jamil | Translate findings into churn drivers and retention strategies, document limitations, lead report and presentation | 3-2 Final Report |

## Deliverables, milestones and timeline

Five deliverables across three stages, each stage closing with a sign-off before the next owner starts.

![placeholder](./Project Charter – Customer Churn Analysis (ATS)_images\image_001.png)

delivery roadmap · 3 stages, 3 gates

Stages run in sequence because each depends on the previous output: clustering and the ANN both need the prepared dataset, and the report needs both models.

## Resources and tools

The project needs only open-source Python tooling and the supplied dataset.

- Language and environment: Python 3 in Jupyter notebooks (or Google Colab), one notebook per stage
- Libraries: Pandas and NumPy (data handling), Matplotlib (visualisation), Scikit-learn (encoding, scaling, K-Means, metrics, MLPClassifier for the ANN); TensorFlow/Keras optional if the team prefers a deeper ANN
- Data: Dataset supplied by ATS through the course site
- Collaboration: shared folder or Git repository for notebooks and outputs, a task tracker, and a weekly team meeting
- People: five team members, one per role, plus the course facilitator for guidance.

## Risks, assumptions and constraints

The largest risks are data quality and class imbalance; both have mitigations planned in Stage 2.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Inconsistent records (260 Phone Service/Multiple Lines conflicts) | Misleading features | Recode Multiple Lines to No where Phone Service is No; document the rule |
| Duplicate rows (302) with no customer ID | Biased model or leakage between train and test | Test model results with and without duplicates; split before any resampling |
| Class imbalance (26.5% churn) | High accuracy but missed churners | Stratified split, class weights, judge on recall, F1 and ROC-AUC |
| Few features (9 predictors) | Limited predictive power | Engineer features such as tenure bands; set realistic expectations |
| ANN overfitting or slow convergence | Poor generalisation | Early stopping, validation split, compare against logistic regression |
| Team availability or handover delays | Missed milestones | Weekly check-ins, clear handover files between stages, shared tracker |

**Assumptions:** the dataset is a representative snapshot of ATS customers; Churn = Yes means the customer left; Monthly Charges is in whole dollars.

**Constraints:** fixed course timeline; Python and the named libraries only; no access to additional ATS data or stakeholders beyond the brief.

## Communication and approval

The team meets weekly, and each stage closes with a handover and sign-off before the next begins.

- Weekly team meeting (Project Manager chairs): progress, blockers, next actions recorded in the tracker
- Stage handovers: each owner shares the notebook, output files and a short README describing what changed
- Facilitator check-ins: submit each stage deliverable for feedback
- Stakeholder presentation: Business Analyst leads at the end of Stage 3

| Role | Name | Signature | Date |
| --- | --- | --- | --- |
| Project Manager | Ridham Rakeshkumar Champaneria |  |  |
| Data Engineer | Sheetal Talreja |  |  |
| Data Analyst – Clustering | Ruyu Liu |  |  |
| Data Analyst – Predictive Modeling | Sabrina Islam |  |  |
| Business Analyst | Maksud Jamil |  |  |