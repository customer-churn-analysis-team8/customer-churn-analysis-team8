**ACS WILDA Program**

**Team 8**

**Customer Churn Analysis**

**Of**

**Advanced Telecommunications Solutions (ATS)**

**Final Report**

|  |  |
| --- | --- |
| **Team Members** | **Role** |
| Ridham Rakeshkumar Champaneria | Project Manager |
| Sheetal Talreja | Data Engineer |
| Ruyu Liu | Data Analyst – Clustering |
| Sabrina Islam | Data Analyst – Predictive Modeling |
| Maksud Jamil | Business Analyst |

## Executive summary

Churn at ATS is driven mainly by **how new a customer is** and **how much they pay**. New, high-spend customers are the clearest place to act: they are 31% of the base but 58% of churners and 65% of the monthly revenue lost to churn.

The analysis covered 7,043 customers, 26.5% of whom churned. It ran in three stages:

### 1. Data preparation: cleaning, encoding and scaling the 10-attribute dataset.
### 2. Customer segmentation: K-Means clustering on tenure and monthly charges, which produced four segments.
### 3. Churn prediction: an artificial neural network (ANN) trained on all 11 encoded attributes and tested on 1,409 customers it had not seen.

| Headline result | Value |
| --- | --- |
| Overall churn rate | 26.5% (1,869 of 7,043 customers) |
| Churn rate, highest-risk segment (New, High-Spend) | 49.2% |
| Churn rate, lowest-risk segment (Loyal, Low-Spend) | 4.8% |
| ANN ranking accuracy (ROC-AUC, test set) | 0.807 |
| Churners caught by the ANN (test set) | 76% (283 of 374) |
| Churners reached by contacting the riskiest 20% | 48%, 2.4× random targeting |
| Monthly revenue lost to churn | $139,107 of $456,096 (30.5%) |

**Top recommendations:**

- Run a first-year onboarding and loyalty programme, because risk falls fastest over a customer's first 12–18 months.
- Give New, High-Spend customers a proactive value review and bill-optimisation offer.
- Use the ANN's risk tiers to rank who gets contacted first, so the retention budget goes where churn is likely.
- Don't base retention on contract-length incentives alone. In this data, churn is nearly the same across contract types (25–27%).
- Collect behavioural data (support contacts, usage, satisfaction) to explain churn that tenure and price don't.

## 1. Summary of key findings

### 1.1 Customer segmentation

K-Means clustering on tenure and monthly charges found **four segments**. Both the elbow method and the silhouette score (0.476, the highest of k = 2–10) pointed to k = 4. **Tenure is the main line between low-risk and high-risk customers.** Spend then decides how risky the new customers are.

| Segment | Customers | Avg tenure (months) | Avg monthly bill ($) | Churn rate | Share of all churners | Share of revenue lost to churn |
| --- | --- | --- | --- | --- | --- | --- |
| C0 New, High-Spend | 2,202 (31%) | 14.8 | 81.23 | 49.2% | 58% | 65% |
| C1 New, Low-Spend | 1,733 (25%) | 10.5 | 32.49 | 24.5% | 23% | 11% |
| C2 Loyal, High-Spend | 1,948 (28%) | 58.8 | 93.08 | 15.7% | 16% | 21% |
| C3 Loyal, Low-Spend | 1,160 (16%) | 54.1 | 34.15 | 4.8% | 3% | 2% |

Revenue lost = total monthly bill of customers who churned ($139,107 per month across all segments).

![Image: image_001](./Final_Report_images/image_001.png)

- C0 is the priority segment. It churns at twice the rate of C1, the other new-customer segment, and accounts for about two-thirds of lost revenue.
- Long tenure protects customers whatever they spend. C2 pays the most of any segment, yet churns at 15.7%, a third of C0's rate.
- A model-free check backs this up. Churn is 47.4% in the first year, 25.5% at 12–36 months and 11.9% beyond 3 years.

### 1.2 Churn prediction (ANN)

The ANN predicts churn well enough to target retention: it ranks a churner above a non-churner **81% of the time** (ROC-AUC 0.807) on customers it never saw in training.

- Architecture: 11 inputs, two hidden layers (32 and 16 neurons, tanh activation) and one sigmoid output giving the probability of churn. This was chosen by 5-fold cross-validation across 72 configurations.
- Training: the Adam optimiser (learning rate 0.001) with L2 regularisation and early stopping, which picked the best of 140 epochs (epoch 115). Training and validation loss stayed within 0.006 of each other, so there was no overfitting.
- Decision threshold: 0.24, tuned on validation data to catch more churners.

| Test-set metric (1,409 customers) | ANN | Logistic regression (baseline) |
| --- | --- | --- |
| ROC-AUC | 0.807 | 0.807 |
| Recall: churners caught | 75.7% | 65.0% |
| Precision: flagged customers who churn | 45.6% | 52.8% |
| F1 | 0.569 | 0.583 |

- The predicted probabilities are well calibrated. Customers scored at 40% churn about 40% of the time, so scores can be read as real likelihoods.
- Predictions agree with the segments. Average predicted churn is within 3 points of actual churn in every segment (e.g. C0: 52% predicted vs 50% actual). The ANN catches 92% of C0's churners.
- Risk tiers separate customers sharply. High-tier customers (probability of 0.60 or more) churn at 67.6%, compared with 11.5% in the Low tier.
- The ANN matches the simpler logistic regression rather than beating it. The churn pattern in these 11 attributes is close to linear, so the ANN's extra flexibility adds no accuracy.

## 2. Factors contributing to churn and retention

The ANN's predictions rest on **two factors: tenure (retention) and monthly charges (churn)**. Senior status adds a little, and the other attributes have almost no effect once those two are known. Importance was measured by shuffling each attribute in turn and recording how much the model's accuracy fell.

![Image: image_002](./Final_Report_images/image_002.png)

Predictive\_Modeling/Model\_Evaluation/feature\_importance.csv · permutation importance, 30 repeats, 1,409 test customers

### 2.1 How each factor moves churn risk

Direction and size were measured by varying one attribute while holding everything else fixed (partial dependence and what-if analysis), then checked against raw churn rates.

| Factor | Effect | ANN predicted churn | Raw churn rate | Role |
| --- | --- | --- | --- | --- |
| Tenure | Risk falls steeply over the first 12–18 months, then keeps falling | ~60% at month 1 → ~30% at ~20 months → <10% after 5 years | 52.9% (0–6 months) vs 9.5% (48+ months) | **Strongest retention factor** |
| Monthly charges | Risk rises steadily with the bill | ~10% at $20 → ~50% at ~$105 | Churners concentrate in the high-bill, low-tenure group | **Strongest churn factor** |
| Senior citizen | Seniors are at higher risk | 36% vs 26% | 41.7% vs 23.6% | Churn factor (moderate) |
| Dependents | Customers with dependents are a little safer | 24% vs 29% | 15.5% vs 31.3% | Retention factor (weak); most of the raw gap comes from their longer tenure |
| Phone service | Customers without phone service are at higher risk at the same bill | 37% vs 27% | 24.9% vs 26.7% | Weak; only 10% of customers have no phone service, so treat with caution |
| Contract type | Almost no effect | 26–29% across all three types | 27.3% month-to-month, 25.8% one year, 25.4% two year | **Not a driver in this data** |
| Internet service, gender, multiple lines | No meaningful effect | Within 1.4 points | Within 2 points | Not drivers |

### 2.2 What this means

- Churn is a new-customer problem. The first year is the riskiest window: 47.4% of first-year customers churned. Customers who get past 18–24 months tend to stay.
- Price, or value for money, is the second lever. At any given tenure, a higher bill means higher risk, and the combination of new and expensive is the most dangerous.
- Contract length doesn't protect customers in this dataset. This goes against common industry experience. Before investing in contract-migration campaigns, ATS should check whether contract data is recorded correctly.
- Some churn can't be explained by these attributes. The 91 churners the ANN missed looked like loyal customers: average tenure was 41 months and only 8% were seniors. Their reasons (service problems, competitor offers, moving house) aren't in the data. See Section 4.

## 3. Recommendations for targeted retention strategies

ATS should combine the two analyses. The **segment** sets the type of offer, and the ANN **risk tier** sets who is contacted first and how intensively. Effort should go first to New, High-Spend customers in the High tier.

### 3.1 Strategy by segment

| Priority | Segment | Customers (High / Medium tier) | Why | Recommended actions |
| --- | --- | --- | --- | --- |
| 1 | C0 New, High-Spend | 2,202 (927 / 993) | 49.2% churn; 65% of lost revenue | Plan and bill review in the first 90 days, moving customers to the best-fit plan; a loyalty credit or discount at months 6 and 12; proactive service check-ins; a named contact for High-tier customers |
| 2 | C1 New, Low-Spend | 1,733 (85 / 630) | 24.5% churn; high volume of new customers | Low-cost onboarding: welcome series, setup help, an early satisfaction survey; usage nudges; avoid upsells that raise the bill early on |
| 3 | C2 Loyal, High-Spend | 1,948 (2 / 454) | Low churn (15.7%) but the highest bills, so 21% of lost revenue | Loyalty recognition and tenure rewards; a price-lock or annual value review; priority support. The model catches fewer churners here, so add complaint and usage triggers |
| 4 | C3 Loyal, Low-Spend | 1,160 (0 / 15) | 4.8% churn | Business as usual; a referral programme to turn satisfied customers into new acquisitions; no retention spend |

Across all segments, give **senior customers** (16% of the base, 41.7% churn) assisted onboarding, a dedicated support line and plain-language billing. They make up 23% of C0.

### 3.2 Action by ANN risk tier

| Tier | Probability | Customers (whole base) | Test-set churn rate | Action |
| --- | --- | --- | --- | --- |
| High | 0.60 or more | 1,014 | 67.6% | Personal outreach by the retention team within 2 weeks, with a segment-specific offer |
| Medium | 0.24 to 0.60 | 2,092 | 34.6% | Automated, lower-cost offers (email/SMS, a plan-review invitation); move to personal outreach if behaviour worsens |
| Low | Below 0.24 | 3,937 | 11.5% | No retention spend; standard service and satisfaction monitoring |

Prioritising by score is much more efficient than contacting customers at random. The riskiest 20% of customers include **48% of all churners**, and the riskiest 40% include 73%.

**Illustrative value (assumption to validate in a pilot):** High-tier customers average $80.90 a month. At a 67.6% churn rate, about $55,000 of monthly revenue is at risk in that tier. If outreach kept 20% of those churners, ATS would retain about **$11,000 a month (about $133,000 a year)**, before the cost of offers.

### 3.3 Product and pricing actions

- Fix the first-year experience. Risk falls fastest over the first 12–18 months, so onboarding quality, first-bill clarity and early issue resolution have the largest payoff.
- Review value for money on high-priced plans. Churn rises steadily with the monthly bill, so test bundles or loyalty pricing for new customers on bills above about $70.
- Don't rely on contract migration alone. Contract length shows no protective effect in this data (25–27% churn across types). Check this against billing records before funding contract-upgrade incentives.

### 3.4 Rollout and measurement

#### 4. Pilot (8–12 weeks): contact High-tier C0 customers, keeping a randomly selected control group with no contact so the true uplift can be measured.
#### 5. Scale: extend to all High-tier customers, then to Medium-tier customers through automated channels, keeping the offers that worked in the pilot.
#### 6. Embed: re-score all customers monthly with the saved model (ann\_churn\_pipeline.joblib) and feed the tiers to the retention team's queue.

| KPI | Target direction |
| --- | --- |
| Churn rate, C0 New, High-Spend | Down from 49.2% |
| First-year churn rate | Down from 47.4% |
| Retention uplift, contacted vs control group | Positive and statistically significant |
| Revenue retained vs cost of offers | Above 1 (positive return) |
| Model recall and precision on new churners | Stay at or above 76% and 46% |

## 4. Limitations and proposed solutions

The biggest limitation is the **data, not the model**. With only 10 attributes the ANN reaches a ROC-AUC of 0.807, and the churn it misses needs behavioural data to explain. Each limitation below has a mitigation that is either already applied or proposed.

### 4.1 Data limitations

| Limitation | Impact | Proposed solution / mitigation |
| --- | --- | --- |
| Only 10 attributes: demographics, services, contract and bill | 24% of test churners were missed. These customers looked loyal (average tenure 41 months), so their reasons aren't captured | Add support tickets, complaints, usage trends, payment history, satisfaction or NPS scores, and price-change history |
| No unique customer ID; 302 duplicate row patterns (541 rows) kept | Duplicates can't be confirmed as the same customer; 6.6% of test rows had a twin in training | Tested: removing them changed ROC-AUC only from 0.807 to 0.795. Ask ATS for a customer ID in future extracts |
| Contract type shows no effect on churn (25–27% across types), unlike typical telecom data | Contract-based strategies can't be supported by this data | Check contract fields against billing records before using them for decisions |
| A single snapshot with no dates | Can't see when churn happened or track trends and seasonality | Use monthly snapshots so time-based features (e.g. a recent bill increase) can be built |
| Tenure spikes at 0 and 72 months | 72 may be a cap on data collection, which understates the longest tenures | Confirm with the data owner how tenure is recorded |

### 4.2 Modelling limitations

| Limitation | Impact | Proposed solution / mitigation |
| --- | --- | --- |
| Precision/recall trade-off at threshold 0.24 | 54% of flagged customers wouldn't have churned (337 false alarms in the test set), which adds offer cost | Set the threshold from the cost of an offer vs the value of a saved customer; use tiers so expensive offers go only to the High tier |
| The ANN performs the same as logistic regression (ROC-AUC 0.807 for both) | Extra complexity with no accuracy gain; harder to explain | Keep the ANN to meet the brief, report logistic regression alongside it, and revisit the ANN once richer data is available |
| Neural networks are less transparent | Stakeholders may find predictions hard to trust | Use permutation importance, partial dependence and the well-calibrated probabilities (Section 2); share the drivers with the retention team |
| Weak at spotting churners in loyal segments (recall 32% in C2, 9% in C3) | Some high-value C2 losses won't be flagged | Add event triggers (complaints, outages, competitor quotes) for C2; extend the attribute set |
| Churn is imbalanced (26.5%) | Accuracy alone is misleading | Mitigated: stratified splits, recall/F1/ROC-AUC reporting and a tuned threshold |
| The model will drift as customers and pricing change | Accuracy will degrade over time | Monitor recall and precision monthly; retrain every quarter or when performance drops |
| Correlation, not causation | The drivers show who is at risk, not the effect of an intervention | Use the pilot's control group (Section 3.4) to measure what retention actions actually achieve |

### 4.3 Project challenges

| Challenge | How it was handled |
| --- | --- |
| The handed-over ANN train/test files (ann\_train.csv, ann\_test.csv) weren't in the repository | Rebuilt the identical 80/20 stratified split from full\_encoded\_cleaned.csv |
| In data preparation the scaler was fitted on all rows before the split (a small leak from the test set) | Refitted the scaler on training rows only for the ANN |
| Deep-learning libraries (TensorFlow/Keras) weren't available in the build environment | Built the ANN with scikit-learn's MLPClassifier, which uses the libraries named in the brief; the model can be ported to Keras if required |
| Inconsistent file paths between notebooks (../data/processed/ vs Data\_Preparation/) | Standardised Predictive Modeling on the files as stored in the repository; earlier notebooks should be aligned the same way |

## Appendix: method and deliverables

### A.1 Method summary

| Stage | Method | Key settings |
| --- | --- | --- |
| Data preparation | Missing-value, duplicate and outlier checks; label encoding for binary attributes, one-hot encoding for Contract; StandardScaler | 7,043 rows × 10 attributes; no missing values; 80/20 stratified split (random\_state 42) |
| Segmentation | K-Means on scaled tenure and monthly charges | k = 4 from the elbow method and silhouette score (0.476); k-means++ initialisation, 10 runs |
| Churn prediction | Feed-forward ANN (scikit-learn MLPClassifier) | 11 → 32 → 16 → 1; tanh hidden layers, sigmoid output; Adam (learning rate 0.001), batch size 64, L2 alpha 0.1; early stopping (best epoch 115); threshold 0.24 |
| Evaluation | Held-out test set of 1,409 customers | ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, calibration, baseline comparison |
| Interpretation | Permutation importance, partial dependence, what-if analysis, segment and risk-tier analysis | 30 shuffles per attribute |

### A.2 Deliverables in the project repository

| Folder | Contents |
| --- | --- |
| 1-1 Project Charter | Shared doc, Objectives, scope, roles, timeline, risks |
| 2-1 Data Preparation | 01\_Data\_Preparation.ipynb, cleaned and encoded datasets |
| 2-2 Clustering Analysis | 02\_Clustering\_Analysis.ipynb / .html, K-Means model, cluster charts, cluster\_assignments.csv |
| 3-1 Predictive Modeling | 03\_Predictive\_Modeling.ipynb / .html, trained ANN (Model/ann\_churn\_pipeline.joblib), training and evaluation charts |
| 3-2 Final Report | churn\_predictions.csv: churn probability, risk tier and segment for all 7,043 customers |