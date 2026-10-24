# TicketLens: Model Explainability, Bias & Responsible AI Audit

## 1. Executive Summary

This report details feature attribution, SHAP explainability, dataset label representation, and privacy/PII risk checks conducted on the TicketLens IT support ticket classification system.

## 2. Dataset Distribution & Bias Audit

- **Total Tickets Audited:** 350
- **Class Imbalance Ratio (Majority / Minority):** 3.86:1
- **Class Distribution:**
  - **Incident:** 139 samples (39.7%)
  - **Request:** 102 samples (29.1%)
  - **Problem:** 73 samples (20.9%)
  - **Change:** 36 samples (10.3%)

### Ticket Length Profile by Category (Words)

| Category | Mean Word Count | Median Word Count | Min Words | Max Words |
| :--- | :---: | :---: | :---: | :---: |
| Incident | 57.2 | 53 | 7 | 167 |
| Request | 60.7 | 60 | 8 | 257 |
| Problem | 57.9 | 53 | 7 | 147 |
| Change | 73.0 | 73 | 11 | 163 |

## 3. Feature Importance & SHAP Attribution

Top driving n-gram features identified for each IT service category via linear attribution and SHAP values:

### Category: `Change`
**Top Positive Driving Features (TF-IDF Weights):**
`request` (+1.2619), `improve` (+1.2049), `update` (+1.1665), `an update` (+1.1442), `enhance` (+0.9481), `requesting` (+0.923)

**Top SHAP Attributions (|SHAP| impact):**
`for` (0.02774), `request` (0.02501), `problem` (0.01982), `an` (0.01978), `we` (0.01832), `our` (0.0169)

### Category: `Incident`
**Top Positive Driving Features (TF-IDF Weights):**
`issue` (+0.9246), `problem` (+0.8695), `system` (+0.8584), `have` (+0.6844), `restarting` (+0.6715), `software` (+0.6629)

**Top SHAP Attributions (|SHAP| impact):**
`for` (0.02948), `problem` (0.02678), `issue` (0.02564), `medical` (0.02421), `have` (0.0235), `you` (0.02178)

### Category: `Problem`
**Top Positive Driving Features (TF-IDF Weights):**
`issues` (+0.7972), `due` (+0.7545), `due to` (+0.7545), `been` (+0.7352), `unauthorized` (+0.6521), `the server` (+0.6516)

**Top SHAP Attributions (|SHAP| impact):**
`for` (0.02852), `issues` (0.02166), `been` (0.02019), `might` (0.01926), `due` (0.01817), `due to` (0.01817)

### Category: `Request`
**Top Positive Driving Features (TF-IDF Weights):**
`information on` (+1.0146), `on` (+1.0141), `information` (+0.9874), `for` (+0.9501), `could you` (+0.8975), `brand` (+0.8334)

**Top SHAP Attributions (|SHAP| impact):**
`on` (0.03368), `for` (0.03026), `about` (0.02465), `have` (0.02435), `you` (0.02377), `we` (0.02306)

## 4. Privacy, PII & Responsible AI Considerations

### PII Audit Findings

- **Emails detected in raw sample:** 0
- **Phone numbers detected in raw sample:** 0
- **IP addresses detected in raw sample:** 0
- **Sanitization Pipeline:** `PII scrubbing enabled (regex redaction)`.

### Limitations & Responsible Deployment

1. **Audit Scope:** A balanced class distribution does not imply absence of demographic or organizational bias. Tickets may exhibit subtle stylistic variance across different regional departments.
2. **Explainability Caveat:** Linear coefficients and SHAP values highlight lexical correlations (e.g. keywords like 'outage', 'access', 'invoice'). They do not capture complex cross-sentence dependencies or emotional sentiment.
3. **Human-in-the-Loop Routing:** In high-stakes IT incidents (such as security breaches or enterprise database dropouts), automated routing decisions with confidence < 0.80 MUST trigger human triage rather than silent auto-assignment.
