# TicketLens: Model Explainability, Bias & Responsible AI Audit

## 1. Executive Summary

This report details feature attribution, SHAP explainability, dataset label representation, and privacy/PII risk checks conducted on the TicketLens IT support ticket classification system.

## 2. Dataset Distribution & Bias Audit

- **Total Tickets Audited:** 210
- **Class Imbalance Ratio (Majority / Minority):** 3.77:1
- **Class Distribution:**
  - **Incident:** 83 samples (39.5%)
  - **Request:** 61 samples (29.0%)
  - **Problem:** 44 samples (21.0%)
  - **Change:** 22 samples (10.5%)

### Ticket Length Profile by Category (Words)

| Category | Mean Word Count | Median Word Count | Min Words | Max Words |
| :--- | :---: | :---: | :---: | :---: |
| Incident | 59.2 | 55 | 7 | 145 |
| Request | 62.1 | 55 | 8 | 257 |
| Problem | 65.0 | 53 | 7 | 162 |
| Change | 87.8 | 93 | 16 | 152 |

## 3. Feature Importance & SHAP Attribution

Top driving n-gram features identified for each IT service category via linear attribution and SHAP values:

### Category: `Change`
**Top Positive Driving Features (TF-IDF Weights):**
`request` (+0.9382), `current` (+0.9103), `upgrade` (+0.8474), `analytics tools` (+0.7795), `request for` (+0.7327), `as` (+0.7291)

**Top SHAP Attributions (|SHAP| impact):**
`as` (0.02403), `request` (0.02125), `for` (0.02058), `data` (0.01996), `security` (0.01791), `problem` (0.01722)

### Category: `Incident`
**Top Positive Driving Features (TF-IDF Weights):**
`system` (+0.7715), `issue` (+0.7059), `the issue` (+0.5922), `unexpected` (+0.5879), `resolve` (+0.5858), `problem` (+0.5609)

**Top SHAP Attributions (|SHAP| impact):**
`for` (0.0247), `issue` (0.02421), `medical` (0.02228), `the issue` (0.0195), `problem` (0.0186), `you` (0.01796)

### Category: `Problem`
**Top Positive Driving Features (TF-IDF Weights):**
`server` (+0.746), `been` (+0.673), `the server` (+0.6357), `during` (+0.6344), `problem with` (+0.6231), `crashes` (+0.6006)

**Top SHAP Attributions (|SHAP| impact):**
`been` (0.02631), `server` (0.02217), `for` (0.01946), `problem` (0.01788), `might be` (0.01785), `you` (0.01664)

### Category: `Request`
**Top Positive Driving Features (TF-IDF Weights):**
`on` (+0.8492), `you` (+0.7797), `could you` (+0.7484), `information on` (+0.7184), `about` (+0.7063), `inquiry` (+0.7049)

**Top SHAP Attributions (|SHAP| impact):**
`on` (0.03251), `you` (0.02482), `for` (0.02359), `issue` (0.02095), `information` (0.02075), `we` (0.02065)

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
