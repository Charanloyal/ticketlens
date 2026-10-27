# TicketLens: Model Explainability, Bias & Responsible AI Audit

## 1. Executive Summary

This report details feature attribution, SHAP explainability, dataset label representation, and privacy/PII risk checks conducted on the TicketLens IT support ticket classification system.

## 2. Dataset Distribution & Bias Audit

- **Total Tickets Audited:** 23747
- **Class Imbalance Ratio (Majority / Minority):** 3.77:1
- **Class Distribution:**
  - **Incident:** 9433 samples (39.7%)
  - **Request:** 6879 samples (29.0%)
  - **Problem:** 4932 samples (20.8%)
  - **Change:** 2503 samples (10.5%)

### Ticket Length Profile by Category (Words)

| Category | Mean Word Count | Median Word Count | Min Words | Max Words |
| :--- | :---: | :---: | :---: | :---: |
| Incident | 59.9 | 58 | 2 | 235 |
| Request | 62.5 | 64 | 6 | 257 |
| Problem | 59.2 | 55 | 4 | 276 |
| Change | 66.1 | 72 | 3 | 247 |

## 3. Feature Importance & SHAP Attribution

Top driving n-gram features identified for each IT service category via linear attribution and SHAP values:

### Category: `Change`
**Top Positive Driving Features (TF-IDF Weights):**
`update` (+11.1404), `enhance` (+10.5704), `improve` (+10.5524), `revise` (+8.0808), `boost` (+7.3086), `enhancement` (+7.2992)

**Top SHAP Attributions (|SHAP| impact):**
`problem` (0.13351), `breach` (0.12747), `issue` (0.11658), `updating` (0.11576), `update` (0.11124), `urgent` (0.11007)

### Category: `Incident`
**Top Positive Driving Features (TF-IDF Weights):**
`problem` (+6.2423), `incident` (+6.1783), `breach` (+5.6523), `issue` (+5.3706), `has` (+5.0011), `issues` (+4.5704)

**Top SHAP Attributions (|SHAP| impact):**
`breach` (0.16815), `issue` (0.15655), `problem` (0.15215), `have` (0.11351), `issues` (0.10557), `incident` (0.09492)

### Category: `Problem`
**Top Positive Driving Features (TF-IDF Weights):**
`problem` (+5.9437), `facing` (+5.7914), `issue` (+5.3167), `issues` (+4.58), `problem with` (+4.5201), `despite` (+3.7879)

**Top SHAP Attributions (|SHAP| impact):**
`issue` (0.15498), `problem` (0.14487), `issues` (0.10579), `for` (0.10026), `breach` (0.09117), `problems` (0.07815)

### Category: `Request`
**Top Positive Driving Features (TF-IDF Weights):**
`for` (+6.2853), `details` (+6.0948), `strategies` (+5.3317), `integrating` (+5.1867), `guidance` (+5.0588), `information` (+4.8044)

**Top SHAP Attributions (|SHAP| impact):**
`issue` (0.19495), `problem` (0.16351), `issues` (0.13757), `breach` (0.13186), `for` (0.12018), `problems` (0.10617)

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
