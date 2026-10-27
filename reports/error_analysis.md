# TicketLens: Precision / Recall Error Analysis

All numbers are on the identical held-out test set (3,563 tickets).

## TF-IDF + Logistic Regression

Errors: **618** (17.3% of test tickets). Macro-F1 **0.8352**.

**Most frequent confusions (true -> predicted):**

| True | Predicted | Count | Share of errors |
| :--- | :--- | ---: | ---: |
| Incident | Problem | 303 | 49.0% |
| Problem | Incident | 279 | 45.1% |
| Change | Request | 13 | 2.1% |
| Request | Change | 13 | 2.1% |
| Change | Incident | 4 | 0.7% |
| Incident | Change | 3 | 0.5% |

**Confidence gate (auto-accept) precision vs. coverage:**

| Threshold | Auto-accepted | Precision of auto-accepted |
| :---: | :---: | :---: |
| 0.5 | 96.9% | 83.7% |
| 0.6 | 82.4% | 87.7% |
| 0.7 | 68.5% | 92.0% |
| 0.8 | 52.6% | 96.0% |
| 0.9 | 36.1% | 98.8% |

**Performance parity by ticket length (bias check):**

| Length | Tickets | Accuracy | Macro-F1 |
| :--- | ---: | :---: | :---: |
| < 50 words | 1,423 | 0.8313 | 0.8483 |
| 50-99 words | 1,838 | 0.8341 | 0.8360 |
| 100-199 words | 296 | 0.7568 | 0.7578 |
| >= 200 words | 6 | 0.8333 | 0.5000 |

**Most confident errors:**

| True | Predicted | Confidence | Ticket (truncated) |
| :--- | :--- | :---: | :--- |
| Problem | Incident | 0.97 | Report of Data Security Concern  Dear Customer Support,<br><br>The healthcare organization encountered data security problems impacting several systems, possibly due to recent configuration updates. Rebooting affected de... |
| Problem | Incident | 0.95 | Report of Data Security Concerns  Dear Customer Support,<br><br>Our healthcare organization has encountered data security challenges impacting several systems, likely due to recent configuration updates. Rebooting affect... |
| Change | Request | 0.94 | Enhance Investment Strategies with Machine Learning  Respected customer support, I am reaching out to seek information on leveraging machine learning for optimizing investment portfolios through real-time market data and... |
| Incident | Problem | 0.94 | Customer Concern  A healthcare provider faced unauthorized access attempts to medical data, possibly due to outdated security protocols. They have attempted to update firewalls and are conducting employee training on dat... |
| Problem | Incident | 0.93 | Identified Error in Payment Processing System  We are encountering problems with the billing system, which are hindering our ability to process payments. This may be related to a recent software update. We have attempted... |
| Problem | Incident | 0.93 | Technical Difficulty  The marketing agency's website has been crashing frequently during traffic spikes, probably due to inadequate server capacity. Attempts to restart the server and clear the cache have not resolved th... |

## Keras CNN (TensorFlow)

Errors: **754** (21.2% of test tickets). Macro-F1 **0.8095**.

**Most frequent confusions (true -> predicted):**

| True | Predicted | Count | Share of errors |
| :--- | :--- | ---: | ---: |
| Incident | Problem | 491 | 65.1% |
| Problem | Incident | 213 | 28.2% |
| Change | Request | 19 | 2.5% |
| Request | Change | 12 | 1.6% |
| Change | Incident | 4 | 0.5% |
| Problem | Request | 4 | 0.5% |

**Confidence gate (auto-accept) precision vs. coverage:**

| Threshold | Auto-accepted | Precision of auto-accepted |
| :---: | :---: | :---: |
| 0.5 | 99.6% | 78.9% |
| 0.6 | 86.5% | 82.4% |
| 0.7 | 73.5% | 86.8% |
| 0.8 | 60.8% | 91.7% |
| 0.9 | 47.0% | 96.8% |

**Performance parity by ticket length (bias check):**

| Length | Tickets | Accuracy | Macro-F1 |
| :--- | ---: | :---: | :---: |
| < 50 words | 1,423 | 0.7723 | 0.8039 |
| 50-99 words | 1,838 | 0.8025 | 0.8174 |
| 100-199 words | 296 | 0.7736 | 0.7774 |
| >= 200 words | 6 | 1.0000 | 1.0000 |

**Most confident errors:**

| True | Predicted | Confidence | Ticket (truncated) |
| :--- | :--- | :---: | :--- |
| Change | Request | 1.00 | Enhance Investment Strategies with Machine Learning  Respected customer support, I am reaching out to seek information on leveraging machine learning for optimizing investment portfolios through real-time market data and... |
| Change | Request | 1.00 | Support Required for Data Analytics Integration with JIRA and Zoho Books  I am contacting you to seek support for integrating data analytics with JIRA and Zoho Books to optimize investments on Ubuntu systems. I need deta... |
| Problem | Request | 1.00 | Question About Marketing Agency Software Compatibility  Dear Support Team,\n\nI hope this message reaches you well. I am reaching out to ask about the compatibility of your products with the specific needs of marketing a... |
| Change | Request | 1.00 | Enhancement of Digital Infrastructure  Hello Customer Support, I'm reaching out to explore the possibilities of upgrading our digital infrastructure to support our brand's growth and optimize our marketing strategy. We a... |
| Request | Change | 0.99 | Protocols for Ensuring Medical Data Security  We highly value the security of medical data and advise the use of encryption and two-factor authentication to safeguard it. Additionally, we recommend maintaining regular up... |
| Problem | Incident | 0.99 | Device Malfunction Threatening Data Security  A healthcare provider encountered several device malfunctions that jeopardized the security of medical data. The devices were rebooted and their software was updated. |

## DistilBERT + LoRA (PyTorch/PEFT)

Errors: **741** (20.8% of test tickets). Macro-F1 **0.7790**.

**Most frequent confusions (true -> predicted):**

| True | Predicted | Count | Share of errors |
| :--- | :--- | ---: | ---: |
| Problem | Incident | 486 | 65.6% |
| Incident | Problem | 208 | 28.1% |
| Request | Change | 19 | 2.6% |
| Change | Request | 11 | 1.5% |
| Change | Problem | 4 | 0.5% |
| Incident | Request | 4 | 0.5% |

**Confidence gate (auto-accept) precision vs. coverage:**

| Threshold | Auto-accepted | Precision of auto-accepted |
| :---: | :---: | :---: |
| 0.5 | 99.8% | 79.3% |
| 0.6 | 82.6% | 85.0% |
| 0.7 | 67.6% | 90.0% |
| 0.8 | 54.8% | 94.4% |
| 0.9 | 44.2% | 97.8% |

**Performance parity by ticket length (bias check):**

| Length | Tickets | Accuracy | Macro-F1 |
| :--- | ---: | :---: | :---: |
| < 50 words | 1,423 | 0.7772 | 0.7817 |
| 50-99 words | 1,838 | 0.8036 | 0.7757 |
| 100-199 words | 296 | 0.7905 | 0.7718 |
| >= 200 words | 6 | 0.8333 | 0.5000 |

**Most confident errors:**

| True | Predicted | Confidence | Ticket (truncated) |
| :--- | :--- | :---: | :--- |
| Request | Change | 1.00 | Enhance Ubiquiti UniFi Dream Machine for better digital marketing strategies and improved network performance and security. |
| Request | Change | 0.99 | Protocols for Ensuring Medical Data Security  We highly value the security of medical data and advise the use of encryption and two-factor authentication to safeguard it. Additionally, we recommend maintaining regular up... |
| Change | Request | 0.99 | Support for Enhancing Data Analytics with Moodle and Azure  I am contacting you to explore ways of enhancing data analytics integration with Moodle and Azure to derive investment insights. Could you share more details on... |
| Request | Change | 0.98 | Enhance digital strategies by integrating Jenkins and Redis to promote brand development. |
| Problem | Incident | 0.98 | Concern Regarding Data Integration System  Faced issues with the data integration system, which has impacted the performance of our investment analytics tools. The disruption might be due to recent software updates or co... |
| Request | Change | 0.98 | Improved Support for Shopify KNIME Marketing Analytics Integration  Hello, we are seeking assistance with integrating Shopify and KNIME for enhanced marketing analytics. We would like to know the detailed steps involved ... |

## Served TF-IDF (all data, model only)

Errors: **459** (12.9% of test tickets). Macro-F1 **0.8782**.

**Most frequent confusions (true -> predicted):**

| True | Predicted | Count | Share of errors |
| :--- | :--- | ---: | ---: |
| Incident | Problem | 232 | 50.5% |
| Problem | Incident | 201 | 43.8% |
| Change | Request | 10 | 2.2% |
| Request | Change | 7 | 1.5% |
| Change | Incident | 2 | 0.4% |
| Incident | Change | 2 | 0.4% |

**Confidence gate (auto-accept) precision vs. coverage:**

| Threshold | Auto-accepted | Precision of auto-accepted |
| :---: | :---: | :---: |
| 0.5 | 99.1% | 87.4% |
| 0.6 | 87.9% | 91.1% |
| 0.7 | 77.1% | 94.2% |
| 0.8 | 65.0% | 96.4% |
| 0.9 | 50.0% | 98.7% |

**Performance parity by ticket length (bias check):**

| Length | Tickets | Accuracy | Macro-F1 |
| :--- | ---: | :---: | :---: |
| < 50 words | 1,423 | 0.8665 | 0.8796 |
| 50-99 words | 1,838 | 0.8814 | 0.8833 |
| 100-199 words | 296 | 0.8311 | 0.8364 |
| >= 200 words | 6 | 0.8333 | 0.5000 |

**Most confident errors:**

| True | Predicted | Confidence | Ticket (truncated) |
| :--- | :--- | :---: | :--- |
| Incident | Problem | 0.96 | Customer Concern  A healthcare provider faced unauthorized access attempts to medical data, possibly due to outdated security protocols. They have attempted to update firewalls and are conducting employee training on dat... |
| Problem | Incident | 0.96 | Report of Data Security Concern  Dear Customer Support,<br><br>The healthcare organization encountered data security problems impacting several systems, possibly due to recent configuration updates. Rebooting affected de... |
| Problem | Incident | 0.95 | Cybersecurity Breach Affecting Medical Data Systems  Dear Customer Support,<br><br>I am submitting a report concerning a cybersecurity breach that has compromised the hospital's medical data systems. The incident may hav... |
| Problem | Incident | 0.95 | Security Incident Involving Medical Records  A security incident has compromised medical records; potential malware infection and unauthorized access detected. System scans and password resets have been attempted. |
| Change | Request | 0.95 | Enhance Investment Strategies with Machine Learning  Respected customer support, I am reaching out to seek information on leveraging machine learning for optimizing investment portfolios through real-time market data and... |
| Problem | Incident | 0.94 | System Downtime  Marketing agency facing system outages on various platforms, hindering progress on brand growth initiatives due to server issues and software incompatibility. |

## Served TF-IDF + ITIL cues (production)

Errors: **517** (14.5% of test tickets). Macro-F1 **0.8635**.

**Most frequent confusions (true -> predicted):**

| True | Predicted | Count | Share of errors |
| :--- | :--- | ---: | ---: |
| Incident | Problem | 292 | 56.5% |
| Problem | Incident | 185 | 35.8% |
| Change | Request | 15 | 2.9% |
| Request | Change | 9 | 1.7% |
| Problem | Change | 6 | 1.2% |
| Change | Incident | 2 | 0.4% |

**Confidence gate (auto-accept) precision vs. coverage:**

| Threshold | Auto-accepted | Precision of auto-accepted |
| :---: | :---: | :---: |
| 0.5 | 98.8% | 86.0% |
| 0.6 | 87.7% | 89.4% |
| 0.7 | 77.5% | 92.5% |
| 0.8 | 66.1% | 94.8% |
| 0.9 | 51.7% | 97.4% |

**Performance parity by ticket length (bias check):**

| Length | Tickets | Accuracy | Macro-F1 |
| :--- | ---: | :---: | :---: |
| < 50 words | 1,423 | 0.8651 | 0.8795 |
| 50-99 words | 1,838 | 0.8536 | 0.8587 |
| 100-199 words | 296 | 0.8108 | 0.8108 |
| >= 200 words | 6 | 1.0000 | 1.0000 |

**Most confident errors:**

| True | Predicted | Confidence | Ticket (truncated) |
| :--- | :--- | :---: | :--- |
| Change | Request | 0.99 | Enhance Investment Strategies with Machine Learning  Respected customer support, I am reaching out to seek information on leveraging machine learning for optimizing investment portfolios through real-time market data and... |
| Incident | Problem | 0.98 | Support for Digital Campaign  Our marketing campaign aimed at digital growth is underperforming, resulting in reduced brand visibility. This might be due to a lack of alignment with the target audience's preferences in o... |
| Change | Request | 0.98 | Secure Data Storage Systems  Hello Customer Support, I am contacting you to seek details on the recent updates concerning secure medical data storage and transmission methods for hospital IT infrastructure and equipment.... |
| Incident | Problem | 0.97 | Troubleshooting Connectivity Issues in Project Management SaaS  Recently, users have been encountering connectivity problems with the project management SaaS platform. These issues might be due to network instability or ... |
| Change | Request | 0.97 | Support Required for Data Analytics Integration with JIRA and Zoho Books  I am contacting you to seek support for integrating data analytics with JIRA and Zoho Books to optimize investments on Ubuntu systems. I need deta... |
| Problem | Incident | 0.97 | Problem Encountered with Cloud Platform Today  The cloud platform experienced a failure, leading to data loss and an outage. Restarting the services enhanced the capacity. |

## Takeaways

- Most errors across all models are **Incident <-> Problem** confusions: both describe something broken; ITIL separates them by recurrence / root-cause intent, which the synthetic labels do not apply consistently.
- Request and Change are separated with high precision and recall by every model.
- The confidence gate trades coverage for precision; at 0.8 the auto-accepted tickets are far more accurate than the overall accuracy, which is why lower-confidence tickets are routed to human review.
