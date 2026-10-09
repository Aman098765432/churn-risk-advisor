## Model card: Churn Risk Advisor v1.0
**Intended use:** rank existing telecom customers by churn risk so a
retention team can prioritize calls. Decision support, not automation.
**Not for:** credit, pricing, or any decision that denies a service.
**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.
**Model:** [name], chosen by 5-fold CV. Features: 30 encoded columns.
**Performance:** CV AUC [X +/- s]; test AUC [X] (test set used once).
**Threshold:** [t], from a cost analysis (FN = PKR 6,000, FP = PKR 1,000).
**Limitations:** one US dataset from one period; no Pakistan data;
associations, not causes; performance may drift as plans change.
**Fairness check (Week 15):** compare recall across gender and seniors.
**Owner and version:** [your name], v1.0, [date].