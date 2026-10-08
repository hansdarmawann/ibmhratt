# Responsible AI

[← Back to README](../README.md)

Sensitive attributes, subgroup behavior, and the conditions any real use would require.

## Responsible AI considerations

Gender, Age, and MaritalStatus are retained in the educational main comparison. A prespecified logistic regression ablation removes all three and reports training CV metrics; its AP difference is in `sensitive_ablation` in the JSON report. This does **not** establish whether those attributes are acceptable for an actual HR application.

[Subgroup metrics](../reports/metrics/subgroup_metrics.csv) report sample counts, positive counts, recall, false-positive rate, precision, and selection rate by Gender, MaritalStatus, and AgeBand. Each rate has a 95% stratified bootstrap interval (1,000 resamples within the group, frozen model and threshold). The intervals are wide for small groups, which is the point: for example, a group with four observed attrition cases cannot support a precise recall estimate. No formal fairness certification is claimed. Excluding sensitive attributes alone cannot eliminate proxies such as job role, compensation, and tenure.

Real use would require a support-oriented purpose, consent/privacy controls, a governance review, human oversight, appropriate fairness definitions, independent validation, and monitoring. No real employee data should be entered into this demo. Model signals must not determine high-impact employment actions.
