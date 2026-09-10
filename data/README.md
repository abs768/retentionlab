# Dataset provenance and redistribution

The pipeline uses IBM's fictional Telco Customer Churn sample from
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d).
The source CSV is `data/Telco-Customer-Churn.csv` at upstream revision
`d5371f5d83a446ad5673cbcca3b814b926491f8a`.

The [upstream README's license section](https://github.com/IBM/telco-customer-churn-on-icp4d#license)
licenses the **code pattern** under Apache 2.0 and notes separate third-party rights.
The data directory contains a CSV without a separate dataset license. This review did not
establish an explicit, dataset-specific redistribution grant. Conservatively, the raw CSV
is no longer tracked in the current tree. Download it for local reproduction using
`python scripts/download_data.py`; that script pins the upstream revision and verifies
SHA256 against the original audited dataset. It never overwrites mismatching local data.

The MIT license at the project root covers original project code, not IBM's dataset.
Derived model/report artifacts do not establish independent data redistribution rights.
The original Git commit still contains the CSV. History was preserved as requested; before
publishing that history, resolve data rights or explicitly authorize a history cleanup.
Nothing has been pushed by this extension.
