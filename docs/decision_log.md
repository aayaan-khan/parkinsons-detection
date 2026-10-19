# Decision Log

| # | Decision | Why (viva answer) |
|---|---|---|
| D1 | Primary dataset: UCI Parkinsons voice (binary) | Has healthy controls, so it supports detection; standard benchmark so results are comparable. |
| D2 | Telemonitoring is secondary, regression only | No healthy controls, so classification would be invalid; feature set also differs. |
| D3 | Grouping key = subject ID parsed from name | Recordings from one person are not independent samples. |
| D4 | The notebook is the single graded artifact; src/ only supports the demo | Reviewer receives only .ipynb and .html. |
| D5 | Subject count is 32 (24 PD, 8 healthy), verified from the file | UCI text says 31/23; the notebook asserts the real numbers and notes the discrepancy. |
| D6 | Majority-class baseline is 75.4% accuracy | A trivial always-PD model scores this, so accuracy alone is meaningless. |
| D7 | Grouped CV uses repeated StratifiedGroupKFold, pooled out-of-fold predictions and a subject-level bootstrap CI | Only 8 healthy subjects, so each test fold has 1-2 of them and single-run metrics are noisy. |

## Results log