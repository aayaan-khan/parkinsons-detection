# Parkinson's Disease Voice Detection

This project explores whether acoustic voice measurements can distinguish people with Parkinson's disease (PD) from healthy controls. It is an educational data-science internship project using the UCI Oxford Parkinsons voice dataset; it does not diagnose or provide medical advice.

## Dataset

The primary UCI Parkinsons (Oxford) file contains 195 recordings from 32 subjects: 24 with PD and 8 healthy controls. There are 147 PD recordings and 48 healthy recordings (75.4% PD). The UCI dataset page describes 31 people / 23 PD, but the supplied file contains 32 / 24; the file is the source of truth and this discrepancy is documented in the notebook.

The secondary UCI Parkinsons Telemonitoring dataset contains 5,875 recordings from 42 PD patients and no healthy controls. It is reserved for regression of motor and total UPDRS scores, not disease detection, and will not be merged with the Oxford dataset because the feature sets differ.

## Approach

The notebook demonstrates the inflated performance possible with a naive random split, then uses subject-grouped validation to prevent a person's recordings from crossing train/test boundaries. Evaluation prioritizes recall, reports calibration, and uses permutation importance and model coefficients for explainability. Model results remain TBD until measured in the notebook.

## Results

| Model | Random-split score | Grouped-CV score | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Repository structure

```text
parkinsons-detection/
├── app/                         # Streamlit demo
├── data/
│   ├── raw/                     # UCI source data
│   └── processed/               # Generated, gitignored data
├── docs/                        # Project decisions
├── models/                      # Generated, gitignored models
├── notebooks/                   # Self-contained graded notebook (.ipynb)
├── reports/                     # Figures and reports
├── src/                         # Demo and test support
├── submission/                  # Notebook HTML export
├── tests/
├── requirements.txt
└── README.md
```

## How to run

From the repository root, create and activate the project environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open and run the notebook in `notebooks/` (or launch Jupyter with `jupyter notebook notebooks/`). Start the demo with:

```powershell
streamlit run app/app.py
```

## Limitations and disclaimer

There are only 32 subjects, including 8 healthy controls, so subject-grouped validation results are uncertain and should not be generalized without further data. Recordings from a person are correlated, making subject grouping essential. The secondary Telemonitoring dataset has PD patients only and cannot provide healthy controls for detection. **This is a research project, not a diagnostic tool.** It must not be used to make medical decisions.

## Citations

- Little, M. A. et al. (2008). Reference for the Oxford Parkinsons voice dataset. [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons).
- Tsanas, A. et al. (2009/2010). Reference for non-invasive speech-test telemonitoring of Parkinson's disease progression. [UCI Parkinsons Telemonitoring dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).

## License

The UCI datasets are distributed under CC BY 4.0. Cite the original work and dataset pages when reusing them.