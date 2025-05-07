# CaliForest: Calibrated Random Forest for Healthcare Applications

CaliForest is a novel implementation of calibrated random forests that achieves both high discrimination and accurate calibration for healthcare predictions. Unlike traditional approaches, CaliForest utilizes out-of-bag samples to avoid explicit calibration sets, making it particularly valuable for healthcare applications where data may be limited.

## Overview

CaliForest addresses a critical gap in healthcare predictive modeling by focusing on both discrimination (ability to differentiate between high and low risk events) and calibration (accuracy of risk estimates). The project demonstrates improved calibration across multiple metrics while maintaining the discriminative power of traditional random forests.

## Installation

1. Create and activate a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Setup

### MIMIC-III Dataset
This project uses the MIMIC-III database for evaluation. To access the data:

1. Obtain access to MIMIC-III database through [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Complete the required training course
3. Download the MIMIC-III files from [MIMIC_Extract](https://github.com/MLforHealth/MIMIC_Extract)
4. Down pre-processed `all hourly data.h5` under `data/` from [here](https://console.cloud.google.com/storage/browser/mimic_extract)


## Project Structure

```
.
├── califorest/             # Main package directory
├── data/                   # Data directory (create this)
├── results/                # Results output directory
├── main.py                 # Main execution script
├── mimic_extract.py        # MIMIC-III data processing
├── preprocess_mimic.py     # MIMIC preprocessing utilities
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Usage

### Basic Example
First run the preprocess_mimic.py to preprocess the data.
```
python preprocess_mimic.py
```

Then run the main.py to train the model.
```
python main.py
```

The script supports different datasets and configurations:
- Hospital mortality prediction
- ICU mortality prediction
- Length of stay prediction (3-day and 7-day thresholds)


## Model Variants

CaliForest offers several variants:
- `CF-Iso`: Isotonic calibration
- `CF-Logit`: Logistic calibration
- `CF-Iso-NoWeight`: Isotonic calibration without OOB weights (ablation study)
- `CF-Logit-NoWeight`: Logistic calibration without OOB weights (ablation study)
- `RC-Iso`: Reference class with isotonic calibration
- `RC-Logit`: Reference class with logistic calibration
- `RF-NoCal`: Standard random forest without calibration

## Evaluation Metrics

The model evaluates performance using:
- ROC AUC score (discrimination)
- Hosmer-Lemeshow test (calibration)
- Spiegelhalter's Z-test (calibration)
- Scaled Brier score
- Reliability metrics

## Citation

```latex
\begin{thebibliography}{9}

\bibitem{califorest}
Park, Y., \& Ho, J. C. (2020). \emph{CaliForest: Calibrated Random Forest for Health Data}. In \emph{Proceedings of the ACM Conference on Health Inference and Learning} (pp. 40--50). \url{https://doi.org/10.1145/3368555.3384461}

\bibitem{pmc3575184}
Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., \& Kattan, M. W. (2010). Assessing the performance of prediction models: A framework for traditional and novel measures. \emph{Epidemiology, 21}(1), 128--138. \url{https://doi.org/10.1097/EDE.0b013e3181c30fb2}.

\bibitem{stata_brier}
StataCorp. (n.d.). \emph{Brier score decomposition}. College Station, TX: Stata Press. \url{https://www.stata.com/manuals/rbrier.pdf}.

\bibitem{mimicextract}
Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, and Marzyeh Ghassemi. \emph{MIMIC-Extract: A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III}. arXiv:1907.08322. \url{https://arxiv.org/abs/1907.08322}

\end{thebibliography}
```




