# Adverse Drug Reaction (ADR) Detection System

A Python project for finding possible adverse drug reactions in patient-written medication reviews. It combines drug-name mapping, symptom matching, negation handling, and sentiment analysis to produce structured results that can be reviewed and explored further.

Built as an academic project, it demonstrates a practical NLP and data-processing workflow: from loading review and reference data to exporting detections and summary analytics.

## What it does

For each review, the pipeline:

1. Normalizes the drug name and maps it to a corresponding Indian generic where a substitute is available.
2. Looks for known side effects using the medicine reference data.
3. Filters symptoms that appear in a negated context.
4. Analyzes review sentiment and records sentiment counts by drug.
5. Exports possible ADRs and sentiment summaries as JSON.

The project includes rule-based sentiment analysis and optional MLP and transformer-based sentiment paths. The transformer configuration currently uses `distilbert-base-uncased-finetuned-sst-2-english`; despite the analyzer module's BioBERT naming, this is a general English sentiment model, not a biomedical model.

## Technical highlights

- **Python data pipeline:** pandas-based loading and processing of review and medicine data.
- **NLP features:** sentiment classification, symptom vocabulary matching, and rule-based negation filtering.
- **Drug mapping:** normalization and substitute lookup to relate brand names to generic medicines.
- **Modular design:** separate modules for loading, mapping, detection, sentiment, validation, and analytics.
- **Structured output:** JSON files that can support later analysis or visualization.

## Project results

The original project notes report a run over 43,762 reviews, with 4,024 possible ADR detections across 33 drugs. These are historical project figures, not a benchmark: the source datasets are not included here, and the results have not been independently reproduced from this repository. Detection counts should be treated as candidates for review, not confirmed medical findings.

## Getting started

### Requirements

- Python 3.8 or later
- The packages listed in [`backend/requirements.txt`](backend/requirements.txt)
- The input datasets listed below (not included in this repository)

### Install dependencies

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

### Add the input data

Create `backend/data/` and place the required files there:

- `DS1_WebMD_Reviews.xlsx` — review data
- `DS2_IndianSubstitutes.xlsx` — drug substitute mappings
- `DS3_MedicineDataset.csv` — medicine and side-effect reference data

These datasets are excluded from the repository. The pipeline will not run until the files are available at these paths.

### Run the pipeline

From the repository root:

```bash
python main.py
```

Optional sentiment modes exposed by the command-line interface:

```bash
python main.py --mlp
python main.py --biobert
```

The transformer model may need to download the first time it is used. The `--mlp` option depends on a trained model being available; `--train-mlp` runs the training step before the pipeline:

```bash
python main.py --train-mlp
```

## Output

A run writes results under `backend/output/`:

- `adr_results.json` — possible ADR detections, including drug, symptom, review text, sentiment, and a heuristic confidence score.
- `drug_sentiment_summary.json` — positive and negative review counts and a small number of example reviews for each drug.

The confidence value is based on simple rules in the pipeline; it is not a calibrated probability or a clinical risk score.

## Repository layout

```text
.
├── backend/
│   ├── data/                 # Required datasets (not included)
│   ├── modules/              # Loading, mapping, NLP, pipeline, and analytics code
│   ├── output/               # Generated JSON results
│   └── requirements.txt
├── main.py                   # Main command-line entry point
├── nlp_demonstration.py      # Sentiment approach demonstration
├── train_mlp_model.py        # MLP training script
└── test_*.py                 # Test and demonstration scripts
```

## Scope and limitations

This is an academic demonstration, not a medical device or a clinical decision-support system. It relies on lexicons, rules, and sentiment models; it can miss reactions, flag unrelated text, or misread context. The supplied confidence heuristic and reported counts are not clinically validated. Do not use its output to make treatment decisions.

## License and data

No license or dataset redistribution terms are specified in this repository. Check the source dataset terms and obtain permission before sharing or reusing the data. Add a repository license if you intend to publish the code for reuse.
