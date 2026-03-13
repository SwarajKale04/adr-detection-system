# ADR Detection System

A clean, modular Python-based system for detecting Adverse Drug Reactions (ADRs) from static drug review datasets. This project processes reviews, analyzes sentiment, detects symptoms, and validates them against Indian medicine reference data.

## Overview

This is a final-year academic project designed for static NLP-based ADR detection without real-time streaming, cloud services, or heavy machine learning training. The system uses pretrained sentiment models from HuggingFace for efficient and reliable analysis.

## Project Structure

```
adr-detection-system/
├── backend/
│   ├── data/
│   │   ├── DS1_WebMD_Reviews.xlsx          # WebMD drug reviews dataset
│   │   ├── DS2_IndianSubstitutes.xlsx      # Drug to Indian equivalent mapping
│   │   └── DS3_MedicineDataset.csv         # Medical reference dataset
│   │
│   ├── modules/
│   │   ├── __init__.py                     # Package initialization
│   │   ├── data_loader.py                  # Dataset loading utilities
│   │   ├── drug_mapper.py                  # Drug name mapping & normalization
│   │   ├── sentiment_analyzer.py           # Sentiment classification
│   │   ├── symptom_detector.py             # ADR symptom extraction
│   │   ├── negation_handler.py             # Negation & modifier handling
│   │   ├── validator.py                    # ADR validation
│   │   └── pipeline.py                     # Main processing pipeline
│   │
│   ├── output/
│   │   └── results.json                    # Structured ADR detection results
│   │
│   ├── main.py                             # Entry point script
│   └── requirements.txt                    # Python dependencies
│
└── README.md                               # This file
```

## Features

### Module Descriptions

- **data_loader.py**: Loads WebMD reviews, Indian medicine substitutes, and medicine reference data
- **drug_mapper.py**: Maps drug names, normalizes them, and handles fuzzy matching with Indian equivalents
- **sentiment_analyzer.py**: Uses pretrained HuggingFace models for sentiment classification
- **symptom_detector.py**: Extracts potential ADR symptoms from review text
- **negation_handler.py**: Handles negations and modifiers to prevent false positives
- **validator.py**: Cross-references detected ADRs with medical literature
- **pipeline.py**: Orchestrates the complete workflow

### Key Characteristics

✓ **Static Processing**: No real-time streaming or event-based processing  
✓ **No Kafka/Cloud**: Runs entirely locally without external dependencies  
✓ **No Docker**: Simple Python environment, easy to set up  
✓ **No Heavy Training**: Uses pretrained models only  
✓ **Modular Design**: Each component is independent and testable  
✓ **Academic Ready**: Clean structure suitable for thesis/research  

## Requirements

- Python 3.8+
- pandas
- numpy
- transformers
- torch
- nltk
- scikit-learn
- tqdm
- openpyxl

## Installation

1. Clone or create the project directory
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

## Usage

1. Place your datasets in the `backend/data/` directory:
   - `DS1_WebMD_Reviews.xlsx`
   - `DS2_IndianSubstitutes.xlsx`
   - `DS3_MedicineDataset.csv`

2. Run the pipeline:
   ```bash
   cd backend
   python main.py
   ```

3. Results will be saved to `output/results.json`

## Output Format

The system generates structured JSON output with the following format:

```json
{
  "summary": {
    "total_reviews_processed": 0,
    "total_symptoms_detected": 0,
    "validated_adrs": 0,
    "execution_time": 0.0
  },
  "results": [
    {
      "drug": "Drug Name",
      "detected_symptoms": [],
      "sentiment": {},
      "validated_adrs": []
    }
  ]
}
```

## Development Notes

- Each module follows a clear interface pattern for easy implementation
- All modules include comprehensive docstrings
- The pipeline orchestrates execution flow with error handling
- Logging is configured for debugging and monitoring

## Implementation Steps (Future)

1. Implement `DataLoader` to read Excel and CSV files
2. Build drug name normalization and fuzzy matching in `DrugMapper`
3. Integrate HuggingFace sentiment model in `SentimentAnalyzer`
4. Create symptom lexicon and extraction logic in `SymptomDetector`
5. Implement negation scoping in `NegationHandler`
6. Build validation logic in `Validator`
7. Complete pipeline orchestration in `ADRDetectionPipeline`

## License

Academic Project

## Author

[Your Name/Team]
