# ADR Detection System

A comprehensive system for detecting Adverse Drug Reactions (ADRs) from patient review datasets using **multiple NLP approaches**. This academic project processes 43,762 WebMD reviews to identify medication side effects and safety signals using rule-based, traditional ML, and modern transformer-based techniques.

## 🤖 **NLP Capabilities Demonstrated**

### **Rule-Based NLP** (Primary)
- Keyword matching for sentiment analysis
- String matching for symptom detection
- Negation handling and validation
- **4,024 ADRs detected** across 33 drugs

### **Traditional ML NLP** (MLP)
- **Multi-Layer Perceptron** classifier with TF-IDF features
- **79.5% accuracy** on sentiment classification
- Scikit-learn implementation

### **Modern NLP** (BioBERT) ⭐
- **BioBERT transformer** for medical text understanding
- Domain-specific sentiment analysis
- State-of-the-art contextual understanding
- Advanced NLP techniques demonstration

## 📊 **Key Results**
- **4,024 ADRs detected** from 43,762 patient reviews
- **33 unique drugs** with adverse reactions identified
- **44 different symptoms** detected (nausea, anxiety, headache, etc.)
- **Average confidence**: 0.82 for ADR detections
- **Sentiment tracking** for all 142 processed drugs

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Required packages: `pip install -r backend/requirements.txt`

### Usage Options

#### Rule-Based Pipeline (Default)
```bash
python main.py
```

#### Traditional ML Pipeline (MLP)
```bash
# Train MLP model first
python main.py --train-mlp

# Run with MLP sentiment analysis
python main.py --mlp
```

#### Modern NLP Pipeline (BioBERT) ⭐
```bash
# Run with BioBERT sentiment analysis
python main.py --biobert
```

#### Comprehensive NLP Demonstration
```bash
# Compare all three approaches
python nlp_demonstration.py

# Test BioBERT specifically
python test_biobert.py
```

## 📁 **Project Structure**

```
adr-detection-system/
├── backend/
│   ├── data/                          # Dataset files (not in repo)
│   │   ├── DS1_WebMD_Reviews.xlsx     # 43,762 patient reviews
│   │   ├── DS2_IndianSubstitutes.xlsx # Drug name mappings
│   │   └── DS3_MedicineDataset.csv    # Medicine reference (248K entries)
│   │
│   ├── modules/                       # Core processing modules
│   │   ├── data_loader.py             # Dataset loading utilities
│   │   ├── drug_mapper.py             # Drug name normalization
│   │   ├── sentiment_analyzer.py      # Rule-based sentiment analysis
│   │   ├── mlp_sentiment_analyzer.py  # 🤖 Traditional ML sentiment analysis
│   │   ├── biobert_sentiment_analyzer.py # 🧠 Modern NLP sentiment analysis
│   │   ├── symptom_detector.py        # Symptom extraction
│   │   ├── negation_handler.py        # Negation handling
│   │   ├── pipeline.py                # Main processing pipeline
│   │   ├── analytics.py               # Statistics & reporting
│   │   └── validator.py               # ADR validation
│   │
│   ├── output/                        # Results and exports
│   │   ├── adr_results.json           # Complete ADR detections
│   │   └── drug_sentiment_summary.json # Sentiment analysis
│   │
│   ├── main.py                        # Entry point
│   └── requirements.txt               # Dependencies
│
├── train_mlp_model.py                 # 🤖 MLP training script
├── nlp_demonstration.py               # 🎭 NLP comparison demo
├── test_biobert.py                    # 🧠 BioBERT test script
├── test_*.py                          # Additional test scripts
└── README.md
```

## 🧠 **Architecture**

### Rule-Based Approach
- **No machine learning training** - uses keyword matching and string operations
- **Deterministic results** - same input always produces same output
- **Explainable AI** - easy to understand why decisions are made
- **Fast processing** - no GPU requirements, runs on CPU

### Processing Pipeline
1. **Data Loading**: Load WebMD reviews and medicine reference data
2. **Drug Mapping**: Normalize drug names and map to Indian equivalents
3. **Sentiment Analysis**: Classify review sentiment (positive/negative)
4. **Symptom Detection**: Extract potential ADR symptoms
5. **Negation Handling**: Filter out negated symptoms
6. **ADR Validation**: Cross-reference with medical knowledge
7. **Analytics**: Generate statistics and insights

## 📈 **Analytics Dashboard**

The system provides comprehensive analytics including:

### Top Drugs by ADR Count
- Venlor: 1,093 ADRs (27.2%)
- Dulojoy: 735 ADRs (18.3%)
- Lexa: 525 ADRs (13.0%)

### Top Symptoms
- Nausea: 978 occurrences
- Anxiety: 694 occurrences
- Headache: 691 occurrences

### Sentiment Breakdown
- All detected ADRs are negative sentiment (as expected for adverse reactions)
- Sentiment tracked separately for all drugs regardless of ADR detection

## 🛠 **Technical Details**

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
```

### Data Sources
- **WebMD Reviews**: 43,762 patient reviews across 143 drugs
- **Indian Medicine Database**: 248,218 medicine entries with substitutes
- **Medical Reference**: Comprehensive drug information and side effects

### Algorithm Details
- **Sentiment**: Keyword-based classification with positive/negative word lists
- **Symptoms**: String matching against predefined symptom vocabulary
- **Negation**: Rule-based negation detection ("not experiencing", "no side effects")
- **Validation**: Cross-referencing detected symptoms with known drug side effects

## 🎯 **Use Cases**

- **Pharmacovigilance**: Monitor drug safety signals from patient reports
- **Medical Research**: Analyze ADR patterns across different medications
- **Healthcare Dashboard**: Real-time monitoring of adverse reactions
- **Drug Safety Analysis**: Identify high-risk medications and symptoms

## 📝 **Academic Context**

This project was developed as a final-year academic submission demonstrating:
- Modular software architecture
- Rule-based NLP processing
- Data analysis and visualization
- Healthcare informatics applications
- Python best practices and documentation

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is for academic and research purposes. Please cite appropriately if used in publications.

---

**Note**: Dataset files are excluded from this repository due to size constraints. Contact the maintainers for access to the original datasets.

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
