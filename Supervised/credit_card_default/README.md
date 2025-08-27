# Credit Card Default Detection

A comprehensive machine learning project for detecting credit card defaults using various classification algorithms. This project demonstrates best practices in machine learning pipeline development, from data exploration to model deployment.

## Project Overview

This project implements a complete machine learning pipeline for credit card default detection, including:

- **Data Analysis & Visualization**: Comprehensive exploratory data analysis with automated visualizations
- **Multiple ML Models**: Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, and Ensemble methods
- **Hyperparameter Tuning**: Grid search optimization for model performance
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Production Ready**: Well-structured, documented, and maintainable code

## Project Structure

```
Supervised/
├── credit_card_default/
│   ├── main.py                         # Main pipeline orchestrator
│   ├── run_example.py                  # Example usage scripts
│   ├── train.py                        # Model training module
│   ├── inference.py                    # Model inference module
│   ├── data_pipeline.py                # Data collection and download
│   ├── utils.py                        # Utility functions
│   ├── config.py                       # Configuration parameters
│   ├── requirements.txt                # Python dependencies
│   ├── README.md                       # This file
│   ├── MAIN_PIPELINE_README.md         # Main pipeline documentation
│   ├── plots/                          # Generated visualizations
│   ├── models/                         # Saved trained models
│   ├── results/                        # Analysis results and reports
│   └── data/                           # Downloaded data files
├── credit_card_default_detection.py    # Basic implementation
├── credit_card_default_enhanced.py     # Enhanced version with advanced features
└── DATA_PIPELINE_README.md             # Data pipeline documentation
```

## Quick Start

### 1. Install Dependencies

```bash
cd Supervised/credit_card_default
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline (Recommended)

The easiest way to get started is using the new unified main pipeline:

```bash
# Run the complete pipeline from data collection to inference
python main.py

# For faster execution during development (skip hyperparameter tuning)
python main.py --no-grid-search

# Force re-download of data
python main.py --force-download
```

### 3. Alternative: Run Individual Components

If you prefer to run components separately:

```bash
# Download data only
python main.py --mode data_collection

# Run specific phases
python main.py --mode preprocessing
python main.py --mode training
python main.py --mode inference

# Or use the original scripts
python train.py
python inference.py
```

## Data Description

The dataset contains credit card default information with the following features:

- **Customer Information**: ID, Gender, Education, Marital Status, Age
- **Credit Information**: Credit Line
- **Payment History**: 6 months of payment behavior (X6-X11)
- **Bill Amounts**: 6 months of bill statements (X12-X17)
- **Payment Amounts**: 6 months of actual payments (X18-X23)
- **Target Variable**: Default status (0=No Default, 1=Default)

### Feature Mapping

| Feature | Description |
|---------|-------------|
| X1      | Credit Line |
| X2      | Gender (1=male, 2=female) |
| X3      | Education (1=grad school, 2=university, 3=high school, 4=others) |
| X4      | Marital Status (1=married, 2=single, 3=others) |
| X5      | Age (years) |
| X6-X11  | Payment History (months) |
| X12-X17 | Bill Amounts (months) |
| X18-X23 | Payment Amounts (months) |

### Payment History Codes

- **-2**: Pay 2 months ahead
- **-1**: Pay 1 month ahead
- **0**: Pay on time
- **1-9**: Delay 1-9 months


### New: Unified Main Pipeline

- **End-to-End Automation**: Complete pipeline from data collection to inference
- **Flexible Execution**: Run full pipeline or specific phases independently
- **Command Line Interface**: Rich CLI with various options and flags
- **Programmatic API**: Can be imported and used in other Python scripts
- **Comprehensive Logging**: Real-time progress tracking with detailed logs
- **Error Recovery**: Graceful handling of failures with restart capabilities

### Core Functionality

- **Automated Data Loading**: Supports both CSV files and sample data generation
- **Data Quality Checks**: Missing values, data types, and distribution analysis
- **Feature Engineering**: Automatic identification of categorical vs. continuous variables
- **Data Preprocessing**: Handling missing values, outliers, and feature scaling
- **Multiple ML Models**: Comprehensive model comparison
- **Hyperparameter Optimization**: Grid search for best parameters
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Multiple metrics and visualizations

### Advanced Features

- **Ensemble Methods**: Voting classifiers combining multiple models
- **Advanced Visualizations**: Correlation matrices, feature relationships, payment history analysis
- **Model Cards**: Detailed documentation for each trained model
- **Results Export**: JSON format for further analysis
- **Logging**: Comprehensive logging throughout the pipeline
- **Error Handling**: Robust error handling and validation

## Model Performance

The project evaluates models using multiple metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Visualizations

Automatically generated visualizations include:

- **Distribution Plots**: Categorical and continuous variable distributions
- **Correlation Matrix**: Feature correlation heatmap
- **Feature Relationships**: Box plots showing feature-target relationships
- **Payment History Analysis**: Pie charts for payment behavior patterns
- **Confusion Matrices**: Model prediction accuracy visualization
- **ROC Curves**: Model performance curves
- **Feature Importance**: Top features for tree-based models

## Configuration

All project parameters are centralized in `config.py`:

- Data paths and directories
- Model hyperparameters
- Visualization settings
- Cross-validation parameters
- Feature descriptions and mappings

## Usage Examples

### New: Main Pipeline Usage (Recommended)

```python
from main import CreditCardDefaultPipeline

# Initialize pipeline
pipeline = CreditCardDefaultPipeline()

# Run complete pipeline
success = pipeline.run_full_pipeline(
    force_download=False,
    use_grid_search=True,
    input_data_path=None
)

# Run specific phases
pipeline.run_data_collection()
pipeline.run_data_preprocessing()
pipeline.run_model_training()
pipeline.run_inference()

# Run with custom configuration
config_overrides = {'custom_setting': 'value'}
pipeline = CreditCardDefaultPipeline(config_overrides)
```

### Command Line Usage

```bash
# Run complete pipeline
python main.py

# Run specific phases
python main.py --mode training --no-grid-search

# Force re-download data
python main.py --force-download

# Custom inference data
python main.py --mode inference --input-data new_data.csv
```


## Testing

The project includes sample data generation for testing:

```python
from utils import load_sample_data

# Generate sample data for testing
train_data, test_data = load_sample_data()
print(f"Sample training data shape: {train_data.shape}")
print(f"Sample test data shape: {test_data.shape}")
```

### New: Pipeline Testing

Test the main pipeline with examples:

```bash
# Run example scripts
python run_example.py

# Test specific pipeline phases
python main.py --mode preprocessing --no-grid-search
```

The pipeline automatically falls back to sample data if actual data is unavailable, making it perfect for testing and development.

## Output Files

### Generated Directories

- **`plots/`**: All visualization plots (PNG format)
- **`models/`**: Trained models (joblib format)
- **`results/`**: Analysis results and model cards (JSON/MD format)

### Key Output Files

- **Pipeline Reports**:
  - `credit_card_pipeline.log`: Comprehensive execution log
  - `pipeline_report.json`: Complete pipeline execution report
- **Model Results**:
  - `data_analysis_results.json`: Comprehensive data analysis
  - `pipeline_results.json`: Complete pipeline results
  - `{model_name}_classification_report.json`: Individual model reports
  - `{model_name}_model_card.md`: Model documentation
- **Visualizations**: Various plots including confusion matrices, ROC curves, etc.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **Data Files Not Found**: Set `use_sample_data=True` to generate sample data
3. **Memory Issues**: Reduce dataset size or use smaller models
4. **Plot Display Issues**: Ensure matplotlib backend is properly configured

### Error Messages

- **"Data files not found"**: Check file paths or enable sample data generation
- **"Model saving failed"**: Ensure write permissions for output directories
- **"Import errors"**: Verify all dependencies are installed
