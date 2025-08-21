# Credit Card Default Detection - Main Pipeline

This document describes the comprehensive main pipeline (`main.py`) that orchestrates the entire credit card default detection process from data collection to inference.

## 🚀 Overview

The `main.py` file provides a unified interface to run the complete machine learning pipeline, integrating all the existing modules:

- **Data Collection** (`data_pipeline.py`)
- **Data Preprocessing** (`utils.py`)
- **Model Training** (`train.py`)
- **Model Inference** (`inference.py`)
- **Configuration Management** (`config.py`)

## 📁 File Structure

```
credit_card_default/
├── main.py                 # Main pipeline orchestrator
├── run_example.py          # Example usage scripts
├── train.py               # Model training module
├── inference.py           # Model inference module
├── data_pipeline.py       # Data collection and download
├── utils.py               # Utility functions
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🎯 Pipeline Phases

The main pipeline consists of 5 main phases:

### 1. Data Collection Phase 📥
- Downloads credit card default data from specified URLs
- Validates data integrity and structure
- Handles data storage and organization

### 2. Data Preprocessing Phase 🔧
- Loads and validates data
- Performs data exploration and analysis
- Identifies feature types (categorical vs continuous)
- Handles missing values and outliers

### 3. Model Training Phase 🤖
- Trains baseline models (Random Forest, KNN, Logistic Regression)
- Performs hyperparameter tuning with grid search
- Trains ensemble models
- Evaluates model performance
- Saves trained models and results

### 4. Inference Phase 🔮
- Loads trained models
- Makes predictions on new data
- Supports both file-based and DataFrame-based inference
- Generates prediction results

### 5. Report Generation Phase 📊
- Creates comprehensive pipeline execution reports
- Saves results in JSON format
- Tracks execution time and performance metrics

## 🚀 Usage

### Command Line Interface

The main pipeline can be run from the command line with various options:

```bash
# Run the complete pipeline
python main.py

# Run specific phases
python main.py --mode data_collection
python main.py --mode preprocessing
python main.py --mode training
python main.py --mode inference

# Force re-download of data
python main.py --force-download

# Skip grid search for faster execution
python main.py --no-grid-search

# Use custom input data for inference
python main.py --input-data path/to/your/data.csv

# Override configuration
python main.py --config-overrides '{"custom_setting": "value"}'
```

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--mode` | Pipeline mode to run | `full` | `full`, `data_collection`, `preprocessing`, `training`, `inference` |
| `--force-download` | Force re-download of data | `False` | Flag |
| `--no-grid-search` | Skip grid search hyperparameter tuning | `False` | Flag |
| `--input-data` | Path to input data for inference | `None` | File path |
| `--config-overrides` | JSON string with configuration overrides | `None` | JSON string |

### Programmatic Usage

You can also use the pipeline programmatically:

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

## 📊 Example Scripts

The `run_example.py` file provides several examples:

1. **Full Pipeline Example**: Runs the complete pipeline end-to-end
2. **Specific Phases Example**: Demonstrates running individual phases
3. **Custom Configuration Example**: Shows how to use custom configurations

Run examples:
```bash
python run_example.py
```

## ⚙️ Configuration

The pipeline uses the `config.py` file for all configurable parameters. You can override these settings programmatically:

```python
config_overrides = {
    'TRAIN_DATA_PATH': 'custom_train.csv',
    'TEST_DATA_PATH': 'custom_test.csv',
    'RANDOM_STATE': 123,
    'TEST_SIZE': 0.3
}

pipeline = CreditCardDefaultPipeline(config_overrides)
```

## 📈 Output and Results

The pipeline generates several outputs:

### Directories Created
- `data/`: Downloaded and processed data files
- `models/`: Trained model files (`.joblib` format)
- `plots/`: Visualization plots and charts
- `results/`: Pipeline execution reports and results

### Key Output Files
- `credit_card_pipeline.log`: Pipeline execution log
- `pipeline_report.json`: Comprehensive execution report
- Trained models: `random_forest.joblib`, `knn.joblib`, etc.
- Performance plots: confusion matrices, ROC curves, etc.

## 🔍 Monitoring and Logging

The pipeline provides comprehensive logging:

- **Console Output**: Real-time progress updates with emojis
- **Log File**: Detailed execution log saved to `credit_card_pipeline.log`
- **Progress Tracking**: Phase-by-phase execution status
- **Error Handling**: Graceful error handling with detailed error messages

## 🚨 Error Handling

The pipeline includes robust error handling:

- **Graceful Failures**: Individual phase failures don't crash the entire pipeline
- **Detailed Error Messages**: Clear error descriptions for debugging
- **Recovery Options**: Ability to restart from specific phases
- **Validation Checks**: Data and model validation at each step

## 🎯 Use Cases

### 1. End-to-End ML Pipeline
```bash
python main.py
```
Complete pipeline execution for production deployment.

### 2. Development and Testing
```bash
python main.py --mode training --no-grid-search
```
Quick model training without hyperparameter tuning.

### 3. Data Refresh
```bash
python main.py --mode data_collection --force-download
```
Force re-download of latest data.

### 4. Custom Inference
```bash
python main.py --mode inference --input-data new_data.csv
```
Run inference on custom data files.

### 5. Phase-Specific Execution
```bash
python main.py --mode preprocessing
```
Run only data preprocessing for analysis.

## 🔧 Customization

### Adding New Models
Extend the pipeline by adding new models in the training phase:

```python
# In train.py, add new model to the models dictionary
self.models['new_model'] = NewModelClass()
```

### Custom Data Sources
Modify `data_pipeline.py` to support additional data sources:

```python
# Add new data URLs
self.data_urls['custom'] = "https://example.com/custom_data.csv"
```

### Custom Evaluation Metrics
Extend the evaluation phase with new metrics:

```python
# Add custom metrics to the evaluation process
custom_metric = calculate_custom_metric(predictions, actual)
```

## 📚 Dependencies

Ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `matplotlib`, `seaborn`: Visualization
- `requests`: Data download
- `joblib`: Model persistence

## 🚀 Performance Tips

1. **Skip Grid Search**: Use `--no-grid-search` for faster execution during development
2. **Use Sample Data**: The pipeline automatically falls back to sample data if actual data is unavailable
3. **Parallel Processing**: Grid search automatically uses parallel processing where available
4. **Memory Management**: Large datasets are processed in chunks to manage memory usage

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Data Download Failures**: Check internet connection and URL accessibility
3. **Memory Issues**: Reduce dataset size or use sample data for testing
4. **Model Loading Errors**: Ensure models are trained before running inference

### Debug Mode

Enable detailed logging by modifying the log level in `config.py`:

```python
LOG_LEVEL = "DEBUG"
```

## 📞 Support

For issues and questions:
1. Check the log files for detailed error messages
2. Review the configuration parameters in `config.py`
3. Ensure all dependencies are properly installed
4. Verify data file paths and permissions

## 🔄 Version History

- **v1.0.0**: Initial release with complete pipeline integration
- Comprehensive error handling and logging
- Support for both command-line and programmatic usage
- Modular design for easy extension and customization

---

**Happy Machine Learning! 🎉**
