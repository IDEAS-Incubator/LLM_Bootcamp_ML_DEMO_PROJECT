"""
Configuration file for Credit Card Default Detection project.

This file contains all the configurable parameters for the project.
"""

# Data paths
TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "test.csv"

# Output directories
PLOTS_DIR = "plots"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Data processing parameters
CATEGORICAL_THRESHOLD = 15
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
BASELINE_RF_PARAMS = {
    "n_estimators": 100,
    "class_weight": {0: 1, 1: 3},
    "random_state": RANDOM_STATE,
}

GRID_SEARCH_RF_PARAMS = {
    "n_estimators": [100, 300, 500],
    "criterion": ["gini", "entropy"],
    "class_weight": [{0: 1, 1: 3}],
}

TUNED_RF_PARAMS = {
    "n_estimators": 300,
    "criterion": "entropy",
    "class_weight": {0: 1, 1: 3},
    "random_state": RANDOM_STATE,
}

# Enhanced script parameters
RF_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

KNN_PARAMS = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

LR_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
}

GB_PARAMS = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
}

SVM_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto", 0.001, 0.01],
}

LOGISTIC_PARAMS = {
    "class_weight": {0: 1, 1: 3},
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
}

# Cross-validation parameters
CV_FOLDS = 5
SCORING_METRIC = "f1_weighted"

# Visualization parameters
FIGURE_SIZE_LARGE = (20, 16)
FIGURE_SIZE_MEDIUM = (12, 5)
FIGURE_SIZE_SMALL = (8, 6)
PLOT_DPI = 300

# Logging parameters
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Feature names mapping for better interpretability
FEATURE_DESCRIPTIONS = {
    "id": "Customer ID",
    "X1": "Credit Line",
    "X2": "Gender (1=male, 2=female)",
    "X3": "Education (1=grad school, 2=university, 3=high school, 4=others)",
    "X4": "Marital Status (1=married, 2=single, 3=others)",
    "X5": "Age (years)",
    "X6": "Payment History - September 2015",
    "X7": "Payment History - August 2015",
    "X8": "Payment History - July 2015",
    "X9": "Payment History - June 2015",
    "X10": "Payment History - May 2015",
    "X11": "Payment History - April 2015",
    "X12": "Bill Amount - September 2015",
    "X13": "Bill Amount - August 2015",
    "X14": "Bill Amount - July 2015",
    "X15": "Bill Amount - June 2015",
    "X16": "Bill Amount - May 2015",
    "X17": "Bill Amount - April 2015",
    "X18": "Payment Amount - September 2015",
    "X19": "Payment Amount - August 2015",
    "X20": "Payment Amount - July 2015",
    "X21": "Payment Amount - June 2015",
    "X22": "Payment Amount - May 2015",
    "X23": "Payment Amount - April 2015",
    "Y": "Default Status (0=no default, 1=default)",
}

# Payment history mapping
PAYMENT_HISTORY_MAPPING = {
    -2: "Pay 2 months ahead",
    -1: "Pay 1 month ahead",
    0: "Pay on time",
    1: "Delay 1 month",
    2: "Delay 2 months",
    3: "Delay 3 months",
    4: "Delay 4 months",
    5: "Delay 5 months",
    6: "Delay 6 months",
    7: "Delay 7 months",
    8: "Delay 8 months",
    9: "Delay 9 months",
}
