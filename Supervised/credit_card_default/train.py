#!/usr/bin/env python3
"""
Enhanced Credit Card Default Detection

This script implements a comprehensive machine learning pipeline for detecting
credit card defaults using various classification algorithms including Random Forest,
KNN, and Logistic Regression. It includes advanced preprocessing, evaluation,
and visualization capabilities.

Author: ML Bootcamp
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
import warnings
import os
import logging
from typing import List, Tuple, Dict, Any
import joblib
from datetime import datetime

# Import local modules
import config
import utils

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class EnhancedCreditCardDefaultDetector:
    """
    An enhanced class to handle credit card default detection using machine learning.
    """

    def __init__(self, train_path: str = None, test_path: str = None):
        """
        Initialize the EnhancedCreditCardDefaultDetector.

        Args:
            train_path (str): Path to training data CSV file
            test_path (str): Path to test data CSV file
        """
        self.train_path = train_path or config.TRAIN_DATA_PATH
        self.test_path = test_path or config.TEST_DATA_PATH
        self.train_data = None
        self.test_data = None
        self.categorical_vars = []
        self.continuous_vars = []
        self.target_var = "Y"
        self.feature_vars = None
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}

        # Create necessary directories
        utils.create_directories()

    def load_data(self, use_sample_data: bool = False) -> bool:
        """
        Load training and test data from CSV files or generate sample data.

        Args:
            use_sample_data (bool): Whether to generate sample data if files not found

        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            logger.info("Loading training and test data...")

            # Check if actual data files exist
            if os.path.exists(self.train_path) and os.path.exists(self.test_path):
                self.train_data = pd.read_csv(self.train_path)
                self.test_data = pd.read_csv(self.test_path)
                logger.info("Loaded actual data files")
            elif use_sample_data:
                logger.info("Generating sample data for demonstration...")
                self.train_data, self.test_data = utils.load_sample_data()
                logger.info("Generated sample data")
            else:
                logger.error("Data files not found and sample data generation disabled")
                return False

            logger.info(f"Training data shape: {self.train_data.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def perform_data_quality_check(self) -> Dict[str, Any]:
        """
        Perform basic data quality checks.

        Returns:
            Dict containing data quality information
        """
        logger.info("Performing data quality checks...")

        quality_info = {
            "train_shape": self.train_data.shape,
            "test_shape": self.test_data.shape,
            "train_columns": list(self.train_data.columns),
            "test_columns": list(self.test_data.columns),
            "train_dtypes": self.train_data.dtypes.to_dict(),
            "test_dtypes": self.test_data.dtypes.to_dict(),
            "train_missing": self.train_data.isnull().sum().to_dict(),
            "test_missing": self.test_data.isnull().sum().to_dict(),
        }

        # Log basic info
        logger.info(f"Training data: {self.train_data.shape}")
        logger.info(f"Test data: {self.test_data.shape}")
        logger.info(f"Training columns: {list(self.train_data.columns)}")
        logger.info(f"Test columns: {list(self.test_data.columns)}")

        # Check for missing values
        train_missing = self.train_data.isnull().sum().sum()
        test_missing = self.test_data.isnull().sum().sum()
        logger.info(f"Missing values in training data: {train_missing}")
        logger.info(f"Missing values in test data: {test_missing}")

        return quality_info

    def identify_variable_types(self) -> None:
        """Identify categorical and continuous variables."""
        logger.info("Identifying variable types...")

        # Exclude target variable
        feature_columns = [
            col for col in self.train_data.columns if col != self.target_var
        ]

        for col in feature_columns:
            unique_count = self.train_data[col].nunique()
            if unique_count <= config.CATEGORICAL_THRESHOLD:
                self.categorical_vars.append(col)
            else:
                self.continuous_vars.append(col)

        self.feature_vars = self.categorical_vars + self.continuous_vars

        logger.info(f"Categorical variables: {self.categorical_vars}")
        logger.info(f"Continuous variables: {self.continuous_vars}")
        logger.info(f"Total features: {len(self.feature_vars)}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training by handling missing values, outliers, and scaling.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        logger.info("Preparing data for training...")
        logger.debug(f"=" * 100)

        # Separate features and target
        X = self.train_data[self.feature_vars].copy()
        y = self.train_data[self.target_var]

        # Handle missing values
        X = utils.preprocess_data(X, is_training=True)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )

        # Scale continuous features
        if self.continuous_vars:
            X_train_cont = X_train[self.continuous_vars]
            X_val_cont = X_val[self.continuous_vars]

            # Fit scaler on training data
            X_train_cont_scaled = self.scaler.fit_transform(X_train_cont)
            X_val_cont_scaled = self.scaler.transform(X_val_cont)

            # Replace continuous features with scaled versions
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            X_train_scaled[self.continuous_vars] = X_train_cont_scaled
            X_val_scaled[self.continuous_vars] = X_val_cont_scaled

            X_train, X_val = X_train_scaled, X_val_scaled

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")

        return X_train, y_train, X_val, y_val

    def build_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build and evaluate multiple machine learning models.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            use_grid_search: Whether to use grid search for hyperparameter tuning

        Returns:
            Dict containing model metrics
        """
        logger.info("Building multiple machine learning models...")
        logger.debug(f"=" * 100)

        # Build baseline Random Forest model
        baseline_rf_metrics = self._build_baseline_model(X_train, y_train, X_val, y_val)

        # Build other models
        models_metrics = {}
        models_metrics["Random_Forest_Baseline"] = baseline_rf_metrics

        if use_grid_search:
            # Random Forest with Grid Search
            rf_metrics = self._build_random_forest(X_train, y_train, X_val, y_val)
            if rf_metrics:
                models_metrics["Random_Forest_Optimized"] = rf_metrics
                self.models["Random_Forest_Optimized"] = self.models.get(
                    "Random_Forest_Optimized"
                )

            # KNN
            knn_metrics = self._build_knn(X_train, y_train, X_val, y_val)
            if knn_metrics:
                models_metrics["KNN"] = knn_metrics
                self.models["KNN"] = self.models.get("KNN")

            # Logistic Regression
            lr_metrics = self._build_logistic_regression(X_train, y_train, X_val, y_val)
            if lr_metrics:
                models_metrics["Logistic_Regression"] = lr_metrics
                self.models["Logistic_Regression"] = self.models.get(
                    "Logistic_Regression"
                )

            # Gradient Boosting
            gb_metrics = self._build_gradient_boosting(X_train, y_train, X_val, y_val)
            if gb_metrics:
                models_metrics["Gradient_Boosting"] = gb_metrics
                self.models["Gradient_Boosting"] = self.models.get("Gradient_Boosting")

            # SVM
            svm_metrics = self._build_svm(X_train, y_train, X_val, y_val)
            if svm_metrics:
                models_metrics["SVM"] = svm_metrics
                self.models["SVM"] = self.models.get("SVM")

            # Build ensemble
            ensemble_metrics = self._build_ensemble(X_train, y_train, X_val, y_val)
            if ensemble_metrics:
                models_metrics["Ensemble"] = ensemble_metrics
                self.models["Ensemble"] = self.models.get("Ensemble")
        else:
            logger.info("Skipping grid search - using baseline models only")
            # Use baseline models for faster execution
            models_metrics["Random_Forest_Baseline"] = baseline_rf_metrics

        return models_metrics

    def _build_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate baseline Random Forest model."""
        logger.info("Building baseline Random Forest model...")
        logger.debug(f"=" * 100)
        baseline_rf = RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_STATE, class_weight="balanced"
        )

        baseline_rf.fit(X_train, y_train)
        y_pred = baseline_rf.predict(X_val)
        y_pred_proba = baseline_rf.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["Random_Forest_Baseline"] = baseline_rf

        logger.info(
            f"Baseline Random Forest - F1 Score: {metrics_dict['f1_score']:.4f}"
        )
        return metrics_dict

    def _build_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate optimized Random Forest model."""
        logger.info("Building optimized Random Forest model...")
        logger.debug(f"=" * 100)
        rf = RandomForestClassifier(
            random_state=config.RANDOM_STATE, class_weight="balanced"
        )
        grid_search = self._perform_grid_search(rf, config.RF_PARAMS, X_train, y_train)

        y_pred = grid_search.predict(X_val)
        y_pred_proba = grid_search.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["Random_Forest_Optimized"] = grid_search

        logger.info(
            f"Optimized Random Forest - F1 Score: {metrics_dict['f1_score']:.4f}"
        )
        return metrics_dict

    def _build_knn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate KNN model."""
        logger.info("Building KNN model...")
        logger.debug(f"=" * 100)
        knn = KNeighborsClassifier()
        grid_search = self._perform_grid_search(
            knn, config.KNN_PARAMS, X_train, y_train
        )

        y_pred = grid_search.predict(X_val)
        y_pred_proba = grid_search.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["KNN"] = grid_search

        logger.info(f"KNN - F1 Score: {metrics_dict['f1_score']:.4f}")
        return metrics_dict

    def _build_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate Logistic Regression model."""
        logger.info("Building Logistic Regression model...")
        logger.debug(f"=" * 100)
        lr = LogisticRegression(
            random_state=config.RANDOM_STATE, class_weight="balanced"
        )
        grid_search = self._perform_grid_search(lr, config.LR_PARAMS, X_train, y_train)

        y_pred = grid_search.predict(X_val)
        y_pred_proba = grid_search.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["Logistic_Regression"] = grid_search

        logger.info(f"Logistic Regression - F1 Score: {metrics_dict['f1_score']:.4f}")
        return metrics_dict

    def _build_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate Gradient Boosting model."""
        logger.info("Building Gradient Boosting model...")
        logger.debug(f"=" * 100)
        gb = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
        grid_search = self._perform_grid_search(gb, config.GB_PARAMS, X_train, y_train)

        y_pred = grid_search.predict(X_val)
        y_pred_proba = grid_search.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["Gradient_Boosting"] = grid_search

        logger.info(f"Gradient Boosting - F1 Score: {metrics_dict['f1_score']:.4f}")
        return metrics_dict

    def _build_svm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate SVM model."""
        logger.info("Building SVM model...")
        logger.debug(f"=" * 100)
        svm = SVC(
            random_state=config.RANDOM_STATE, class_weight="balanced", probability=True
        )
        grid_search = self._perform_grid_search(
            svm, config.SVM_PARAMS, X_train, y_train
        )

        y_pred = grid_search.predict(X_val)
        y_pred_proba = grid_search.predict_proba(X_val)[:, 1]

        metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        self.models["SVM"] = grid_search

        logger.info(f"SVM - F1 Score: {metrics_dict['f1_score']:.4f}")
        return metrics_dict

    def _build_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Build and evaluate ensemble model."""
        logger.info("Building ensemble model...")
        logger.debug(f"=" * 100)
        # Get the best models for ensemble
        best_models = []
        for name, model in self.models.items():
            if name != "Random_Forest_Baseline":  # Exclude baseline
                best_models.append((name, model))

        if len(best_models) >= 2:
            ensemble = VotingClassifier(estimators=best_models, voting="soft")

            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_val)
            y_pred_proba = ensemble.predict_proba(X_val)[:, 1]

            metrics_dict = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            self.models["Ensemble"] = ensemble

            logger.info(f"Ensemble - F1 Score: {metrics_dict['f1_score']:.4f}")
            return metrics_dict
        else:
            logger.warning("Not enough models for ensemble")
            return None

    def _perform_grid_search(
        self, estimator, param_grid: Dict, X_train: pd.DataFrame, y_train: pd.Series
    ) -> GridSearchCV:
        """Perform grid search for hyperparameter tuning."""
        logger.info(f"Performing grid search for {type(estimator).__name__}...")
        logger.debug(f"=" * 100)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=config.SCORING_METRIC,
            cv=config.CV_FOLDS,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

        return grid_search

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various performance metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

    def generate_detailed_reports(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Generate detailed evaluation reports for all models."""
        logger.info("Generating detailed evaluation reports...")
        logger.debug(f"=" * 100)
        for model_name, model in self.models.items():
            if hasattr(model, "predict"):
                logger.info(f"Generating report for {model_name}...")

                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = (
                    model.predict_proba(X_val)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Generate classification report
                report = classification_report(y_val, y_pred, output_dict=True)

                # Save report
                utils.save_results_to_file(
                    report, f"{model_name}_classification_report.json"
                )

                # Create confusion matrix plot
                utils.plot_confusion_matrix(y_val, y_pred, output_dir=config.PLOTS_DIR)

                # Create ROC curve if probabilities available
                if y_pred_proba is not None:
                    utils.plot_roc_curve(
                        y_val, y_pred_proba, output_dir=config.PLOTS_DIR
                    )

                # Create feature importance plot for tree-based models
                if hasattr(model, "feature_importances_"):
                    utils.plot_feature_importance(
                        model, self.feature_vars, output_dir=config.PLOTS_DIR
                    )

    def save_all_models(self) -> None:
        """Save all trained models to disk."""
        logger.info("Saving all trained models...")
        logger.debug(f"=" * 100)
        for model_name, model in self.models.items():
            filepath = f"{config.MODELS_DIR}/{model_name}.joblib"
            try:
                joblib.dump(model, filepath)
                logger.info(f"Model {model_name} saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving model {model_name}: {str(e)}")

        # Save scaler
        scaler_path = f"{config.MODELS_DIR}/scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    def create_model_cards(self) -> None:
        """Create model cards for all trained models."""
        logger.info("Creating model cards...")
        logger.debug(f"=" * 100)
        for model_name, model in self.models.items():
            if model_name in self.results.get("model_metrics", {}):
                metrics = self.results["model_metrics"][model_name]

                # Get model parameters
                if hasattr(model, "get_params"):
                    params = model.get_params()
                else:
                    params = {}

                utils.create_model_card(model_name, metrics, params, config.RESULTS_DIR)

    def run_complete_pipeline(self, use_sample_data: bool = False) -> Dict[str, Any]:
        """
        Run the complete machine learning pipeline.

        Args:
            use_sample_data (bool): Whether to use sample data if actual data not available

        Returns:
            Dict containing all pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING ENHANCED CREDIT CARD DEFAULT DETECTION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load data
        if not self.load_data(use_sample_data=use_sample_data):
            logger.error("Failed to load data. Pipeline cannot continue.")
            return {}

        # Step 2: Data quality check
        quality_info = self.perform_data_quality_check()

        # Step 3: Identify variable types
        self.identify_variable_types()

        # Step 4: Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data()

        # Step 5: Build and evaluate models
        model_metrics = self.build_multiple_models(X_train, y_train, X_val, y_val)

        # Step 6: Generate detailed reports
        self.generate_detailed_reports(X_val, y_val)

        # Step 7: Save models
        self.save_all_models()

        # Step 8: Create model cards
        self.create_model_cards()

        # Compile results
        self.results = {
            "data_quality": quality_info,
            "model_metrics": model_metrics,
            "pipeline_timestamp": datetime.now().isoformat(),
        }

        # Save results
        utils.save_results_to_file(self.results, "pipeline_results.json")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Results summary:")
        for model_name, metrics in model_metrics.items():
            if metrics:
                logger.info(
                    f"  {model_name}: F1={metrics['f1_score']:.4f}, "
                    f"Accuracy={metrics['accuracy']:.4f}"
                )

        return self.results


def main():
    """Main function to run the enhanced credit card default detection pipeline."""
    # Initialize detector
    detector = EnhancedCreditCardDefaultDetector()

    # Run complete pipeline
    results = detector.run_complete_pipeline(use_sample_data=True)

    if results:
        print("\n🎉 Enhanced pipeline completed successfully!")
        print("\n📊 Model Performance Summary:")
        for model_name, metrics in results.get("model_metrics", {}).items():
            if metrics:
                print(f"  {model_name}:")
                print(f"    F1-Score: {metrics['f1_score']:.4f}")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
                print()
    else:
        print("\n❌ Pipeline failed. Check the logs above for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
