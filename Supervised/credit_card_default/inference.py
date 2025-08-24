#!/usr/bin/env python3
"""
Inference Script for Credit Card Default Detection

This script loads trained models and makes predictions on new data.
It can be used for production inference or batch prediction.

Author: ML Bootcamp
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import warnings

# Import local modules
import config
import utils

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class CreditCardDefaultInference:
    """
    A class for making predictions using trained credit card default detection models.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the inference engine.

        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_vars = None
        self.categorical_vars = []
        self.continuous_vars = []

        # Load models and scaler
        self._load_models()

    def _load_models(self) -> None:
        """Load all trained models and the scaler."""
        logger.info("Loading trained models...")

        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return

        # Load scaler
        scaler_path = self.models_dir / "scaler.joblib"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ Scaler loaded successfully")
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")

        # Load all model files
        model_files = list(self.models_dir.glob("*.joblib"))
        model_files = [f for f in model_files if f.name != "scaler.joblib"]

        for model_file in model_files:
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.models[model_name] = model
                logger.info(f"✓ Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model {model_file.name}: {e}")

        if not self.models:
            logger.warning("No models loaded. Please ensure models are trained first.")
        else:
            logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def set_feature_variables(
        self, categorical_vars: List[str], continuous_vars: List[str]
    ) -> None:
        """
        Set the feature variables for preprocessing.

        Args:
            categorical_vars (List[str]): List of categorical variable names
            continuous_vars (List[str]): List of continuous variable names
        """
        self.categorical_vars = categorical_vars
        self.continuous_vars = continuous_vars
        self.feature_vars = categorical_vars + continuous_vars
        logger.info(
            f"Set {len(self.categorical_vars)} categorical and {len(self.continuous_vars)} continuous features"
        )

    def preprocess_data(
        self, data: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess the input data for inference.

        Args:
            data (pd.DataFrame): Input data
            is_training (bool): Whether this is training data (affects missing value handling)

        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Preprocessing data for inference...")

        # Make a copy to avoid modifying original data
        processed_data = data.copy()

        # Handle missing values
        if is_training:
            # For training data, use median for continuous variables
            for col in self.continuous_vars:
                if col in processed_data.columns and processed_data[col].isnull().any():
                    median_val = processed_data[col].median()
                    processed_data[col].fillna(median_val, inplace=True)
                    logger.info(
                        f"Filled missing values in {col} with median: {median_val}"
                    )
        else:
            # For inference data, fill with 0 (conservative approach)
            for col in processed_data.columns:
                if processed_data[col].isnull().any():
                    processed_data[col].fillna(0, inplace=True)
                    logger.info(f"Filled missing values in {col} with 0")

        # Handle outliers for continuous variables (IQR method)
        for col in self.continuous_vars:
            if col in processed_data.columns:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                processed_data[col] = processed_data[col].clip(
                    lower=lower_bound, upper=upper_bound
                )

        # Scale continuous features if scaler is available
        if self.scaler and self.continuous_vars:
            continuous_cols = [
                col for col in self.continuous_vars if col in processed_data.columns
            ]
            if continuous_cols:
                try:
                    processed_data[continuous_cols] = self.scaler.transform(
                        processed_data[continuous_cols]
                    )
                    logger.info(f"Scaled continuous features: {continuous_cols}")
                except Exception as e:
                    logger.warning(f"Could not scale continuous features: {e}")

        logger.info("Data preprocessing completed")
        return processed_data

    def predict_single(
        self, features: Dict[str, Union[int, float]], model_name: str = None
    ) -> Dict[str, Union[int, float]]:
        """
        Make prediction for a single sample.

        Args:
            features (Dict[str, Union[int, float]]): Feature values for single sample
            model_name (str): Name of specific model to use. If None, uses ensemble if available

        Returns:
            Dict[str, Union[int, float]]: Prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Preprocess
        processed_df = self.preprocess_data(df, is_training=False)

        # Select features
        if self.feature_vars:
            processed_df = processed_df[self.feature_vars]

        # Make prediction
        prediction, probability = self._make_prediction(processed_df, model_name)

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "default_risk": "High" if prediction[0] == 1 else "Low",
            "model_used": model_name or "ensemble",
        }

    def predict_batch(self, data: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """
        Make predictions for a batch of samples.

        Args:
            data (pd.DataFrame): DataFrame containing features for multiple samples
            model_name (str): Name of specific model to use. If None, uses ensemble if available

        Returns:
            pd.DataFrame: Original data with prediction columns added
        """
        logger.info(f"Making batch predictions for {len(data)} samples...")

        # Preprocess data
        processed_data = self.preprocess_data(data, is_training=False)

        # Select features
        if self.feature_vars:
            processed_data = processed_data[self.feature_vars]

        # Make predictions
        predictions, probabilities = self._make_prediction(processed_data, model_name)

        # Add predictions to original data
        result_data = data.copy()
        result_data["prediction"] = predictions
        result_data["probability"] = probabilities
        result_data["default_risk"] = result_data["prediction"].map(
            {0: "Low", 1: "High"}
        )
        result_data["model_used"] = model_name or "ensemble"

        logger.info(
            f"Batch predictions completed. {sum(predictions)} samples predicted as default"
        )
        return result_data

    def _make_prediction(
        self, processed_data: pd.DataFrame, model_name: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the specified or best available model.

        Args:
            processed_data (pd.DataFrame): Preprocessed feature data
            model_name (str): Name of specific model to use

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and probabilities
        """
        # Select model
        if model_name and model_name in self.models:
            model = self.models[model_name]
            logger.info(f"Using specified model: {model_name}")
        elif "Ensemble" in self.models:
            model = self.models["Ensemble"]
            logger.info("Using ensemble model")
        elif "Random_Forest_Optimized" in self.models:
            model = self.models["Random_Forest_Optimized"]
            logger.info("Using optimized Random Forest model")
        elif "Random_Forest_Baseline" in self.models:
            model = self.models["Random_Forest_Baseline"]
            logger.info("Using baseline Random Forest model")
        elif self.models:
            # Use first available model
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            logger.info(f"Using available model: {model_name}")
        else:
            raise ValueError("No trained models available for prediction")

        # Make predictions
        try:
            predictions = model.predict(processed_data)

            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed_data)[
                    :, 1
                ]  # Probability of default
            else:
                # If no probabilities available, use predictions as probabilities
                probabilities = predictions.astype(float)
                logger.warning("Model does not support probability predictions")

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about available models.

        Returns:
            Dict[str, Dict]: Information about each model
        """
        model_info = {}

        for name, model in self.models.items():
            info = {
                "type": type(model).__name__,
                "has_probabilities": hasattr(model, "predict_proba"),
                "has_feature_importance": hasattr(model, "feature_importances_"),
            }

            # Add model-specific parameters if available
            if hasattr(model, "get_params"):
                try:
                    info["parameters"] = model.get_params()
                except:
                    info["parameters"] = {}

            model_info[name] = info

        return model_info

    def get_feature_importance(self, model_name: str = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance from a tree-based model.

        Args:
            model_name (str): Name of the model to get feature importance from

        Returns:
            Optional[pd.DataFrame]: Feature importance DataFrame if available
        """
        # Select model
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif "Random_Forest_Optimized" in self.models:
            model = self.models["Random_Forest_Optimized"]
        elif "Random_Forest_Baseline" in self.models:
            model = self.models["Random_Forest_Baseline"]
        else:
            logger.warning("No suitable model found for feature importance")
            return None

        # Check if model supports feature importance
        if not hasattr(model, "feature_importances_"):
            logger.warning(
                f"Model {type(model).__name__} does not support feature importance"
            )
            return None

        # Get feature importance
        if hasattr(model, "best_estimator_"):
            # GridSearchCV model
            feature_importance = model.best_estimator_.feature_importances_
        else:
            # Direct model
            feature_importance = model.feature_importances_

        # Create DataFrame
        if self.feature_vars:
            importance_df = pd.DataFrame(
                {"feature": self.feature_vars, "importance": feature_importance}
            ).sort_values("importance", ascending=False)

            return importance_df

        return None


def main():
    """Main function to demonstrate inference usage."""
    print("=" * 60)
    print("CREDIT CARD DEFAULT DETECTION - INFERENCE")
    print("=" * 60)

    # Initialize inference engine
    try:
        inference = CreditCardDefaultInference()

        if not inference.models:
            print("❌ No trained models found. Please train models first.")
            print("Run: python credit_card_default_enhanced.py")
            return 1

        # Set feature variables (these should match your training data)
        categorical_vars = ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"]
        continuous_vars = [
            "X1",
            "X5",
            "X12",
            "X13",
            "X14",
            "X15",
            "X16",
            "X17",
            "X18",
            "X19",
            "X20",
            "X21",
            "X22",
            "X23",
        ]

        inference.set_feature_variables(categorical_vars, continuous_vars)

        # Show available models
        print("\n📊 Available Models:")
        model_info = inference.get_model_info()
        for name, info in model_info.items():
            print(f"  {name}: {info['type']}")
            print(f"    Probabilities: {'✓' if info['has_probabilities'] else '✗'}")
            print(
                f"    Feature Importance: {'✓' if info['has_feature_importance'] else '✗'}"
            )

        # Example single prediction
        print("\n🔍 Example Single Prediction:")
        sample_features = {
            "X1": 200000,  # Credit line
            "X2": 1,  # Gender (1=male)
            "X3": 2,  # Education (2=university)
            "X4": 1,  # Marital status (1=married)
            "X5": 35,  # Age
            "X6": 0,  # Payment history
            "X7": 0,  # Payment history
            "X8": 0,  # Payment history
            "X9": 0,  # Payment history
            "X10": 0,  # Payment history
            "X11": 0,  # Payment history
            "X12": 200000,  # Bill amount
            "X13": 200000,  # Bill amount
            "X14": 200000,  # Bill amount
            "X15": 200000,  # Bill amount
            "X16": 200000,  # Bill amount
            "X17": 200000,  # Bill amount
            "X18": 200000,  # Payment amount
            "X19": 200000,  # Payment amount
            "X20": 200000,  # Payment amount
            "X21": 200000,  # Payment amount
            "X22": 200000,  # Payment amount
            "X23": 200000,  # Payment amount
        }

        try:
            result = inference.predict_single(sample_features)
            print(f"  Prediction: {result['prediction']} ({result['default_risk']})")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Model Used: {result['model_used']}")
        except Exception as e:
            print(f"  Error making prediction: {e}")

        # Show feature importance if available
        print("\n📈 Feature Importance (Top 10):")
        importance_df = inference.get_feature_importance()
        if importance_df is not None:
            top_features = importance_df.head(10)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        else:
            print("  Feature importance not available")

        print("\n🎉 Inference engine ready for use!")
        print("\n💡 Usage examples:")
        print("  # Single prediction")
        print("  result = inference.predict_single(features_dict)")
        print("  # Batch prediction")
        print("  results = inference.predict_batch(dataframe)")

    except Exception as e:
        print(f"❌ Error initializing inference engine: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
