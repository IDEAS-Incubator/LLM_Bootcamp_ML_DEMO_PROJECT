#!/usr/bin/env python3
"""
Main Pipeline for Credit Card Default Detection

This script orchestrates the complete machine learning pipeline from data collection
to inference, including:
1. Data collection and download
2. Data preprocessing and exploration
3. Model training and evaluation
4. Model inference and prediction
5. Results visualization and analysis

Author: ML Bootcamp
Date: 2024
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import pandas as pd

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
import config
import data_pipeline
import utils
from train import EnhancedCreditCardDefaultDetector
from inference import CreditCardDefaultInference

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[logging.FileHandler("credit_card_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class CreditCardDefaultPipeline:
    """
    Main pipeline class that orchestrates the entire credit card default detection process.
    """

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the main pipeline.

        Args:
            config_overrides: Optional configuration overrides
        """
        self.config_overrides = config_overrides or {}
        self.data_pipeline = None
        self.trainer = None
        self.inference_engine = None
        self.pipeline_results = {}

        # Create necessary directories
        utils.create_directories()

        logger.info("Credit Card Default Detection Pipeline Initialized")

    def run_data_collection(self, force_download: bool = False) -> bool:
        """
        Run the data collection and download phase.

        Args:
            force_download: Whether to force re-download of data

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Data Collection Phase...")
        logger.debug(f"=" * 100)

        try:
            self.data_pipeline = data_pipeline.CreditCardDataPipeline()

            # Check if data already exists
            if not force_download and self.data_pipeline.check_data_exists():
                logger.info("Data already exists, skipping download")
                return True

            # Download data
            success = self.data_pipeline.download_all_data()

            if success:
                logger.info("Data collection completed successfully")
                return True
            else:
                logger.error("Data collection failed")
                return False

        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return False

    def run_data_preprocessing(self) -> bool:
        """
        Run the data preprocessing and exploration phase.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Data Preprocessing Phase...")
        logger.debug(f"=" * 100)
        try:
            # Initialize trainer to handle data preprocessing
            self.trainer = EnhancedCreditCardDefaultDetector()

            # Load and preprocess data
            if not self.trainer.load_data(use_sample_data=True):
                logger.error("Failed to load data")
                return False

            # Perform data quality checks
            self.trainer.perform_data_quality_check()

            # Identify variable types
            self.trainer.identify_variable_types()

            # Run data exploration
            self._run_data_exploration()

            logger.info("Data preprocessing completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return False

    def _run_data_exploration(self):
        """Run data exploration and generate initial insights."""
        logger.info("Running data exploration...")

        try:
            # Basic data info
            logger.info(f"Training data shape: {self.trainer.train_data.shape}")
            logger.info(f"Test data shape: {self.trainer.test_data.shape}")

            # Target distribution
            if "Y" in self.trainer.train_data.columns:
                target_dist = self.trainer.train_data["Y"].value_counts()
                logger.info(f"Target distribution:\n{target_dist}")

                # Calculate class imbalance
                imbalance_ratio = (
                    target_dist[1] / target_dist[0]
                    if 0 in target_dist and 1 in target_dist
                    else 0
                )
                logger.info(
                    f"Class imbalance ratio (default/non-default): {imbalance_ratio:.3f}"
                )

            # Feature information
            logger.info(
                f"Number of features: {len(self.trainer.feature_vars) if self.trainer.feature_vars else 'Not set'}"
            )

        except Exception as e:
            logger.warning(f"Data exploration failed: {e}")

    def run_model_training(self, use_grid_search: bool = True) -> bool:
        """
        Run the model training and evaluation phase.

        Args:
            use_grid_search: Whether to use grid search for hyperparameter tuning

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Model Training Phase...")
        logger.debug(f"=" * 100)
        try:
            # Initialize trainer if not already done
            if not self.trainer:
                logger.info("Initializing trainer for training phase...")
                self.trainer = EnhancedCreditCardDefaultDetector()

                # Load and preprocess data
                if not self.trainer.load_data(use_sample_data=True):
                    logger.error("Failed to load data")
                    return False

                # Perform data quality checks
                self.trainer.perform_data_quality_check()

                # Identify variable types
                self.trainer.identify_variable_types()

            # Prepare data for training
            logger.info("Preparing data for training...")
            X_train, y_train, X_val, y_val = self.trainer.prepare_data()

            # Train models
            logger.info("Training multiple models...")
            self.trainer.build_multiple_models(
                X_train, y_train, X_val, y_val, use_grid_search
            )

            # Generate reports
            logger.info("Generating detailed reports...")
            self.trainer.generate_detailed_reports(X_val, y_val)

            # Save models
            logger.info("Saving trained models...")
            self.trainer.save_all_models()

            # Create model cards
            logger.info("Creating model documentation...")
            self.trainer.create_model_cards()

            logger.info("Model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False

    def run_inference(self, input_data_path: Optional[str] = None) -> bool:
        """
        Run the inference and prediction phase.

        Args:
            input_data_path: Path to input data for inference

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Inference Phase...")
        logger.debug(f"=" * 100)
        try:
            # Initialize inference engine
            self.inference_engine = CreditCardDefaultInference()

            # Set feature variables
            if self.trainer and self.trainer.feature_vars:
                self.inference_engine.set_feature_variables(
                    categorical_vars=self.trainer.categorical_vars,
                    continuous_vars=self.trainer.continuous_vars,
                )

            # Load models
            if not self.inference_engine.models:
                logger.error("No models loaded for inference")
                return False

            # Run inference on test data or provided input
            if input_data_path and os.path.exists(input_data_path):
                logger.info(f"Running inference on: {input_data_path}")
                # Load data from file
                input_data = pd.read_csv(input_data_path)
                predictions = self.inference_engine.predict_batch(input_data)
            else:
                logger.info("Running inference on test data...")
                predictions = self.inference_engine.predict_batch(
                    self.trainer.test_data
                )

            if predictions is not None:
                logger.info(
                    f"Inference completed. Generated {len(predictions)} predictions"
                )
                self.pipeline_results["inference"] = {
                    "predictions_count": len(predictions),
                    "models_used": list(self.inference_engine.models.keys()),
                }
                return True
            else:
                logger.error("Inference failed")
                return False

        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return False

    def generate_report(self) -> bool:
        """
        Generate a comprehensive pipeline report.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Generating Pipeline Report...")
        logger.debug(f"=" * 100)
        try:
            report = {
                "pipeline_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": "1.0.0",
                "config": {
                    "data_paths": {
                        "train": config.TRAIN_DATA_PATH,
                        "test": config.TEST_DATA_PATH,
                    },
                    "output_dirs": {
                        "plots": config.PLOTS_DIR,
                        "models": config.MODELS_DIR,
                        "results": config.RESULTS_DIR,
                    },
                },
                "results": self.pipeline_results,
            }

            # Save report
            report_path = Path(config.RESULTS_DIR) / "pipeline_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Pipeline report saved to: {report_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False

    def run_full_pipeline(
        self,
        force_download: bool = False,
        use_grid_search: bool = True,
        input_data_path: Optional[str] = None,
    ) -> bool:
        """
        Run the complete pipeline from start to finish.

        Args:
            force_download: Whether to force re-download of data
            use_grid_search: Whether to use grid search for hyperparameter tuning
            input_data_path: Path to input data for inference

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Full Pipeline Execution...")
        start_time = time.time()

        try:
            # Phase 1: Data Collection
            logger.info("Running Data Collection Phase...")
            logger.debug(f"=" * 100)
            if not self.run_data_collection(force_download):
                return False

            # Phase 2: Data Preprocessing
            logger.info("Running Data Preprocessing Phase...")
            logger.debug(f"=" * 100)
            if not self.run_data_preprocessing():
                return False

            # Phase 3: Model Training
            logger.info("Running Model Training Phase...")
            logger.debug(f"=" * 100)
            if not self.run_model_training(use_grid_search):
                return False

            # Phase 4: Inference
            logger.info("Running Inference Phase...")
            logger.debug(f"=" * 100)
            if not self.run_inference(input_data_path):
                return False

            # Phase 5: Generate Report
            logger.info("Running Generate Report Phase...")
            logger.debug(f"=" * 100)
            if not self.generate_report():
                return False

            # Calculate execution time
            execution_time = time.time() - start_time
            logger.info(
                f"Full pipeline completed successfully in {execution_time:.2f} seconds!"
            )

            return True

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False

    def run_specific_phase(self, phase: str, **kwargs) -> bool:
        """
        Run a specific phase of the pipeline.

        Args:
            phase: Phase to run ('data_collection', 'preprocessing', 'training', 'inference')
            **kwargs: Additional arguments for the specific phase

        Returns:
            bool: True if successful, False otherwise
        """
        phase_mapping = {
            "data_collection": self.run_data_collection,
            "preprocessing": self.run_data_preprocessing,
            "training": self.run_model_training,
            "inference": self.run_inference,
        }

        if phase not in phase_mapping:
            logger.error(f"Unknown phase: {phase}")
            return False

        logger.info(f"Running specific phase: {phase}")

        # Filter kwargs based on the specific phase
        if phase == "data_collection":
            # Only pass force_download to data_collection
            filtered_kwargs = {k: v for k, v in kwargs.items() if k == "force_download"}
        elif phase == "training":
            # Only pass use_grid_search to training
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k == "use_grid_search"
            }
        elif phase == "inference":
            # Only pass input_data_path to inference
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k == "input_data_path"
            }
        else:
            # preprocessing doesn't take any additional parameters
            filtered_kwargs = {}

        return phase_mapping[phase](**filtered_kwargs)


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Credit Card Default Detection Pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["full", "data_collection", "preprocessing", "training", "inference"],
        default="full",
        help="Pipeline mode to run",
    )

    parser.add_argument(
        "--force-download", action="store_true", help="Force re-download of data"
    )

    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip grid search hyperparameter tuning",
    )

    parser.add_argument(
        "--input-data", type=str, help="Path to input data for inference"
    )

    parser.add_argument(
        "--config-overrides", type=str, help="JSON string with configuration overrides"
    )

    args = parser.parse_args()

    try:
        # Parse config overrides if provided
        config_overrides = {}
        if args.config_overrides:
            config_overrides = json.loads(args.config_overrides)

        # Initialize pipeline
        pipeline = CreditCardDefaultPipeline(config_overrides)

        # Run pipeline based on mode
        if args.mode == "full":
            success = pipeline.run_full_pipeline(
                force_download=args.force_download,
                use_grid_search=not args.no_grid_search,
                input_data_path=args.input_data,
            )
        else:
            success = pipeline.run_specific_phase(
                args.mode,
                force_download=args.force_download,
                use_grid_search=not args.no_grid_search,
                input_data_path=args.input_data,
            )

        if success:
            logger.info("Pipeline execution completed successfully!")
            return 0
        else:
            logger.error("Pipeline execution failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
