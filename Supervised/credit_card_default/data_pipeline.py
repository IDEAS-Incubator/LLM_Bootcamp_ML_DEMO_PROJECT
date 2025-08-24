#!/usr/bin/env python3
"""
Data Pipeline for Credit Card Default Detection

This script downloads credit card default data from the provided URLs,
validates the data, and prepares it for the machine learning pipeline.

Data Sources:
- Training data: https://public.3.basecamp.com/p/fe3ojngZ5dZsvajj1JFRNzmpJwE
- Test data: https://public.3.basecamp.com/p/6Py3Z5dZsvajj1JFRNzmpJwE

Author: ML Bootcamp
Date: 2024
"""

import requests
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import hashlib
import time
from urllib.parse import urlparse
import zipfile
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CreditCardDataPipeline:
    """
    A comprehensive data pipeline for downloading and preparing credit card default data.
    """

    def __init__(self, data_dir: str = "data", download_timeout: int = 300):
        """
        Initialize the data pipeline.

        Args:
            data_dir (str): Directory to store downloaded data
            download_timeout (int): Timeout for downloads in seconds
        """
        self.data_dir = Path(data_dir)
        self.download_timeout = download_timeout

        # Data URLs
        self.data_urls = {
            "train": "https://public.3.basecamp.com/p/fe3ojngYGH6nHwNPBJviuFMz",
            "test": "https://public.3.basecamp.com/p/6Py3Z5dZsvajj1JFRNzmpJwE",
        }

        # Expected file names
        self.expected_files = {"train": "train.csv", "test": "test.csv"}

        # Expected data shapes
        self.expected_shapes = {
            "train": (25000, 25),  # 25,000 samples, 25 features (including target)
            "test": (5000, 24),  # 5,000 samples, 24 features (no target)
        }

        # Create data directory
        self.data_dir.mkdir(exist_ok=True)

        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def download_file(self, url: str, filename: str) -> bool:
        """
        Download a file from the given URL.

        Args:
            url (str): URL to download from
            filename (str): Name to save the file as

        Returns:
            bool: True if download successful, False otherwise
        """
        filepath = self.data_dir / filename

        try:
            logger.info(f"Downloading {filename} from {url}")

            # Make the request
            response = self.session.get(url, timeout=self.download_timeout, stream=True)
            response.raise_for_status()

            # Check if it's a redirect to actual file
            if "text/html" in response.headers.get("content-type", ""):
                logger.warning(
                    f"Received HTML response from {url}. This might be a redirect page."
                )
                # Try to extract actual download link from HTML
                actual_url = self._extract_download_link(response.text, url)
                if actual_url:
                    logger.info(f"Extracted actual download URL: {actual_url}")
                    response = self.session.get(
                        actual_url, timeout=self.download_timeout, stream=True
                    )
                    response.raise_for_status()
                else:
                    logger.error(
                        "Could not extract actual download URL from HTML response"
                    )
                    return False

            # Download the file
            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Progress logging
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024 * 1024) == 0:  # Log every MB
                                logger.info(
                                    f"Downloaded {downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)"
                                )

            logger.info(f"Successfully downloaded {filename}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {filename}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def _extract_download_link(self, html_content: str, base_url: str) -> Optional[str]:
        """
        Extract actual download link from HTML response.

        Args:
            html_content (str): HTML content to parse
            base_url (str): Base URL for relative links

        Returns:
            Optional[str]: Actual download URL if found
        """
        try:
            # Look for common download link patterns
            import re

            # Pattern 1: Look for direct file links
            file_patterns = [
                r'href=["\']([^"\']*\.(?:csv|zip|gz))["\']',
                r'href=["\']([^"\']*download[^"\']*)["\']',
                r'href=["\']([^"\']*file[^"\']*)["\']',
            ]

            for pattern in file_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if match.startswith("http"):
                            return match
                        else:
                            # Construct absolute URL
                            parsed_base = urlparse(base_url)
                            if match.startswith("/"):
                                return f"{parsed_base.scheme}://{parsed_base.netloc}{match}"
                            else:
                                return f"{parsed_base.scheme}://{parsed_base.netloc}/{match}"

            # Pattern 2: Look for JavaScript redirects
            js_patterns = [
                r'window\.location\s*=\s*["\']([^"\']*)["\']',
                r'location\.href\s*=\s*["\']([^"\']*)["\']',
                r'redirect\s*\(["\']([^"\']*)["\']',
            ]

            for pattern in js_patterns:
                matches = re.findall(pattern, html_content)
                if matches:
                    for match in matches:
                        if match.startswith("http"):
                            return match
                        else:
                            # Construct absolute URL
                            parsed_base = urlparse(base_url)
                            if match.startswith("/"):
                                return f"{parsed_base.scheme}://{parsed_base.netloc}{match}"
                            else:
                                return f"{parsed_base.scheme}://{parsed_base.netloc}/{match}"

            logger.warning("No download link patterns found in HTML response")
            return None

        except Exception as e:
            logger.error(f"Error extracting download link: {e}")
            return None

    def check_data_exists(self) -> bool:
        """
        Check if all required data files already exist.

        Returns:
            bool: True if all files exist, False otherwise
        """
        logger.info("Checking if data files already exist...")

        for data_type, filename in self.expected_files.items():
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.info(f"File {filename} not found")
                return False

            # Check if file is not empty
            if filepath.stat().st_size == 0:
                logger.info(f"File {filename} is empty")
                return False

            logger.info(f"File {filename} exists and is not empty")

        logger.info("All required data files exist")
        return True

    def download_all_data(self) -> bool:
        """
        Download all required data files.

        Returns:
            bool: True if all downloads successful, False otherwise
        """
        logger.info("Starting data download process...")

        success_count = 0
        total_files = len(self.data_urls)

        for data_type, url in self.data_urls.items():
            filename = self.expected_files[data_type]

            if self.download_file(url, filename):
                success_count += 1
            else:
                logger.error(f"Failed to download {filename}")

        logger.info(
            f"Download completed: {success_count}/{total_files} files downloaded successfully"
        )
        return success_count == total_files

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate downloaded data files.

        Returns:
            Dict containing validation results
        """
        logger.info("Validating downloaded data...")

        validation_results = {
            "files_exist": {},
            "file_sizes": {},
            "data_shapes": {},
            "data_types": {},
            "missing_values": {},
            "target_distribution": {},
            "validation_passed": True,
        }

        for data_type, filename in self.expected_files.items():
            filepath = self.data_dir / filename

            # Check if file exists
            if not filepath.exists():
                validation_results["files_exist"][data_type] = False
                validation_results["validation_passed"] = False
                logger.error(f"File {filename} not found")
                continue

            validation_results["files_exist"][data_type] = True

            # Check file size
            file_size = filepath.stat().st_size
            validation_results["file_sizes"][data_type] = file_size

            if file_size == 0:
                validation_results["validation_passed"] = False
                logger.error(f"File {filename} is empty")
                continue

            # Load and validate data
            try:
                data = pd.read_csv(filepath)
                validation_results["data_shapes"][data_type] = data.shape

                # Check expected shape
                expected_shape = self.expected_shapes[data_type]
                if data.shape != expected_shape:
                    logger.warning(
                        f"Expected shape for {filename}: {expected_shape}, got: {data.shape}"
                    )

                # Check data types
                validation_results["data_types"][data_type] = data.dtypes.to_dict()

                # Check missing values
                missing_counts = data.isnull().sum()
                validation_results["missing_values"][
                    data_type
                ] = missing_counts.to_dict()

                # Check target distribution for training data
                if data_type == "train" and "Y" in data.columns:
                    target_counts = data["Y"].value_counts()
                    validation_results["target_distribution"] = {
                        "counts": target_counts.to_dict(),
                        "percentages": (target_counts / len(data) * 100).to_dict(),
                    }

                    # Log target distribution
                    logger.info(f"Target distribution in training data:")
                    for value, count in target_counts.items():
                        percentage = (count / len(data)) * 100
                        logger.info(
                            f"  Class {value}: {count:,} samples ({percentage:.1f}%)"
                        )

                logger.info(f"✓ {filename} validation passed")

            except Exception as e:
                validation_results["validation_passed"] = False
                logger.error(f"Error validating {filename}: {e}")

        return validation_results

    def create_data_summary(self) -> pd.DataFrame:
        """
        Create a comprehensive summary of the downloaded data.

        Returns:
            DataFrame containing data summary
        """
        logger.info("Creating data summary...")

        summary_data = []

        for data_type, filename in self.expected_files.items():
            filepath = self.data_dir / filename

            if not filepath.exists():
                continue

            try:
                data = pd.read_csv(filepath)

                for col in data.columns:
                    col_info = {
                        "Dataset": data_type,
                        "Feature": col,
                        "Data Type": str(data[col].dtype),
                        "Missing Values": data[col].isnull().sum(),
                        "Missing %": (data[col].isnull().sum() / len(data)) * 100,
                        "Unique Values": data[col].nunique(),
                        "Min": (
                            data[col].min()
                            if data[col].dtype in ["int64", "float64"]
                            else None
                        ),
                        "Max": (
                            data[col].max()
                            if data[col].dtype in ["int64", "float64"]
                            else None
                        ),
                        "Mean": (
                            data[col].mean()
                            if data[col].dtype in ["int64", "float64"]
                            else None
                        ),
                        "Std": (
                            data[col].std()
                            if data[col].dtype in ["int64", "float64"]
                            else None
                        ),
                    }
                    summary_data.append(col_info)

            except Exception as e:
                logger.error(f"Error creating summary for {filename}: {e}")

        return pd.DataFrame(summary_data)

    def save_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """
        Save validation results to a JSON file.

        Args:
            validation_results (Dict): Validation results to save
        """
        report_file = self.data_dir / "validation_report.json"

        # Add timestamp
        validation_results["timestamp"] = datetime.now().isoformat()
        validation_results["data_urls"] = self.data_urls

        try:
            with open(report_file, "w") as f:
                json.dump(validation_results, f, indent=2, default=str)
            logger.info(f"Validation report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the downloaded data.

        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        logger.info("Loading downloaded data...")

        train_file = self.data_dir / self.expected_files["train"]
        test_file = self.data_dir / self.expected_files["test"]

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                "Training or test data files not found. Run download_all_data() first."
            )

        try:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)

            logger.info(f"Training data loaded: {train_data.shape}")
            logger.info(f"Test data loaded: {test_data.shape}")

            return train_data, test_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def run_complete_pipeline(self) -> bool:
        """
        Run the complete data pipeline: download, validate, and prepare data.

        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING CREDIT CARD DATA PIPELINE")
        logger.info("=" * 60)

        # Step 1: Download data
        logger.info("Step 1: Downloading data files...")
        if not self.download_all_data():
            logger.error("Data download failed. Pipeline cannot continue.")
            return False

        # Step 2: Validate data
        logger.info("Step 2: Validating downloaded data...")
        validation_results = self.validate_data()

        if not validation_results["validation_passed"]:
            logger.error("Data validation failed. Please check the validation report.")
            self.save_validation_report(validation_results)
            return False

        # Step 3: Create data summary
        logger.info("Step 3: Creating data summary...")
        try:
            summary = self.create_data_summary()
            summary_file = self.data_dir / "data_summary.csv"
            summary.to_csv(summary_file, index=False)
            logger.info(f"Data summary saved to {summary_file}")
        except Exception as e:
            logger.warning(f"Could not create data summary: {e}")

        # Step 4: Save validation report
        logger.info("Step 4: Saving validation report...")
        self.save_validation_report(validation_results)

        # Step 5: Verify data can be loaded
        logger.info("Step 5: Verifying data can be loaded...")
        try:
            train_data, test_data = self.load_data()
            logger.info("✓ Data loading verification passed")
        except Exception as e:
            logger.error(f"Data loading verification failed: {e}")
            return False

        logger.info("=" * 60)
        logger.info("DATA PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Training data: {train_data.shape}")
        logger.info(f"Test data: {test_data.shape}")
        logger.info("=" * 60)

        return True


def main():
    """Main function to run the data pipeline."""
    # Initialize pipeline
    pipeline = CreditCardDataPipeline()

    # Run complete pipeline
    success = pipeline.run_complete_pipeline()

    if success:
        print("\n🎉 Data pipeline completed successfully!")
        print(f"📁 Data files are available in: {pipeline.data_dir.absolute()}")
        print("\n📋 Next steps:")
        print("   1. Run the credit card default detection scripts")
        print("   2. Check validation_report.json for data quality details")
        print("   3. Review data_summary.csv for feature information")
    else:
        print("\n❌ Data pipeline failed. Check the logs above for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
