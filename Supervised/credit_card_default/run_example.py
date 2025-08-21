#!/usr/bin/env python3
"""
Example script demonstrating how to use the Credit Card Default Detection Pipeline.

This script shows different ways to run the pipeline programmatically.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from main import CreditCardDefaultPipeline


def example_full_pipeline():
    """Example: Run the complete pipeline."""
    print("🚀 Running Full Pipeline Example...")

    pipeline = CreditCardDefaultPipeline()

    # Run the complete pipeline
    success = pipeline.run_full_pipeline(
        force_download=False,  # Don't force re-download
        use_grid_search=True,  # Use grid search for hyperparameter tuning
        input_data_path=None,  # Use test data for inference
    )

    if success:
        print("✅ Full pipeline completed successfully!")
    else:
        print("❌ Full pipeline failed!")

    return success


def example_specific_phases():
    """Example: Run specific phases of the pipeline."""
    print("🎯 Running Specific Phases Example...")

    pipeline = CreditCardDefaultPipeline()

    # Phase 1: Data Collection
    print("\n📥 Phase 1: Data Collection")
    if pipeline.run_data_collection(force_download=False):
        print("✅ Data collection completed")
    else:
        print("❌ Data collection failed")
        return False

    # Phase 2: Data Preprocessing
    print("\n🔧 Phase 2: Data Preprocessing")
    if pipeline.run_data_preprocessing():
        print("✅ Data preprocessing completed")
    else:
        print("❌ Data preprocessing failed")
        return False

    # Phase 3: Model Training
    print("\n🤖 Phase 3: Model Training")
    if pipeline.run_model_training(use_grid_search=True):
        print("✅ Model training completed")
    else:
        print("❌ Model training failed")
        return False

    # Phase 4: Inference
    print("\n🔮 Phase 4: Inference")
    if pipeline.run_inference():
        print("✅ Inference completed")
    else:
        print("❌ Inference failed")
        return False

    # Generate report
    print("\n📊 Generating Report")
    if pipeline.generate_report():
        print("✅ Report generated")
    else:
        print("❌ Report generation failed")

    return True


def example_custom_config():
    """Example: Run pipeline with custom configuration."""
    print("⚙️ Running Custom Configuration Example...")

    # Custom configuration overrides
    config_overrides = {
        "custom_setting": "custom_value",
        "model_params": {"random_state": 123, "test_size": 0.3},
    }

    pipeline = CreditCardDefaultPipeline(config_overrides)

    # Run only the training phase with custom config
    success = pipeline.run_specific_phase(
        "training", use_grid_search=False  # Skip grid search for faster execution
    )

    if success:
        print("✅ Custom configuration pipeline completed!")
    else:
        print("❌ Custom configuration pipeline failed!")

    return success


def main():
    """Main function to run examples."""
    print("🎬 Credit Card Default Detection Pipeline Examples")
    print("=" * 50)

    examples = [
        ("Full Pipeline", example_full_pipeline),
        ("Specific Phases", example_specific_phases),
        ("Custom Configuration", example_custom_config),
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
        print("-" * 30)

        try:
            success = func()
            if success:
                print(f"✅ {name} completed successfully!")
            else:
                print(f"❌ {name} failed!")
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")

        print()

    print("🎉 All examples completed!")


if __name__ == "__main__":
    main()
