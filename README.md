# Context-Aware Performance Modeling in Cricket

This project analyzes T20 cricket data, computes player and matchup statistics, and trains machine learning models to predict match outcomes.

## Dataset

https://cricsheet.org/downloads/t20s_male_json.zip

## File Structure
- `train.py`: Data processing, feature engineering, model training, and evaluation.
- `infer.py`: Loads trained models and makes predictions on sample data.
- `models/`: Saved models and scalers.
- `plots/`: Generated plots.

## Usage

1. **Train models and generate plots:**
   ```bash
   python train.py
   ```

2. **Run inference with trained models:**
   ```bash
   python infer.py
   ```

Ensure required dependencies are installed (see imports in scripts).