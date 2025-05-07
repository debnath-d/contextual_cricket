from pathlib import Path

import joblib
import numpy as np
import torch
from train import CricketModel
from xgboost import XGBRegressor


def main():
    user_input = np.array(
        [
            [617.526, 0.319, 0.014, 2794.971, -0.979, -0.042, 23.22],
            [1194.915, 0.206, 0.007, 62.0, 2.697, 0.3, 9.0],
            [3755.099, 1.266, 0.045, 2838.844, -3.201, -0.166, 19.333],
            [60.149, -0.37, -0.053, 392.511, -0.555, -0.034, 16.315],
            [1045.84, 0.16, 0.006, 282.71, -0.657, -0.016, 40.494],
        ]
    )

    # Load Torch model and scaler
    saved_models_dir = Path("models")
    torch_model_path = saved_models_dir / "best_torch_model.pt"
    scaler_path = saved_models_dir / "scaler.pkl"
    xgb_model_path = saved_models_dir / "xgboost_model.json"

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(user_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model = CricketModel(X_tensor.shape[1])
    model.load_state_dict(torch.load(torch_model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        torch_output = model(X_tensor).numpy().flatten()
    print("Torch model predictions:", torch_output)

    # Load XGBoost model
    xgb = XGBRegressor()
    xgb.load_model(xgb_model_path)
    xgb_output = xgb.predict(user_input)
    print("XGBoost model predictions:", xgb_output)


if __name__ == "__main__":
    main()
