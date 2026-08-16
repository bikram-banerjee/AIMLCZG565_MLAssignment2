import sys
import pandas as pd
import joblib

def run_inference(model_name: str, input_csv: str, output_csv: str):
    pipe = joblib.load(f"model/{model_name}.joblib")
    df = pd.read_csv(input_csv)

    df["prediction"] = pipe.predict(df)
    df["prob_subscribe"] = pipe.predict_proba(df)[:, 1]
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    # python model/predict.py RandomForest input.csv output.csv
    if len(sys.argv) != 4:
        print("Usage: python model/predict.py <ModelName> <input.csv> <output.csv>")
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2], sys.argv[3])