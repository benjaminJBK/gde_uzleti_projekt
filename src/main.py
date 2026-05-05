import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from config import TRAIN_PATH, RANDOM_STATE
from preprocessing import build_preprocessor
from models import create_models
from data_quality import analyze_outliers
from plots import *
from evaluation import evaluate_model


def main():
    df = pd.read_csv(TRAIN_PATH)

    plot_target_distribution(df)
    plot_missing_values(df)

    out_df = analyze_outliers(df)
    plot_outlier_summary(out_df)

    plot_boxplots(df)
    plot_histograms(df)
    plot_correlation(df)

    loan_ids = df["Loan_ID"]

    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X,
        y,
        loan_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_valid, X_test, y_valid, y_test, ids_valid, ids_test = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    preprocessor = build_preprocessor(X)
    models = create_models(preprocessor)

    results = []

    prediction_summary = pd.DataFrame({
        "Loan_ID": ids_test.values,
        "Actual": y_test.values
    })

    for name, model in models.items():
        res, scores = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )

        results.append(res)

        prediction_summary[name] = model.predict(X_test)

        plot_precision_recall_curve(
            (y_test == "Y").astype(int),
            scores,
            name
        )

    prediction_summary.to_csv("model_prediction_summary.csv", index=False)

    print("\nModellek predikciós összesítője:")
    print(prediction_summary.head())
    print("\nCSV mentve: model_prediction_summary.csv")

    results_df = pd.DataFrame(results)
    print(results_df)

    plot_model_comparison(results_df)
    plt.show()


if __name__ == "__main__":
    main()