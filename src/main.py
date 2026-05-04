import pandas as pd
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

    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    preprocessor = build_preprocessor(X)
    models = create_models(preprocessor)

    results = []

    for name, model in models.items():
        res, scores = evaluate_model(
            name, model,
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test
        )

        results.append(res)
        plot_precision_recall_curve((y_test == "Y").astype(int), scores, name)

    results_df = pd.DataFrame(results)
    print(results_df)

    plot_model_comparison(results_df)
    plt.show()


if __name__ == "__main__":
    main()