import pandas as pd


def analyze_outliers(df):
    numeric_cols = df.select_dtypes(exclude=["object"]).columns
    outlier_summary = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        lower_out = series[series < lower]
        upper_out = series[series > upper]

        print(f"\n--- {col} ---")
        print(f"Alsó határ: {lower:.2f}")
        print(f"Felső határ: {upper:.2f}")
        print(f"Alsó túllépések: {len(lower_out)}")
        print(f"Felső túllépések: {len(upper_out)}")

        outlier_summary.append({
            "Oszlop": col,
            "Alsó határ": lower,
            "Felső határ": upper,
            "Alsó túllépések": len(lower_out),
            "Felső túllépések": len(upper_out)
        })

    return pd.DataFrame(outlier_summary)