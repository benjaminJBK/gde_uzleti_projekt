import matplotlib.pyplot as plt


def plot_target_distribution(df):
    df["Loan_Status"].value_counts().plot(kind="bar")
    plt.title("Kimeneti eloszlás")
    plt.xlabel("Kategória")
    plt.ylabel("Darabszám")
    plt.tight_layout()


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nHiányzó értékek:")
    print(missing)

    missing.plot(kind="bar")
    plt.title("Hiányzó értékek száma")
    plt.xlabel("Oszlop")
    plt.ylabel("Darabszám")
    plt.tight_layout()


def plot_outlier_summary(df):
    if df.empty:
        return

    df.set_index("Oszlop")[["Alsó túllépések", "Felső túllépések"]].plot(kind="bar")
    plt.title("Outlierek száma")
    plt.xlabel("Oszlop")
    plt.ylabel("Darabszám")
    plt.xticks(rotation=30)
    plt.tight_layout()


def plot_boxplots(df):
    df.select_dtypes(exclude=["object"]).plot(kind="box")
    plt.title("Boxplot")
    plt.ylabel("Érték")
    plt.xticks(rotation=30)
    plt.tight_layout()


def plot_histograms(df):
    df.select_dtypes(exclude=["object"]).hist(figsize=(10, 8))
    plt.suptitle("Hisztogramok")
    plt.tight_layout()


def plot_correlation(df):
    numeric_df = df.select_dtypes(exclude=["object"])
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Korrelációs mátrix")

    # értékek kiírása
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)

    plt.tight_layout()


def plot_model_comparison(df):
    df.set_index("Modell")[["Teszt pontosság", "Teszt AUC-PR"]].plot(kind="bar")

    plt.title("Modellek összehasonlítása")
    plt.ylabel("Érték")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()


def plot_precision_recall_curve(y_true, y_scores, name):
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure("Precision-Recall görbe")
    plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall görbe")
    plt.legend()
    plt.tight_layout()