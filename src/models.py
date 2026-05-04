from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def create_models(preprocessor):
    return {
        "SVM": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(probability=True, random_state=42))
        ]),
        "Logisztikus regresszió": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Döntési fa": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
        ])
    }