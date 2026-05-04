from sklearn.metrics import accuracy_score, average_precision_score


def get_positive_class_scores(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_scores = get_positive_class_scores(model, X_test)

    acc = accuracy_score(y_test, test_pred)
    auc_pr = average_precision_score((y_test == "Y").astype(int), test_scores)

    print(f"{name} -> Accuracy: {acc}, AUC-PR: {auc_pr}")

    return {
        "Modell": name,
        "Teszt pontosság": acc,
        "Teszt AUC-PR": auc_pr
    }, test_scores