import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from config import TRAIN_PATH, RANDOM_STATE
from preprocessing import build_preprocessor
from models import create_models


@pytest.fixture(scope="module")
def trained_models():
    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    models = create_models(preprocessor)

    for model in models.values():
        model.fit(X_train, y_train)

    return models


@pytest.fixture(scope="module")
def approved_rows():
    """15 olyan sor ahol hitelt kaptak (Loan_Status == Y)"""
    df = pd.read_csv(TRAIN_PATH)
    approved = df[df["Loan_Status"] == "Y"].head(15)
    return approved.drop(columns=["Loan_ID", "Loan_Status"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def rejected_rows():
    """15 olyan sor ahol nem kaptak hitelt (Loan_Status == N)"""
    df = pd.read_csv(TRAIN_PATH)
    rejected = df[df["Loan_Status"] == "N"].head(15)
    return rejected.drop(columns=["Loan_ID", "Loan_Status"]).reset_index(drop=True)


def test_income_increase_keeps_approval(trained_models, approved_rows):
    """
    MR2 — Jövedelem növelés monotonitás:
    Ha egy jóváhagyott kérelmen növeljük az ApplicantIncome-ot (kétszeresére),
    a modell predikciója nem változhat meg.
    """
    modified = approved_rows.copy()
    modified["ApplicantIncome"] = modified["ApplicantIncome"] * 2

    for model_name, model in trained_models.items():
        original_preds = model.predict(approved_rows)
        modified_preds = model.predict(modified)

        for i in range(len(approved_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: jövedelem növelés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti ApplicantIncome: {approved_rows.loc[i, 'ApplicantIncome']}, "
                f"módosított: {modified.loc[i, 'ApplicantIncome']})"
            )


def test_income_decrease_keeps_rejection(trained_models, rejected_rows):
    """
    MR2 — Jövedelem csökkentés monotonitás:
    Ha egy elutasított kérelmen csökkentjük az ApplicantIncome-ot (felére),
    a modell predikciója nem változhat meg.
    """
    modified = rejected_rows.copy()
    modified["ApplicantIncome"] = modified["ApplicantIncome"] * 0.5

    for model_name, model in trained_models.items():
        original_preds = model.predict(rejected_rows)
        modified_preds = model.predict(modified)

        for i in range(len(rejected_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: jövedelem csökkentés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti ApplicantIncome: {rejected_rows.loc[i, 'ApplicantIncome']}, "
                f"módosított: {modified.loc[i, 'ApplicantIncome']})"
            )


# ---------------------------------------------------------------------------
# MR1 — Sor-permutáció invariancia
# ---------------------------------------------------------------------------

def test_row_permutation_invariance(trained_models):
    """
    MR1 — Sor-permutáció invariancia:
    Ha a bemeneti sorok sorrendjét megkeverjük, az egyes sorokra adott
    predikció nem változhat. 15 vegyes sort használunk (jóváhagyott + elutasított).
    """
    df = pd.read_csv(TRAIN_PATH)
    sample = df.head(15).drop(columns=["Loan_ID", "Loan_Status"]).reset_index(drop=True)

    shuffled = sample.sample(frac=1, random_state=RANDOM_STATE)
    shuffled_order = shuffled.index.tolist()

    for model_name, model in trained_models.items():
        original_preds = model.predict(sample)
        shuffled_preds = model.predict(shuffled)

        # A shuffled sorait visszarendezzük az eredeti indexek szerint
        reordered_preds = pd.Series(shuffled_preds, index=shuffled_order).sort_index().values

        for i in range(len(sample)):
            assert original_preds[i] == reordered_preds[i], (
                f"[{model_name}] {i+1}. sor: sorcsere után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{reordered_preds[i]}'"
            )


# ---------------------------------------------------------------------------
# MR3 — Hitelösszeg monotonitás
# ---------------------------------------------------------------------------

def test_loan_amount_increase_keeps_rejection(trained_models, rejected_rows):
    """
    MR3 — Hitelösszeg növelés monotonitás:
    Ha egy elutasított kérelmen növeljük a LoanAmount-ot (kétszeresére),
    a predikció nem változhat meg.
    """
    modified = rejected_rows.copy()
    modified["LoanAmount"] = modified["LoanAmount"] * 2

    for model_name, model in trained_models.items():
        original_preds = model.predict(rejected_rows)
        modified_preds = model.predict(modified)

        for i in range(len(rejected_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: hitelösszeg növelés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti LoanAmount: {rejected_rows.loc[i, 'LoanAmount']}, "
                f"módosított: {modified.loc[i, 'LoanAmount']})"
            )


def test_loan_amount_decrease_keeps_approval(trained_models, approved_rows):
    """
    MR3 — Hitelösszeg csökkentés monotonitás:
    Ha egy jóváhagyott kérelmen csökkentjük a LoanAmount-ot (felére),
    a predikció nem változhat meg.
    """
    modified = approved_rows.copy()
    modified["LoanAmount"] = modified["LoanAmount"] * 0.5

    for model_name, model in trained_models.items():
        original_preds = model.predict(approved_rows)
        modified_preds = model.predict(modified)

        for i in range(len(approved_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: hitelösszeg csökkentés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti LoanAmount: {approved_rows.loc[i, 'LoanAmount']}, "
                f"módosított: {modified.loc[i, 'LoanAmount']})"
            )


# ---------------------------------------------------------------------------
# MR6 — Társ-kérelmező jövedelem monotonitás
# ---------------------------------------------------------------------------

def test_coapplicant_income_increase_keeps_approval(trained_models, approved_rows):
    """
    MR6 — Társ-kérelmező jövedelem növelés monotonitás:
    Ha egy jóváhagyott kérelmen növeljük a CoapplicantIncome-ot (+2000),
    a predikció nem változhat meg.
    Additív módosítást használunk, hogy a nullás értékű sorok is ténylegesen
    módosuljanak (0 * 2 = 0 nem változtat semmit).
    """
    modified = approved_rows.copy()
    modified["CoapplicantIncome"] = modified["CoapplicantIncome"] + 2000

    for model_name, model in trained_models.items():
        original_preds = model.predict(approved_rows)
        modified_preds = model.predict(modified)

        for i in range(len(approved_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: társ-kérelmező jövedelem növelés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti CoapplicantIncome: {approved_rows.loc[i, 'CoapplicantIncome']}, "
                f"módosított: {modified.loc[i, 'CoapplicantIncome']})"
            )


def test_coapplicant_income_decrease_keeps_rejection(trained_models, rejected_rows):
    """
    MR6 — Társ-kérelmező jövedelem csökkentés monotonitás:
    Ha egy elutasított kérelmen csökkentjük a CoapplicantIncome-ot (felére),
    a predikció nem változhat meg.
    Csak olyan sorokat vizsgálunk ahol CoapplicantIncome > 0,
    hogy a módosítás ténylegesen megtörténjen és ne kerüljön negatív tartományba.
    """
    rows = rejected_rows[rejected_rows["CoapplicantIncome"] > 0].reset_index(drop=True)
    modified = rows.copy()
    modified["CoapplicantIncome"] = modified["CoapplicantIncome"] * 0.5

    for model_name, model in trained_models.items():
        original_preds = model.predict(rows)
        modified_preds = model.predict(modified)

        for i in range(len(rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: társ-kérelmező jövedelem csökkentés után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti CoapplicantIncome: {rows.loc[i, 'CoapplicantIncome']}, "
                f"módosított: {modified.loc[i, 'CoapplicantIncome']})"
            )


# ---------------------------------------------------------------------------
# MR7 — Arányos skálázás invariancia
# ---------------------------------------------------------------------------

def test_proportional_income_loan_scaling(trained_models, approved_rows):
    """
    MR7 — Arányos skálázás stabilitás:
    Ha az ApplicantIncome-ot és a LoanAmount-ot egyszerre, azonos arányban növeljük
    (kétszeresére), a modell predikcióját stabilnak várjuk, mivel a két jellemző
    relatív viszonya nem változik. Ez a teszt a modell stabilitását ellenőrzi
    együttes, arányos perturbáció esetén.
    """
    modified = approved_rows.copy()
    modified["ApplicantIncome"] = modified["ApplicantIncome"] * 2
    modified["LoanAmount"] = modified["LoanAmount"] * 2

    for model_name, model in trained_models.items():
        original_preds = model.predict(approved_rows)
        modified_preds = model.predict(modified)

        for i in range(len(approved_rows)):
            assert modified_preds[i] == original_preds[i], (
                f"[{model_name}] {i+1}. sor: arányos skálázás után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(ApplicantIncome: {approved_rows.loc[i, 'ApplicantIncome']} -> {modified.loc[i, 'ApplicantIncome']}, "
                f"LoanAmount: {approved_rows.loc[i, 'LoanAmount']} -> {modified.loc[i, 'LoanAmount']})"
            )


# ---------------------------------------------------------------------------
# MR8 — Gender fairness invariancia
# ---------------------------------------------------------------------------

def test_gender_swap_fairness(trained_models):
    """
    MR8 — Gender fairness invariancia:
    Ha egy kérelmező nemét Male-ről Female-re (vagy fordítva) változtatjuk,
    minden más jellemző változatlan marad, a predikciónak nem szabad megváltoznia.
    A modell nem hozhat eltérő döntést pusztán a nem alapján.
    15 vegyes sort használunk.
    """
    df = pd.read_csv(TRAIN_PATH)
    sample = (
        df[df["Gender"].notna()]
        .head(15)
        .drop(columns=["Loan_ID", "Loan_Status"])
        .reset_index(drop=True)
    )

    modified = sample.copy()
    modified["Gender"] = modified["Gender"].map({"Male": "Female", "Female": "Male"})

    for model_name, model in trained_models.items():
        original_preds = model.predict(sample)
        modified_preds = model.predict(modified)

        for i in range(len(sample)):
            assert original_preds[i] == modified_preds[i], (
                f"[{model_name}] {i+1}. sor: nem csere után a predikció megváltozott: "
                f"'{original_preds[i]}' -> '{modified_preds[i]}' "
                f"(eredeti Gender: '{sample.loc[i, 'Gender']}', "
                f"módosított: '{modified.loc[i, 'Gender']}')"
            )


# ---------------------------------------------------------------------------
# MR5 — Duplikált sor konzisztencia
# ---------------------------------------------------------------------------

def test_duplicate_row_consistency(trained_models):
    """
    MR5 — Duplikált sor konzisztencia:
    Az eredeti és a duplikált sorokat egyetlen batch-be fűzzük, együtt predikálunk.
    Így azt ellenőrizzük, hogy a modell kontextus-független: egy sor predikciója
    nem függhet attól, hogy milyen más sorok vannak mellette a batch-ben.
    15 vegyes sort használunk.
    """
    df = pd.read_csv(TRAIN_PATH)
    sample = df.head(15).drop(columns=["Loan_ID", "Loan_Status"]).reset_index(drop=True)

    combined = pd.concat([sample, sample], ignore_index=True)

    for model_name, model in trained_models.items():
        preds = model.predict(combined)
        original_preds = preds[:len(sample)]
        duplicate_preds = preds[len(sample):]

        for i in range(len(sample)):
            assert original_preds[i] == duplicate_preds[i], (
                f"[{model_name}] {i+1}. sor: azonos sor eltérő predikciót kapott a batch-ben: "
                f"'{original_preds[i]}' vs '{duplicate_preds[i]}'"
            )
