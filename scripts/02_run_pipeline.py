"""Ejecuta todo el pipeline: preprocesa, vectoriza, clasifica y evalua."""

import sys
import os
import pandas as pd
import numpy as np

# para poder importar desde src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocessing import preprocess
from vsm import VectorSpaceModel
from classifier import CentroidCosineClassifier, cosine_similarity
from extraction import extract_durations
from evaluation import confusion_matrix, print_report


DATA_DIR = os.environ.get("APIT_DATA", os.path.join(ROOT, "data", "raw"))
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# --- corpus ---
def _read(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), engine="python",
                       encoding="latin-1", on_bad_lines="skip")


def build_corpus(per_class=None):
    # une las fuentes y balancea las clases
    frames = []

    dep = _read("Depression.csv")
    frames.append(dep[["TEXT"]].rename(columns={"TEXT": "text"}).assign(label="Depresion"))

    sui = _read("Suicadal_tendencies_data.csv")
    frames.append(sui[["TEXT"]].rename(columns={"TEXT": "text"}).assign(label="Suicida"))

    sch = _read("Reddit-Based_Schizophrenia_Detection_Dataset.csv")
    sch_pos = sch[sch["is_schizophrenic"].astype(str) == "1"].copy()
    sch_pos["text"] = sch_pos["title"].fillna("") + " " + sch_pos["selftext"].fillna("")
    frames.append(sch_pos[["text"]].assign(label="Esquizofrenia"))

    neu = _read("Neutral.csv")
    frames.append(neu[["TEXT"]].rename(columns={"TEXT": "text"}).assign(label="Neutral"))

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]

    # misma cantidad por clase
    if per_class is None:
        per_class = df["label"].value_counts().min()
    balanced = []
    for lab, g in df.groupby("label"):
        balanced.append(g.sample(min(len(g), per_class), random_state=RANDOM_SEED))
    df = pd.concat(balanced, ignore_index=True)
    return df.reset_index(drop=True)


def stratified_split(df, test_frac=0.3):
    # particion train/test manteniendo proporcion de clases
    train_idx, test_idx = [], []
    for _, g in df.groupby("label"):
        idx = g.sample(frac=1.0, random_state=RANDOM_SEED).index.tolist()
        cut = int(len(idx) * (1 - test_frac))
        train_idx += idx[:cut]
        test_idx += idx[cut:]
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def calibrate_threshold(clf, X_val, y_val, labels, grid=np.linspace(0.0, 0.3, 16)):
    # busca el umbral que da mejor recall macro
    from evaluation import per_class_metrics, macro_average
    best_t, best_recall = 0.0, -1.0
    for t in grid:
        clf.threshold = t
        pred = clf.predict(X_val)
        M = confusion_matrix(y_val, pred, labels)
        r = macro_average(per_class_metrics(M, labels))["recall"]
        if r > best_recall:
            best_recall, best_t = r, t
    return best_t, best_recall


def main():
    print(">> FASE 1: cargando y preprocesando corpus...")
    df = build_corpus()
    print(f"   Corpus balanceado: {len(df)} docs | {df['label'].value_counts().to_dict()}")

    df["tokens"] = df["text"].astype(str).apply(preprocess)
    df = df[df["tokens"].map(len) > 0]

    train_df, test_df = stratified_split(df)
    labels = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]

    print(">> FASE 1: construyendo VSM (TF-IDF desde cero)...")
    vsm = VectorSpaceModel(sublinear_tf=True, max_features=5000, min_df=3)
    X_train = vsm.fit_transform(train_df["tokens"].tolist())
    X_test = vsm.transform(test_df["tokens"].tolist())
    print(f"   Vocabulario: {len(vsm.vocabulary_)} términos | matriz {X_train.shape}")

    print(">> FASE 2: entrenando centroides y calibrando umbral por Recall...")
    clf = CentroidCosineClassifier(neutral_label="Neutral").fit(X_train, train_df["label"].tolist())
    best_t, best_r = calibrate_threshold(clf, X_train, train_df["label"].tolist(), labels)
    clf.threshold = best_t
        # umbral mas bajo solo para Suicida (ver script 03)
    clf.class_thresholds = {"Suicida": 0.04}
    print(f"   Umbral global = {best_t:.3f} | umbral Suicida = 0.04 (recall macro train = {best_r:.2f})")

    for c in ["Depresion", "Suicida", "Esquizofrenia"]:
        terms = [t for t, _ in vsm.top_terms(clf.centroids_[c], 10)]
        print(f"   Centroide {c}: {terms}")

    print(">> FASE 4: evaluando sobre TEST (datos no vistos)...")
    pred = clf.predict(X_test)
    M = confusion_matrix(test_df["label"].tolist(), pred, labels)
    print()
    print_report(M, labels)

    print("\n>> FASE 3: extracción de duración en posts marcados como trastorno...")
    marked = test_df[test_df["label"].isin(["Depresion", "Suicida", "Esquizofrenia"])]
    found = 0
    for _, row in marked.iterrows():
        durs = extract_durations(str(row["text"]).lower())
        if durs and found < 5:
            print(f"   [{row['label']}] {durs}")
            found += 1
        if found >= 5:
            break


if __name__ == "__main__":
    main()
