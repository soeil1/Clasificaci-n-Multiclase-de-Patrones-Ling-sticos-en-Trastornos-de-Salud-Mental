"""Compara el clasificador por centroides con un baseline (Naive Bayes)
usando validacion cruzada 5-fold. En cada fold el VSM se ajusta solo con
el train para no hacer trampa."""
import sys, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("APIT_DATA", os.path.join(ROOT, "data", "raw"))

from preprocessing import preprocess
from vsm import VectorSpaceModel
from classifier import CentroidCosineClassifier
from naive_bayes import MultinomialNaiveBayes
from cross_validation import cross_validate
from evaluation import confusion_matrix, per_class_metrics, macro_average

import importlib.util
spec = importlib.util.spec_from_file_location("pipe", os.path.join(ROOT, "scripts", "02_run_pipeline.py"))
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)

LABELS = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]
K = 5
MAX_FEATURES = 5000
MIN_DF = 3


def _scores(y_true, y_pred):
    # metricas de un fold
    M = confusion_matrix(list(y_true), list(y_pred), LABELS)
    mt = per_class_metrics(M, LABELS)
    ma = macro_average(mt)
    return {"f1_macro": ma["f1"], "recall_macro": ma["recall"],
            "recall_suicida": mt["Suicida"]["recall"]}


def main():
    print(">> Cargando y preprocesando corpus...")
    df = pipe.build_corpus().reset_index(drop=True)
    df["tokens"] = df["text"].astype(str).apply(preprocess)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
    tokens = df["tokens"].tolist()
    y = df["label"].to_numpy()
    print(f"   Corpus: {len(df)} documentos | clases: {dict(zip(*np.unique(y, return_counts=True)))}")

    # fold del clasificador por centroides
    def eval_centroids(train_idx, test_idx, fold_id):
        vsm = VectorSpaceModel(sublinear_tf=True, max_features=MAX_FEATURES, min_df=MIN_DF)
        Xtr = vsm.fit_transform([tokens[i] for i in train_idx])
        Xte = vsm.transform([tokens[i] for i in test_idx])
        clf = CentroidCosineClassifier(neutral_label="Neutral").fit(Xtr, y[train_idx].tolist())
        bt, _ = pipe.calibrate_threshold(clf, Xtr, y[train_idx].tolist(), LABELS)
        clf.threshold = bt
        clf.class_thresholds = {"Suicida": 0.04}   # mejora del script 03
        return _scores(y[test_idx], clf.predict(Xte))

    # fold del baseline Naive Bayes
    def eval_nb(train_idx, test_idx, fold_id):
        vsm = VectorSpaceModel(sublinear_tf=True, max_features=MAX_FEATURES, min_df=MIN_DF)
        vsm.fit_transform([tokens[i] for i in train_idx])   # aprende vocabulario
        Xtr = vsm.count_transform([tokens[i] for i in train_idx])   # conteos para Naive Bayes
        Xte = vsm.count_transform([tokens[i] for i in test_idx])
        nb = MultinomialNaiveBayes(alpha=1.0).fit(Xtr, y[train_idx].tolist())
        return _scores(y[test_idx], nb.predict(Xte))

    print(f"\n>> Validación cruzada {K}-fold — CLASIFICADOR PROPUESTO (Centroides + Coseno)")
    cen_folds, cen_sum = cross_validate(eval_centroids, y, k=K)

    print(f"\n>> Validación cruzada {K}-fold — BASELINE (Naïve Bayes Multinomial)")
    nb_folds, nb_sum = cross_validate(eval_nb, y, k=K)

    # tabla comparativa
    print("\n" + "=" * 64)
    print("RESUMEN COMPARATIVO (media ± desviación estándar sobre los 5 folds)")
    print("=" * 64)
    print(f"{'Métrica':22s}{'Centroides':>20s}{'Naïve Bayes':>20s}")
    for key, name in [("f1_macro", "F1 macro"), ("recall_macro", "Recall macro"),
                      ("recall_suicida", "Recall Suicida")]:
        c_m, c_s = cen_sum[key]; n_m, n_s = nb_sum[key]
        print(f"{name:22s}{c_m:>10.3f} ± {c_s:.3f}{n_m:>10.3f} ± {n_s:.3f}")

    print("\nLa baja desviacion estandar indica que el resultado es estable.")
    print("Naive Bayes sirve como modelo base para comparar.")


if __name__ == "__main__":
    main()
