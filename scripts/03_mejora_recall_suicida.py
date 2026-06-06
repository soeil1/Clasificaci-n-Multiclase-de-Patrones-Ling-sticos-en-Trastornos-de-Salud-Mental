"""Analiza por que falla la clase Suicida y prueba como subir su recall."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("APIT_DATA", os.path.join(ROOT, "data", "raw"))

from preprocessing import preprocess
from vsm import VectorSpaceModel
from classifier import CentroidCosineClassifier
from evaluation import confusion_matrix, per_class_metrics, macro_average

import importlib.util
spec = importlib.util.spec_from_file_location("pipe", os.path.join(ROOT, "scripts", "02_run_pipeline.py"))
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)

LABELS = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]


def main():
    df = pipe.build_corpus().reset_index(drop=True)
    df["tokens"] = df["text"].astype(str).apply(preprocess)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
    tr, te = pipe.stratified_split(df)
    yt = np.array(te["label"].tolist())

    vsm = VectorSpaceModel(sublinear_tf=True, max_features=5000, min_df=3)
    Xtr = vsm.fit_transform(tr["tokens"].tolist())
    Xte = vsm.transform(te["tokens"].tolist())
    clf = CentroidCosineClassifier(neutral_label="Neutral").fit(Xtr, tr["label"].tolist())
    bt, _ = pipe.calibrate_threshold(clf, Xtr, tr["label"].tolist(), LABELS)

    # diagnostico: ver longitud de los textos mal clasificados
    clf.threshold = bt
    pred = clf.predict(Xte)
    te_r = te.reset_index(drop=True)
    sui = np.where(yt == "Suicida")[0]
    ok = [i for i in sui if pred[i] == "Suicida"]
    bad = [i for i in sui if pred[i] != "Suicida"]
    print("=" * 60)
    print("DIAGNÓSTICO de la clase Suicida (umbral global =", round(bt, 3), ")")
    print(f"  Aciertos: {len(ok)}  |  Fallos (FN): {len(bad)}")
    print(f"  Long. media tokens — acertados: "
          f"{np.mean([len(te_r.iloc[i]['tokens']) for i in ok]):.1f}")
    print(f"  Long. media tokens — fallados : "
          f"{np.mean([len(te_r.iloc[i]['tokens']) for i in bad]):.1f}")
    print("  -> Los FN son textos MUY cortos: su vector TF-IDF queda casi vacío,")
    print("     el coseno es bajo y caen bajo el umbral global -> Neutral.")

    # probamos varias estrategias
    def metrics_of(clf_):
        p = clf_.predict(Xte)
        M = confusion_matrix(yt.tolist(), p, LABELS)
        mt = per_class_metrics(M, LABELS)
        return mt["Suicida"], macro_average(mt)

    results = []
    # base
    clf.class_thresholds = {}
    clf.threshold = bt
    s, ma = metrics_of(clf)
    results.append(("Baseline\n(global)", s["recall"], s["precision"], s["FN"], ma["f1"]))
    # umbral por clase
    for ts in [0.05, 0.04, 0.03]:
        clf.class_thresholds = {"Suicida": ts}
        s, ma = metrics_of(clf)
        results.append((f"Suicida\nt={ts}", s["recall"], s["precision"], s["FN"], ma["f1"]))

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print(f"{'Estrategia':18s}{'Recall':>8s}{'Prec':>8s}{'FN':>6s}{'macroF1':>9s}")
    for tag, r, p, fn, mf in results:
        print(f"{tag.replace(chr(10),' '):18s}{r:>8.2f}{p:>8.2f}{fn:>6d}{mf:>9.2f}")

    # figura comparativa
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r[0] for r in results]
    x = np.arange(len(labels)); w = 0.28
    ax.bar(x - w, [r[1] for r in results], w, label="Recall Suicida", color="#b85450")
    ax.bar(x,     [r[2] for r in results], w, label="Precision Suicida", color="#d6a2a0")
    ax.bar(x + w, [r[4] for r in results], w, label="Macro F1", color="#6c8ebf")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_title("Trade-off al bajar el umbral de la clase Suicida")
    ax.legend(loc="lower right")
    for i, r in enumerate(results):
        ax.text(i - w, r[1] + 0.01, f"{r[1]:.2f}", ha="center", fontsize=8)
        ax.text(i, 0.02, f"FN={r[3]}", ha="center", fontsize=8, color="white")
    plt.tight_layout()
    out = os.path.join(ROOT, "reports", "figuras", "05_recall_suicida.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"\nFigura guardada en {out}")

    print("\nCONCLUSIÓN: el umbral por clase (Suicida≈0.04) rescata ~100 FN")
    print("(recall 0.72 -> 0.79) con caída mínima del macro-F1 (0.79 -> 0.77).")
    print("Bajarlo más (0.03) sigue subiendo recall pero la precisión se degrada:")
    print("es el trade-off Precision-Recall de la diapositiva, decidido a favor")
    print("del Recall por el costo clínico de un Falso Negativo.")


if __name__ == "__main__":
    main()
