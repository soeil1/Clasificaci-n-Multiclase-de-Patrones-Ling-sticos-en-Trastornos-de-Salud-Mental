import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/claude/proyecto_apit"
sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocessing import preprocess
from vsm import VectorSpaceModel
from classifier import CentroidCosineClassifier
from evaluation import confusion_matrix, per_class_metrics, macro_average
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("pipe", os.path.join(ROOT, "scripts", "02_run_pipeline.py"))
pipe = importlib.util.module_from_spec(spec)
# evita ejecutar main()
os.environ["APIT_DATA"] = os.path.join(ROOT, "data", "raw")
spec.loader.exec_module(pipe)

FIG = os.path.join(ROOT, "reports", "figuras")
LABELS = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]
COLORS = ["#6c8ebf", "#9673a6", "#b85450", "#82b366"]

df = pipe.build_corpus()
df["tokens"] = df["text"].astype(str).apply(preprocess)
df = df[df["tokens"].map(len) > 0]

# --- Fig 1: distribución de clases ---
vc = df["label"].value_counts().reindex(LABELS)
plt.figure(figsize=(6, 3.5))
plt.bar(vc.index, vc.values, color=COLORS)
plt.title("Distribución de documentos por clase (corpus balanceado)")
plt.ylabel("nº documentos"); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(os.path.join(FIG, "01_distribucion_clases.png"), dpi=130); plt.close()

train_df, test_df = pipe.stratified_split(df)
vsm = VectorSpaceModel(sublinear_tf=True, max_features=5000, min_df=3)
Xtr = vsm.fit_transform(train_df["tokens"].tolist())
Xte = vsm.transform(test_df["tokens"].tolist())
clf = CentroidCosineClassifier(neutral_label="Neutral").fit(Xtr, train_df["label"].tolist())
best_t, _ = pipe.calibrate_threshold(clf, Xtr, train_df["label"].tolist(), LABELS)
clf.threshold = best_t
pred = clf.predict(Xte)
M = confusion_matrix(test_df["label"].tolist(), pred, LABELS)

# --- Fig 2: matriz de confusión (heatmap) ---
plt.figure(figsize=(5.5, 5))
plt.imshow(M, cmap="Blues")
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
plt.yticks(range(len(LABELS)), LABELS)
plt.xlabel("Predicho"); plt.ylabel("Real")
plt.title(f"Matriz de Confusión (umbral={best_t:.3f})")
thr = M.max() / 2
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        plt.text(j, i, M[i, j], ha="center", va="center",
                 color="white" if M[i, j] > thr else "black")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "02_matriz_confusion.png"), dpi=130); plt.close()

# --- Fig 3: P/R/F1 por clase ---
mt = per_class_metrics(M, LABELS)
x = np.arange(len(LABELS)); w = 0.25
plt.figure(figsize=(7, 4))
plt.bar(x - w, [mt[l]["precision"] for l in LABELS], w, label="Precision")
plt.bar(x,     [mt[l]["recall"] for l in LABELS],    w, label="Recall")
plt.bar(x + w, [mt[l]["f1"] for l in LABELS],        w, label="F1")
plt.xticks(x, LABELS, rotation=15); plt.ylim(0, 1)
plt.title("Métricas por clase"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "03_metricas_clase.png"), dpi=130); plt.close()

# --- Fig 4: top términos por centroide ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, c, col in zip(axes, ["Depresion", "Suicida", "Esquizofrenia"], COLORS[1:]):
    terms = vsm.top_terms(clf.centroids_[c], 10)
    names = [t for t, _ in terms][::-1]; vals = [v for _, v in terms][::-1]
    ax.barh(names, vals, color=col); ax.set_title(f"Centroide {c}")
fig.suptitle("Términos de mayor peso TF-IDF por centroide")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "04_top_terminos.png"), dpi=130); plt.close()

macro = macro_average(mt)
print("Figuras generadas en reports/figuras/")
print("Macro:", {k: round(v, 3) for k, v in macro.items()})
print("FN por clase:", {l: mt[l]["FN"] for l in LABELS})
