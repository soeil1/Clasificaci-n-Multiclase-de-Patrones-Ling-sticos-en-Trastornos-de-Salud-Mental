"""Matriz de confusion y metricas: precision, recall y F1.

Optimizamos recall porque en deteccion de riesgo un falso negativo
(no detectar a alguien en peligro) es mucho mas grave que un falso positivo.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    # M[i][j] = ejemplos de la clase real i predichos como j
    idx = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def per_class_metrics(M: np.ndarray, labels) -> dict:
    metrics = {}
    total = M.sum()
    for i, lab in enumerate(labels):
        TP = M[i, i]
        FP = M[:, i].sum() - TP
        FN = M[i, :].sum() - TP
        TN = total - TP - FP - FN
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        metrics[lab] = {"TP": int(TP), "FP": int(FP), "FN": int(FN),
                        "TN": int(TN), "precision": P, "recall": R, "f1": F1}
    return metrics


def macro_average(metrics: dict) -> dict:
    P = np.mean([m["precision"] for m in metrics.values()])
    R = np.mean([m["recall"] for m in metrics.values()])
    F1 = np.mean([m["f1"] for m in metrics.values()])
    return {"precision": P, "recall": R, "f1": F1}


def print_report(M: np.ndarray, labels) -> dict:
    print("Matriz de Confusion (fila=real, col=predicho):")
    header = "real\\pred".ljust(14) + "".join(l[:10].ljust(12) for l in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = lab[:12].ljust(14) + "".join(str(M[i, j]).ljust(12) for j in range(len(labels)))
        print(row)
    print()
    metrics = per_class_metrics(M, labels)
    print("Clase".ljust(16) + "Prec".ljust(8) + "Recall".ljust(8) + "F1".ljust(8) + "FN")
    for lab in labels:
        m = metrics[lab]
        print(lab[:14].ljust(16) + f"{m['precision']:.2f}".ljust(8) +
              f"{m['recall']:.2f}".ljust(8) + f"{m['f1']:.2f}".ljust(8) + str(m["FN"]))
    macro = macro_average(metrics)
    print("-" * 48)
    print("MACRO AVG".ljust(16) + f"{macro['precision']:.2f}".ljust(8) +
          f"{macro['recall']:.2f}".ljust(8) + f"{macro['f1']:.2f}")
    return metrics


if __name__ == "__main__":
    labels = ["Neutral", "Depresion", "Suicida", "Esquizofrenia"]
    y_true = ["Suicida", "Suicida", "Depresion", "Neutral", "Esquizofrenia", "Suicida"]
    y_pred = ["Suicida", "Neutral",  "Depresion", "Neutral", "Esquizofrenia", "Depresion"]
    M = confusion_matrix(y_true, y_pred, labels)
    print_report(M, labels)
