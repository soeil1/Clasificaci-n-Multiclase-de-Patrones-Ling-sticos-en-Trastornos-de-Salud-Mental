"""Clasificacion por centroides y similitud coseno."""

import numpy as np


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # cos = (A.B) / (||A|| ||B||)
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    num = A @ B.T
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    denom = norm_A @ norm_B.T
    denom[denom == 0] = 1e-12
    return num / denom


class CentroidCosineClassifier:

    def __init__(self, threshold: float = 0.05, neutral_label="Neutral",
                 class_thresholds: dict | None = None):
        # threshold: coseno minimo para aceptar una clase; si no llega -> Neutral
        # class_thresholds: umbral distinto por clase (bajamos el de Suicida)
        self.threshold = threshold
        self.neutral_label = neutral_label
        self.class_thresholds = class_thresholds or {}
        self.centroids_: dict = {}
        self.classes_: list = []

    def _thr_for(self, clase) -> float:
        return self.class_thresholds.get(clase, self.threshold)

    def fit(self, X: np.ndarray, y: list) -> "CentroidCosineClassifier":
        # el centroide de cada clase es el promedio de sus vectores
        y = np.asarray(y)
        self.classes_ = [c for c in np.unique(y) if c != self.neutral_label]
        for c in self.classes_:
            self.centroids_[c] = X[y == c].mean(axis=0)
        return self

    def decision_scores(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        centroid_matrix = np.vstack([self.centroids_[c] for c in self.classes_])
        scores = cosine_similarity(X, centroid_matrix)
        return scores, self.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        # se queda con la clase de mayor coseno; si no pasa el umbral -> Neutral
        scores, classes = self.decision_scores(X)
        best_idx = scores.argmax(axis=1)
        best_val = scores.max(axis=1)
        preds = np.array([classes[i] for i in best_idx], dtype=object)
        for k in range(len(preds)):
            if best_val[k] < self._thr_for(preds[k]):
                preds[k] = self.neutral_label
        return preds


if __name__ == "__main__":
    X = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    y = ["A", "B", "A"]
    clf = CentroidCosineClassifier(threshold=0.3).fit(X, y)
    print("Centroides:", {k: np.round(v, 2) for k, v in clf.centroids_.items()})
    print("Pred:", clf.predict(np.array([[1, 0, 0], [0, 0.05, 0]])))
