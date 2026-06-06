"""Naive Bayes multinomial (modelo base para comparar).

Usa el teorema de Bayes asumiendo que las palabras son independientes.
Clasifica con la clase de mayor probabilidad:
    log P(c|d) = log P(c) + suma_t  tf(t,d) * log P(t|c)
Trabajamos en logaritmos para no multiplicar numeros muy pequenos.
"""

import numpy as np


class MultinomialNaiveBayes:

    def __init__(self, alpha: float = 1.0):
        # alpha = suavizado de Laplace, para que ninguna palabra tenga prob. 0
        self.alpha = alpha
        self.classes_: list = []
        self.log_prior_: dict = {}
        self.log_likelihood_: np.ndarray = None
        self.class_index_: dict = {}

    def fit(self, X_counts: np.ndarray, y: list) -> "MultinomialNaiveBayes":
        y = np.asarray(y)
        self.classes_ = sorted(np.unique(y).tolist())
        n_docs, V = X_counts.shape
        self.log_likelihood_ = np.zeros((len(self.classes_), V), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            self.class_index_[c] = i
            mask = (y == c)
            self.log_prior_[c] = np.log(mask.sum() / n_docs)         # P(c)
            term_counts = X_counts[mask].sum(axis=0)
            total_count = term_counts.sum()
            # P(t|c) con suavizado de Laplace
            smoothed = (term_counts + self.alpha) / (total_count + self.alpha * V)
            self.log_likelihood_[i] = np.log(smoothed)
        return self

    def predict(self, X_counts: np.ndarray) -> np.ndarray:
        log_priors = np.array([self.log_prior_[c] for c in self.classes_])
        scores = X_counts @ self.log_likelihood_.T + log_priors
        best = scores.argmax(axis=1)
        return np.array([self.classes_[i] for i in best], dtype=object)


if __name__ == "__main__":
    # vocabulario: [feliz, alegre, triste, vacio]
    X = np.array([[3, 2, 0, 0], [2, 3, 0, 0], [0, 0, 3, 2], [0, 0, 2, 3]])
    y = ["Pos", "Pos", "Neg", "Neg"]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(X, y)
    print("Clases:", nb.classes_)
    print("Pred:", nb.predict(np.array([[2, 2, 0, 0], [0, 0, 1, 2]])))
