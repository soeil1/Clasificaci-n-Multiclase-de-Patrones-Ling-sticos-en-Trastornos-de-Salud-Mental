"""Modelo de espacio vectorial: bag of words y TF-IDF."""

import numpy as np


class VectorSpaceModel:

    def __init__(self, sublinear_tf: bool = True, max_features: int | None = None,
                 min_df: int = 2):
        # max_features y min_df limitan el vocabulario para no quedarnos sin memoria
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _build_vocabulary(self, tokenized_docs: list[list[str]]) -> None:
        # cuenta en cuantos documentos aparece cada palabra
        df_counter: dict[str, int] = {}
        for doc in tokenized_docs:
            for token in set(doc):
                df_counter[token] = df_counter.get(token, 0) + 1
        items = [(t, c) for t, c in df_counter.items() if c >= self.min_df]
        items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            items = items[: self.max_features]
        sorted_terms = sorted(t for t, _ in items)
        self.vocabulary_ = {t: i for i, t in enumerate(sorted_terms)}

    def _compute_tf(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        N, V = len(tokenized_docs), len(self.vocabulary_)
        tf = np.zeros((N, V), dtype=np.float64)
        for d, doc in enumerate(tokenized_docs):
            for token in doc:
                j = self.vocabulary_.get(token)
                if j is not None:
                    tf[d, j] += 1.0
        if self.sublinear_tf:
            tf = np.where(tf > 0, 1.0 + np.log(tf, where=tf > 0), 0.0)
        return tf

    def _compute_idf(self, tf: np.ndarray) -> np.ndarray:
        # idf = log(N / df), con +1 para no dividir entre cero
        N = tf.shape[0]
        df = np.count_nonzero(tf > 0, axis=0)
        return np.log((1.0 + N) / (1.0 + df)) + 1.0

    def fit_transform(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        self._build_vocabulary(tokenized_docs)
        tf = self._compute_tf(tokenized_docs)
        self.idf_ = self._compute_idf(tf)
        return tf * self.idf_

    def transform(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        # vectoriza documentos nuevos con el vocabulario ya aprendido
        if self.idf_ is None:
            raise RuntimeError("Llama a fit_transform antes que transform.")
        N, V = len(tokenized_docs), len(self.vocabulary_)
        tf = np.zeros((N, V), dtype=np.float64)
        for d, doc in enumerate(tokenized_docs):
            for token in doc:
                j = self.vocabulary_.get(token)
                if j is not None:
                    tf[d, j] += 1.0
        if self.sublinear_tf:
            tf = np.where(tf > 0, 1.0 + np.log(tf, where=tf > 0), 0.0)
        return tf * self.idf_

    def top_terms(self, vector: np.ndarray, k: int = 15) -> list[tuple[str, float]]:
        # devuelve las k palabras con mayor peso en un vector
        inv_vocab = {idx: term for term, idx in self.vocabulary_.items()}
        top_idx = np.argsort(vector)[::-1][:k]
        return [(inv_vocab[i], float(vector[i])) for i in top_idx if vector[i] > 0]

    def count_transform(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        # conteos crudos (sin TF-IDF), que es lo que necesita Naive Bayes
        if not self.vocabulary_:
            raise RuntimeError("Llama a fit_transform antes que count_transform.")
        N, V = len(tokenized_docs), len(self.vocabulary_)
        counts = np.zeros((N, V), dtype=np.float64)
        for d, doc in enumerate(tokenized_docs):
            for token in doc:
                j = self.vocabulary_.get(token)
                if j is not None:
                    counts[d, j] += 1.0
        return counts


if __name__ == "__main__":
    docs = [["auto", "rojo"], ["auto", "veloz"]]
    vsm = VectorSpaceModel(sublinear_tf=False, min_df=1)
    m = vsm.fit_transform(docs)
    print("Vocabulario:", vsm.vocabulary_)
    print("IDF        :", np.round(vsm.idf_, 3))
    print("Matriz TF-IDF:\n", np.round(m, 3))
