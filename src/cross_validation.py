"""Validacion cruzada k-fold estratificada.

En vez de una sola particion train/test (hold-out), dividimos en k bloques
y rotamos cual usamos para probar. Asi cada dato se prueba una vez y el
resultado depende menos del azar. Estratificada = cada bloque mantiene la
misma proporcion de clases.
"""

import numpy as np


def stratified_kfold_indices(y, k: int = 5, seed: int = 42):
    # reparte los indices de cada clase entre los k folds
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        for fold_id, chunk in enumerate(np.array_split(idx_c, k)):
            folds[fold_id].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in folds]


def cross_validate(build_and_eval, y, k: int = 5, seed: int = 42, verbose: bool = True):
    # build_and_eval(train_idx, test_idx, fold_id) entrena y evalua un fold
    folds = stratified_kfold_indices(y, k=k, seed=seed)
    all_idx = np.arange(len(y))
    per_fold = []
    for fold_id, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(all_idx, test_idx)
        metrics = build_and_eval(train_idx, test_idx, fold_id)
        per_fold.append(metrics)
        if verbose:
            desc = "  ".join(f"{m}={v:.3f}" for m, v in metrics.items())
            print(f"  Fold {fold_id + 1}/{k}: {desc}")
    # media y desviacion estandar de cada metrica
    keys = per_fold[0].keys()
    summary = {m: (float(np.mean([f[m] for f in per_fold])),
                   float(np.std([f[m] for f in per_fold]))) for m in keys}
    return per_fold, summary


if __name__ == "__main__":
    from collections import Counter
    y = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    folds = stratified_kfold_indices(y, k=5)
    y = np.asarray(y)
    for i, f in enumerate(folds):
        print(f"Fold {i+1}: n={len(f)}  dist={dict(Counter(y[f]))}")
