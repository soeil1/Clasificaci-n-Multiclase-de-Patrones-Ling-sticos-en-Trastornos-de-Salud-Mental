# Guia del proyecto

Sistema que clasifica posts de Reddit en cuatro clases (Neutral, Depresion,
Ideacion Suicida, Esquizofrenia) usando solo tecnicas clasicas de recuperacion
de informacion: TF-IDF, similitud coseno y automatas. Sin modelos entrenados
en el nucleo (salvo el baseline de Naive Bayes, que se hizo a mano para comparar).

## Estructura

```
proyecto_apit/
├── data/
│   ├── raw/          aqui van los 4 CSV (no se suben al repo)
│   └── processed/    corpus unificado
├── src/              codigo del sistema
│   ├── config.py             parametros (vocabulario, umbral, etc.)
│   ├── preprocessing.py      limpieza y tokenizacion
│   ├── vsm.py                bag of words y TF-IDF
│   ├── classifier.py         centroides + coseno
│   ├── naive_bayes.py        baseline Naive Bayes
│   ├── cross_validation.py   validacion cruzada k-fold
│   ├── extraction.py         regex + FST (duracion de sintomas)
│   └── evaluation.py         matriz de confusion y metricas
├── scripts/
│   ├── 01_build_corpus.py            arma el corpus
│   ├── 02_run_pipeline.py            corre todo el pipeline
│   ├── 03_mejora_recall_suicida.py   analisis de la clase Suicida
│   └── 04_baseline_y_cross_validation.py   comparacion con k-fold
├── notebooks/exploracion.ipynb
├── reports/          figuras y reporte
├── requirements.txt
└── README.md
```

## Como correrlo

```bash
pip install -r requirements.txt
# poner los 4 CSV en data/raw/
python scripts/02_run_pipeline.py
```

Cada modulo de src/ se puede correr solo para ver una demo:
`python src/vsm.py`, `python src/extraction.py`, etc.

## Que hace cada parte

- **preprocessing.py**: pasa el texto a minusculas, quita URLs, numeros y
  puntuacion, tokeniza y quita stopwords. Deja palabras como "never" o "empty"
  porque sirven para detectar depresion.
- **vsm.py**: construye la matriz TF-IDF a mano. TF = que tan seguido sale una
  palabra; IDF = que tan rara es. Se limita el vocabulario a 5000 palabras para
  no quedarnos sin memoria.
- **classifier.py**: cada clase es un centroide (el promedio de sus vectores).
  Un texto se asigna a la clase con mayor coseno; si ninguna pasa el umbral, va
  a Neutral.
- **extraction.py**: regex que detecta cuanto tiempo lleva la persona con
  sintomas ("for 3 years") y lo convierte a dias.
- **evaluation.py**: matriz de confusion, precision, recall y F1. Nos importa
  el recall porque un falso negativo es peligroso.

## Resultados

Con hold-out: F1 macro 0.79. La clase Depresion es la que mejor sale (F1 0.89)
y Suicida la que mas falla.

## Mejora de la clase Suicida (script 03)

Los falsos negativos de Suicida son textos muy cortos (10.7 palabras en
promedio, contra 98.3 de los aciertos). Al ser tan cortos, su vector casi no
tiene informacion y el coseno queda bajo. Bajamos el umbral solo para Suicida
(0.04 en vez de 0.08): el recall sube de 0.72 a 0.79 y recuperamos ~100 posts
de riesgo, perdiendo solo 2 puntos de F1 macro.

## Baseline y validacion cruzada (script 04)

Se agrego Naive Bayes como modelo base y se cambio el hold-out por validacion
cruzada 5-fold (ambos a mano).

| Metrica (media ± DE) | Centroides | Naive Bayes |
|---|---|---|
| F1 macro | 0.77 ± 0.004 | 0.81 ± 0.004 |
| Recall macro | 0.77 ± 0.005 | 0.82 ± 0.004 |
| Recall Suicida | 0.79 ± 0.013 | 0.84 ± 0.012 |

La desviacion estandar baja confirma que el resultado no fue suerte. Naive Bayes
gana por ~4 puntos porque aprende probabilidades, mientras que los centroides
solo promedian. Los dos coinciden en que Suicida es la clase mas dificil.
