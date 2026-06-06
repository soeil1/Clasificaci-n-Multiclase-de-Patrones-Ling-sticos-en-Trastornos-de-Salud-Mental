# Clasificación Multiclase de Patrones Lingüísticos en Trastornos de Salud Mental

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-VSM%20desde%20cero-013243)
![Status](https://img.shields.io/badge/F1%20macro-0.79-success)
![License](https://img.shields.io/badge/License-MIT-green)

Proyecto final — *Análisis y Procesamiento Inteligente de Textos* (APIT), FI-UNAM.

Sistema de IR/NLP **clásico** (sin ML entrenado) que clasifica posts de Reddit en
**Neutral, Depresión, Ideación Suicida, Esquizofrenia** mediante el Modelo de
Espacio Vectorial (TF-IDF) y similitud coseno, con extracción de información por
regex/FST y evaluación rigurosa.

> 📖 **Leer `GUIA.md` para la explicación de uso, archivo por archivo.**
> 📄 **El reporte completo tambien está en `reports/TrabajoFinal_DeteccionDeTrastornos.pdf`.**

## Arquitectura (4 fases)
1. **VSM / BoW** (`preprocessing.py`, `vsm.py`): TF-IDF desde cero con NumPy.
2. **Clasificación por coseno** (`classifier.py`): centroides mas argmax mas umbral→Neutral.
3. **Extracción** (`extraction.py`): regex mas FST para normalizar duración de síntomas.
4. **Evaluación** (`evaluation.py`): matriz de confusión mas P/R/F1 desde cero (énfasis en Recall).

## Uso rápido
```bash
pip install -r requirements.txt
# coloca los 4 CSV en data/raw/
python scripts/02_run_pipeline.py
```
CSV: `Depression.csv`, `Suicadal_tendencies_data.csv`, `Neutral.csv`,
`Reddit-Based_Schizophrenia_Detection_Dataset.csv`.

## Resultados (test 30%, 18 292 docs balanceados)
| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Neutral | 0.66 | 0.79 | 0.72 |
| Depresión | 0.88 | 0.89 | 0.88 |
| Suicida | 0.76 | 0.72 | 0.74 |
| Esquizofrenia | 0.87 | 0.74 | 0.80 |
| **Macro avg** | **0.79** | **0.79** | **0.79** |

## Estructura
```
proyecto_apit/
├── README.md / GUIA.md / requirements.txt
├── data/raw/          
├── data/processed/    <- corpus.csv generado
├── src/               <- núcleo (config, preprocessing, vsm, classifier, extraction, evaluation)
├── scripts/           <- 01_build_corpus, 02_run_pipeline, make_figures, build_notebook
├── notebooks/         <- exploracion.ipynb
└── reports/figuras/   
```
