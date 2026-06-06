const fs = require("fs");
const U = require("./report_utils.js");
const {
  H1, H2, H3, P, bullet, b, t, it, code, caption, imgRatio, makeTable, ref,
  Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, Header, Footer, PageNumber, ACCENT
} = U;

const part1 = require("./report_part1.js");
const children = [...part1];

// ============================ 4. MONTAJE EXPERIMENTAL ============================
children.push(H1("4. Montaje Experimental"));
children.push(H2("4.1 Hardware"));
children.push(P("Los experimentos se ejecutaron en un entorno de cómputo de propósito general (CPU, sin GPU). La ausencia de requerimientos de aceleración por hardware es, en sí misma, una ventaja del enfoque clásico frente a los modelos neuronales, que dependen de GPU para entrenarse e inferir. El recurso crítico en nuestro caso es la memoria principal (RAM), debido a la representación densa de la matriz TF-IDF, cuyo dimensionamiento se discute en el Experimento 1 y en la Sección 7."));
children.push(H2("4.2 Software"));
children.push(P("El sistema se implementó en Python 3.12 con dependencias mínimas y deliberadas:"));
children.push(bullet([ b("NumPy. "), t("Todas las operaciones matriciales: construcción de TF, IDF, TF-IDF, centroides y similitud coseno. Es el único motor numérico del núcleo del modelo.") ]));
children.push(bullet([ b("pandas. "), t("Carga, unificación y balanceo de los archivos CSV que componen el corpus.") ]));
children.push(bullet([ b("matplotlib. "), t("Únicamente para la generación de las figuras del reporte.") ]));
children.push(P([ t("El módulo "), code("re"), t(" de la biblioteca estándar implementa las expresiones regulares y el transductor. En cumplimiento del objetivo pedagógico, no se utilizó "), code("scikit-learn"), t(" en el núcleo del modelo. El código se organizó siguiendo una separación profesional entre biblioteca y aplicación: un paquete "), code("src/"), t(" con módulos independientes (preprocesamiento, modelo vectorial, clasificador, extracción y evaluación) y un conjunto de scripts que los orquestan para ejecutar las cuatro fases de extremo a extremo, garantizando la reproducibilidad mediante una semilla aleatoria fija.") ]));
children.push(H2("4.3 Corpus y preprocesamiento"));
children.push(P("Se construyó un corpus unificado a partir de varias fuentes públicas de Reddit, asignando una etiqueta a cada publicación según su subreddit o conjunto de origen. Para evitar que los centroides quedaran sesgados hacia la clase mayoritaria y para que la exactitud no resultara engañosa, el corpus se balanceó a un número equivalente de documentos por clase mediante submuestreo aleatorio. La Tabla 1 resume su composición."));
children.push(makeTable(
  ["Clase", "Fuente principal", "Documentos"],
  [
    ["Neutral", "Publicaciones neutrales (humor, relaciones, crianza)", "4 558"],
    ["Depresión", "r/depression y conjunto de depresión", "4 493"],
    ["Ideación Suicida", "r/SuicideWatch y conjunto de tendencias suicidas", "4 548"],
    ["Esquizofrenia", "r/schizophrenia (conjunto etiquetado)", "4 560"],
    ["Total", "Corpus balanceado y unificado", "18 159"]
  ],
  [2500, 4360, 2500]
));
children.push(caption("Tabla 1. Composición del corpus balanceado de cuatro clases."));
children.push(P([ t("El preprocesamiento aplicó: conversión a minúsculas ("), it("case folding"), t("), eliminación de URLs, menciones, hashtags, números y puntuación mediante expresiones regulares, tokenización y eliminación de palabras vacías ("), it("stopwords"), t("). Una decisión de diseño relevante fue conservar deliberadamente términos como "), it("never, nothing, alone, empty"), t(" o "), it("tired"), t(", que algunas listas estándar eliminan, por constituir marcadores léxicos del dominio clínico cuya pérdida degradaría la señal. El corpus se dividió en 70 % para entrenamiento (12 710 documentos) y 30 % para prueba (5 449 documentos) mediante un muestreo estratificado que preserva la proporción de clases en ambos conjuntos.") ]));
children.push(...imgRatio("01_distribucion_clases.png", 380, 0.58, "Figura 1. Distribución de documentos por clase en el corpus balanceado."));
children.push(H2("4.4 Objetivo y protocolo de evaluación"));
children.push(P("El objetivo es asignar correctamente cada publicación a una de las cuatro clases. Toda la evaluación se realiza sobre el conjunto de prueba —datos no vistos durante la construcción de los centroides—, reportando precisión, exhaustividad y F1 por clase y su promedio macro. El umbral de rechazo hacia Neutral se calibró automáticamente sobre el conjunto de entrenamiento, buscando maximizar la exhaustividad macro, en coherencia con la prioridad de minimizar falsos negativos. Este protocolo evita la fuga de información y la sobrestimación del desempeño."));
children.push(H2("4.5 Experimentos"));
children.push(P("Se diseñaron cuatro experimentos incrementales, en correspondencia con las fases del sistema. Cada uno se describe por su material, su forma de evaluación y su resultado."));
children.push(H3("4.5.1 Experimento 1 — Representación vectorial (TF-IDF)"));
children.push(P([ b("Material: "), t("corpus tokenizado de entrenamiento. "), b("Evaluación: "), t("verificación de la matriz contra un ejemplo de control de las diapositivas y análisis de su dimensionamiento. "), b("Resultado: "), t("se obtuvo una matriz TF-IDF de 12 710 × 5 000. El vocabulario se limitó a los 5 000 términos más frecuentes, descartando los que aparecían en menos de tres documentos. Esta selección de características por frecuencia de documento no es un mero ajuste práctico: es necesaria para controlar la memoria de la matriz densa, cuyo tamaño crece con el producto documentos × vocabulario. Sin ella, una matriz de 18 000 documentos por ~30 000 términos excede la memoria disponible. Las bibliotecas profesionales evitan este problema con matrices dispersas; reproducir el límite manualmente es parte del aprendizaje del enfoque clásico.") ]));
children.push(H3("4.5.2 Experimento 2 — Clasificación por centroides y coseno"));
children.push(P([ b("Material: "), t("matriz TF-IDF de entrenamiento y etiquetas. "), b("Evaluación: "), t("F1 macro y métricas por clase sobre el conjunto de prueba. "), b("Resultado: "), t("se construyeron tres centroides (Depresión, Ideación Suicida, Esquizofrenia) y se clasificó por argmax con umbral. La inspección de los términos de mayor peso de cada centroide (Figura 2) muestra vocabulario coherente con cada trastorno, pero también revela el origen de futuras confusiones: términos genéricos del dominio ("), it("want, feel, life, know"), t(") aparecen con peso alto en varios centroides, anticipando el solapamiento entre Depresión e Ideación Suicida.") ]));
children.push(...imgRatio("04_top_terminos.png", 460, 0.31, "Figura 2. Términos de mayor peso TF-IDF en cada centroide de trastorno."));
children.push(H3("4.5.3 Experimento 3 — Extracción de información (Regex + FST)"));
children.push(P([ b("Material: "), t("publicaciones clasificadas como trastorno. "), b("Evaluación: "), t("revisión manual de las duraciones extraídas y su normalización. "), b("Resultado: "), t("el reconocedor (FSA) detecta expresiones de duración mediante agrupaciones, disparadores de contexto («for», «since», «past», entre otros), manejo de cantidades en cifra y en palabra (incluyendo cuantificadores vagos como «a couple» o «few») y verificación de frontera de palabra mediante "), it("lookahead"), t(". El transductor (FST) las normaliza a días: por ejemplo, «for nine years» → 3 285 días; «for six months» → 180 días; «last week» → 7 días. La función convierte texto libre en datos estructurados y comparables, cumpliendo el objetivo de la fase de extracción de información.") ]));
children.push(H3("4.5.4 Experimento 4 — Evaluación y mejora dirigida"));
children.push(P([ b("Material: "), t("predicciones del clasificador sobre el conjunto de prueba. "), b("Evaluación: "), t("matriz de confusión y métricas por clase. "), b("Resultado: "), t("se obtuvo el desempeño base reportado en la Sección 5 y se ejecutó un análisis dirigido de los errores de la clase Ideación Suicida, descrito en la Sección 6.") ]));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 5. RESULTADO GENERAL ============================
children.push(H1("5. Resultado General"));
children.push(P("El sistema, empleando únicamente técnicas clásicas y sin modelos entrenados, alcanzó un F1 macro de 0.79 sobre el conjunto de prueba. La Tabla 2 presenta las métricas por clase de la configuración base (umbral global)."));
children.push(makeTable(
  ["Clase", "Precisión", "Recall", "F1", "Falsos Neg."],
  [
    ["Neutral", "0.67", "0.79", "0.72", "284"],
    ["Depresión", "0.88", "0.89", "0.89", "143"],
    ["Ideación Suicida", "0.76", "0.72", "0.74", "384"],
    ["Esquizofrenia", "0.87", "0.74", "0.80", "358"],
    ["Macro", "0.79", "0.79", "0.79", "—"]
  ],
  [3000, 1640, 1640, 1640, 1440]
));
children.push(caption("Tabla 2. Métricas por clase de la configuración base (umbral global)."));
children.push(...imgRatio("02_matriz_confusion.png", 360, 0.92, "Figura 3. Matriz de confusión de la configuración base (filas = clase real, columnas = predicción)."));
children.push(...imgRatio("03_metricas_clase.png", 420, 0.57, "Figura 4. Precisión, exhaustividad y F1 por clase."));
children.push(H2("5.1 ¿Responde a la hipótesis?"));
children.push(P("Sí. La hipótesis principal sostenía que un enfoque clásico de RI podía lograr un desempeño útil. Un F1 macro de 0.79 sin entrenamiento de modelos lo confirma: el sistema distingue las cuatro clases muy por encima del azar, que para cuatro clases balanceadas sería de 0.25. La clase Depresión, con F1 de 0.89, se acerca notablemente al F1 de 0.93 reportado por Tadesse et al. (2019) con métodos supervisados, lo que demuestra que el vocabulario depresivo es especialmente separable en el espacio vectorial. La hipótesis secundaria también se sostiene: el desempeño no es uniforme, y las clases de riesgo (Ideación Suicida y Esquizofrenia) concentran la mayor cantidad de falsos negativos, lo que motiva el análisis de la siguiente sección."));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 6. ANÁLISIS DE RESULTADOS ============================
children.push(H1("6. Análisis de Resultados"));
children.push(P("El resultado más relevante desde la perspectiva clínica no es el promedio, sino dónde y por qué falla el sistema. La clase Ideación Suicida concentraba 384 falsos negativos: publicaciones de riesgo que el sistema etiquetó como Neutral. Dado que un falso negativo equivale a no señalar a una persona en peligro, se realizó un análisis dirigido para diagnosticar su causa y reducirlos."));
children.push(H2("6.1 Diagnóstico de la causa raíz"));
children.push(P("Al comparar las publicaciones de la clase Ideación Suicida correctamente clasificadas frente a las fallidas se observó una diferencia marcada en la longitud: los textos acertados tenían en promedio 98.3 tokens, mientras que los fallidos apenas 10.7. Ejemplos de falsos negativos fueron mensajes muy breves como «I’m giving it one week» o «Hey hit me up! I’m looking for someone to talk to!»."));
children.push(P("La explicación es geométrica y está documentada en la literatura. Un texto muy corto produce un vector TF-IDF con muy pocas dimensiones activas y norma pequeña; al medir su similitud coseno con cualquier centroide, el escaso solapamiento de términos arroja un valor bajo que cae por debajo del umbral global, enviando el documento a Neutral. De Boom et al. (2015) y trabajos sobre representación de textos muy cortos señalan precisamente que las medidas basadas en TF-IDF «fallan a menudo» en fragmentos breves, porque dependen del solapamiento exacto de palabras, que en textos cortos es raro o inexistente. Se trata, por tanto, de una limitación intrínseca del Modelo de Espacio Vectorial, vinculada a la pérdida de contexto de la Bolsa de Palabras, y no de un defecto de implementación."));
children.push(H2("6.2 Estrategias evaluadas"));
children.push(P("Se probaron varias estrategias, todas dentro del enfoque clásico, midiendo su efecto sobre la exhaustividad de Ideación Suicida y sobre el F1 macro global (Tabla 3)."));
children.push(makeTable(
  ["Estrategia", "Recall Sui.", "Precisión Sui.", "FN", "F1 macro"],
  [
    ["Base (umbral global 0.08)", "0.72", "0.76", "384", "0.79"],
    ["Incorporar bigramas", "0.70", "0.79", "412", "0.79"],
    ["Umbral global bajo (0.03)", "0.81", "0.66", "264", "0.70"],
    ["Umbral por clase (Suicida = 0.04)", "0.79", "0.67", "281", "0.77"]
  ],
  [3360, 1700, 1880, 800, 1620]
));
children.push(caption("Tabla 3. Comparación de estrategias para reducir los falsos negativos de Ideación Suicida."));
children.push(P("Los bigramas no mejoraron la exhaustividad —incluso la redujeron levemente—, lo que confirma que el problema es la escasez de palabras y no la falta de contexto bigrámico: si el texto tiene tres tokens, añadir pares de tokens no genera información nueva suficiente. Bajar el umbral global de forma uniforme elevó la exhaustividad de Ideación Suicida, pero degradó severamente el F1 macro (de 0.79 a 0.70) al introducir falsos positivos en todas las clases. La solución más equilibrada fue asignar un umbral específico más bajo únicamente a la clase Ideación Suicida, haciéndola más sensible sin relajar el criterio para el resto."));
children.push(...imgRatio("05_recall_suicida.png", 440, 0.56, "Figura 5. Compromiso entre exhaustividad, precisión y F1 macro al reducir el umbral de la clase Ideación Suicida."));
children.push(H2("6.3 Decisión adoptada y su justificación"));
children.push(P("Se adoptó un umbral por clase de 0.04 para Ideación Suicida (0.08 para el resto). El efecto se resume en la Tabla 4."));
children.push(makeTable(
  ["Métrica", "Base", "Con mejora"],
  [
    ["Exhaustividad (Recall) Suicida", "0.72", "0.79"],
    ["Falsos Negativos (Suicida)", "384", "281"],
    ["F1 macro global", "0.79", "0.77"]
  ],
  [3760, 2400, 2400]
));
children.push(caption("Tabla 4. Impacto de la mejora por umbral de clase."));
children.push(P("El sistema recuperó 103 publicaciones de riesgo a cambio de un descenso de apenas dos puntos en el F1 macro y cierta pérdida de precisión en la clase. Esta decisión materializa el compromiso precisión–exhaustividad estudiado en la asignatura, resuelto deliberadamente a favor de la exhaustividad por el elevado costo de un falso negativo en la detección de riesgo suicida. La elección de 0.04 —frente a 0.03, que elevaría aún más la exhaustividad— responde a no degradar excesivamente la precisión, manteniendo el sistema utilizable como herramienta de cribado y priorización para revisión humana, no como diagnóstico autónomo."));
children.push(H2("6.4 Sobre la confusión entre Depresión e Ideación Suicida"));
children.push(P("La matriz de confusión (Figura 3) muestra que parte de los errores de Ideación Suicida se desvían hacia Depresión, y viceversa. Este fenómeno no es un artefacto de nuestro sistema: el análisis a gran escala de Bauer et al. (2024) sobre 2.9 millones de publicaciones encontró que cerca del 13 % de los posts de r/SuicideWatch se sitúan más cerca del centroide de r/Depression que del propio. Ambas condiciones comparten un sustrato lingüístico de afecto negativo, desesperanza y aislamiento, lo que las hace intrínsecamente próximas en el espacio vectorial. Que nuestro sistema clásico reproduzca un patrón observado con embeddings neuronales en estudios recientes es una validación indirecta de su comportamiento."));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 7. BASELINE Y VALIDACIÓN CRUZADA ============================
children.push(H1("7. Validación Cruzada y Comparación con un Modelo Base"));
children.push(P("Para fortalecer la validez de los resultados se incorporaron dos mejoras metodológicas: la sustitución del protocolo hold-out por una validación cruzada estratificada, y la introducción de un modelo base supervisado (Naïve Bayes) como punto de comparación. Ambas se implementaron desde cero, en coherencia con el enfoque clásico del proyecto."));

children.push(H2("7.1 De hold-out a validación cruzada k-fold"));
children.push(P("El protocolo hold-out empleado en las secciones anteriores estima el desempeño sobre una única partición de prueba, por lo que el resultado depende del azar de esa partición concreta. La validación cruzada de k iteraciones (k-fold) divide el corpus en k bloques de tamaño similar; en cada iteración entrena con k-1 bloques y evalúa con el restante, rotando el bloque de prueba. De este modo cada documento se emplea exactamente una vez para prueba, y se obtienen k estimaciones cuya media es más robusta y cuya desviación estándar cuantifica la estabilidad del modelo. Se utilizó la variante estratificada (k = 5), que preserva la proporción de clases en cada bloque. Para evitar fuga de información, en cada iteración el Modelo de Espacio Vectorial se ajustó exclusivamente con los datos de entrenamiento del fold correspondiente."));

children.push(H2("7.2 Modelo base: Naïve Bayes Multinomial"));
children.push(P([ t("Como línea base supervisada se implementó un clasificador Naïve Bayes Multinomial (Manning et al., 2008, cap. 13). Este aplica el teorema de Bayes asumiendo independencia condicional entre términos dada la clase, y asigna cada documento a la clase de máxima probabilidad a posteriori (regla MAP). Las probabilidades se estiman por conteo con suavizado de Laplace y se operan en el espacio logarítmico para evitar desbordamientos numéricos:") ]));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: "log P(c | d) = log P(c) + Σ_t  tf(t,d) · log P(t | c)", italics: true, size: 24 })] }));
children.push(P([ t("A diferencia del clasificador por centroides —que promedia vectores TF-IDF—, Naïve Bayes estima parámetros probabilísticos a partir de conteos de términos, por lo que constituye un baseline supervisado legítimo. La comparación entre ambos permite valorar cuánto aporta un modelo que aprende distribuciones frente a uno geométrico basado en proximidad.") ]));

children.push(H2("7.3 Resultados comparativos"));
children.push(P("La Tabla 5 resume el desempeño de ambos clasificadores bajo validación cruzada 5-fold, expresado como media ± desviación estándar. La Figura 6 lo ilustra con barras de error."));
children.push(makeTable(
  ["Métrica (media ± DE, 5-fold)", "Centroides + Coseno", "Naïve Bayes (baseline)"],
  [
    ["F1 macro", "0.769 ± 0.004", "0.812 ± 0.004"],
    ["Recall macro", "0.769 ± 0.005", "0.816 ± 0.004"],
    ["Recall Ideación Suicida", "0.791 ± 0.013", "0.840 ± 0.012"]
  ],
  [3760, 2600, 2840]
));
children.push(caption("Tabla 5. Comparación de clasificadores bajo validación cruzada estratificada 5-fold."));
children.push(...imgRatio("06_baseline_comparacion.png", 450, 0.55, "Figura 6. Comparación del clasificador propuesto y el baseline Naïve Bayes (validación 5-fold; las barras de error indican la desviación estándar entre folds)."));

children.push(H2("7.4 Discusión de la comparación"));
children.push(P("Dos conclusiones se desprenden de estos resultados. Primero, la validación cruzada confirma la robustez del clasificador por centroides: su F1 macro de 0.77 presenta una desviación estándar de apenas 0.004 entre folds, lo que indica que el resultado reportado con hold-out (0.79) no fue producto del azar de una partición favorable, sino un comportamiento estable. Segundo, el baseline Naïve Bayes supera al clasificador por centroides en todas las métricas (F1 macro 0.81 frente a 0.77). Este resultado es esperable y pedagógicamente revelador: Naïve Bayes es un método supervisado que estima la distribución de probabilidad de cada término por clase, aprovechando la información de los datos de entrenamiento de forma más fina que el simple promedio vectorial del centroide, que descarta la varianza intra-clase. La brecha de aproximadamente cuatro puntos cuantifica el costo de la simplicidad e interpretabilidad del enfoque por centroides. Notablemente, ambos métodos coinciden en que la clase Ideación Suicida es de las más difíciles, lo que refuerza que su dificultad es intrínseca a los datos y no un artefacto de un clasificador particular."));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 8. DISCUSIÓN ============================
children.push(H1("8. Discusión, Consideraciones Éticas y Trabajo Futuro"));
children.push(H2("8.1 Limitaciones"));
children.push(bullet([ b("Fragilidad ante textos cortos. "), t("Como se demostró, el VSM produce vectores poco informativos para publicaciones muy breves, donde el solapamiento de términos con los centroides es escaso (De Boom et al., 2015).") ]));
children.push(bullet([ b("Pérdida de contexto y negación. "), t("La Bolsa de Palabras ignora el orden: «no estoy bien» y «estoy bien» comparten casi todo el vector pese a significar lo opuesto. El modelo no maneja la negación, la ironía ni la temporalidad.") ]));
children.push(bullet([ b("Vocabulario solapado (sinonimia y polisemia). "), t("La confusión Depresión–Ideación Suicida proviene de términos compartidos; el VSM trata «coche» y «auto» como ejes ortogonales (sinonimia) y no distingue acepciones (polisemia), limitaciones señaladas en las propias diapositivas de la materia.") ]));
children.push(bullet([ b("Restricción de memoria. "), t("La matriz TF-IDF densa obliga a limitar el vocabulario; sin ello se agota la RAM. Las representaciones dispersas resolverían el problema sin perder términos.") ]));
children.push(bullet([ b("Calidad de las etiquetas. "), t("Las etiquetas derivan del subreddit de origen, lo que introduce ruido: no toda publicación en r/depression expresa depresión clínica, y el etiquetado por comunidad no equivale a un diagnóstico profesional.") ]));
children.push(H2("8.2 Consideraciones éticas"));
children.push(P("Teferra et al. (2024) subrayan que el uso de PLN en salud mental plantea riesgos de privacidad, sesgo e interpretabilidad. Tres principios guían el alcance responsable de este trabajo. Primero, el sistema es una herramienta de cribado y priorización, no un instrumento de diagnóstico: su salida debe acompañar, nunca sustituir, el juicio de un profesional. Segundo, los datos provienen de foros públicos y anónimos, y no deben emplearse para identificar ni perfilar a individuos; por ello el repositorio del proyecto excluye los textos crudos. Tercero, la priorización de la exhaustividad sobre la precisión es una decisión ética explícita: ante la duda, es preferible revisar de más que omitir un caso de riesgo. La interpretabilidad del método clásico —cada decisión es trazable hasta pesos de términos concretos— constituye además una salvaguarda frente a la opacidad de los modelos neuronales."));
children.push(H2("8.3 Trabajo futuro"));
children.push(bullet([ b("Manejo explícito de la negación, "), t("mediante marcado de alcance (p. ej., prefijar tokens tras una negación), para reducir confusiones de polaridad.") ]));
children.push(bullet([ b("Lexicones y n-gramas selectivos, "), t("para capturar expresiones de finalidad y despedida características de la ideación suicida, integrándolos como rasgos de alto valor en los centroides.") ]));
children.push(bullet([ b("Representación dispersa, "), t("para escalar a todo el vocabulario sin la restricción de memoria, acercándose a la eficiencia de las bibliotecas profesionales.") ]));
children.push(bullet([ b("Otros clasificadores supervisados. "), t("Habiendo establecido el baseline de Naïve Bayes (Sección 7), un paso natural es comparar también con regresión logística o máquinas de soporte vectorial sobre los mismos vectores TF-IDF, e incorporar selección de rasgos por información mutua.") ]));
children.push(bullet([ b("Reducción semántica (LSI/LDA) "), t("para mitigar la sinonimia proyectando los términos a un espacio de tópicos, abordando una de las limitaciones de fondo del VSM.") ]));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 8. CONCLUSIONES ============================
children.push(H1("9. Conclusiones"));
children.push(P("Este trabajo demostró que un sistema construido exclusivamente con técnicas clásicas de Recuperación de Información —TF-IDF implementado desde cero, centroides de clase y similitud coseno— puede clasificar publicaciones de redes sociales en cuatro categorías de salud mental con un F1 macro de 0.79, confirmando la hipótesis principal. La clase Depresión alcanzó un desempeño cercano al de sistemas supervisados de referencia, evidenciando la separabilidad del vocabulario depresivo en el espacio vectorial."));
children.push(P("El valor del proyecto, sin embargo, reside tanto en el resultado como en el análisis de sus errores. Se identificó que los falsos negativos de la clase de mayor riesgo —Ideación Suicida— obedecían a una limitación intrínseca del modelo ante textos cortos, y se diseñó una mejora dirigida —el umbral por clase— que elevó su exhaustividad de 0.72 a 0.79 y recuperó 103 publicaciones de riesgo, asumiendo conscientemente el compromiso precisión–exhaustividad a favor de esta última. Asimismo, la confusión observada entre Depresión e Ideación Suicida coincide con hallazgos recientes obtenidos con modelos neuronales, lo que respalda la validez del comportamiento del sistema."));
children.push(P("En conjunto, el enfoque clásico ofrece una línea base sólida, interpretable y económica, cuyos errores son explicables y, por tanto, abordables —una cualidad especialmente valiosa en un dominio tan sensible como la salud mental, donde la transparencia de las decisiones es un requisito ético, no solo técnico."));
children.push(new Paragraph({ children: [new TextRun("")], pageBreakBefore: true }));

// ============================ 9. REFERENCIAS ============================
children.push(H1("10. Aparato Crítico (Referencias)"));
children.push(ref([ t("Bauer, B., Norel, R., Leow, A., Abi Rached, Z., Wen, B., & Cecchi, G. (2024). Using Large Language Models to Understand Suicidality in a Social Media–Based Taxonomy of Mental Health Disorders: Linguistic Analysis of Reddit Posts. "), it("JMIR Mental Health, "), t("11, e57234. https://doi.org/10.2196/57234") ]));
children.push(ref([ t("Bucur, A.-M., & Dinu, L. P. (2020). Detecting Early Onset of Depression from Social Media Text Using Learned Confidence Scores. "), it("Proceedings of the Seventh Italian Conference on Computational Linguistics (CLiC-it 2020). "), t("https://arxiv.org/abs/2011.01695") ]));
children.push(ref([ t("Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying Mental Health Signals in Twitter. "), it("Proceedings of the Workshop on Computational Linguistics and Clinical Psychology, "), t("51–60. https://doi.org/10.3115/v1/W14-3207") ]));
children.push(ref([ t("De Boom, C., Van Canneyt, S., Bohez, S., Demeester, T., & Dhoedt, B. (2015). Learning Semantic Similarity for Very Short Texts. "), it("IEEE International Conference on Data Mining Workshop (ICDMW), "), t("1229–1234. https://arxiv.org/abs/1512.00765") ]));
children.push(ref([ t("De Choudhury, M., Gamon, M., Counts, S., & Horvitz, E. (2013). Predicting Depression via Social Media. "), it("Proceedings of the International AAAI Conference on Web and Social Media, "), t("7(1), 128–137. https://doi.org/10.1609/icwsm.v7i1.14432") ]));
children.push(ref([ t("Kleene, S. C. (1956). Representation of Events in Nerve Nets and Finite Automata. En C. E. Shannon & J. McCarthy (Eds.), "), it("Automata Studies "), t("(pp. 3–41). Princeton University Press.") ]));
children.push(ref([ t("Manning, C. D., Raghavan, P., & Schütze, H. (2008). "), it("Introduction to Information Retrieval. "), t("Cambridge University Press. (Caps. 6, 8 y 14.) https://nlp.stanford.edu/IR-book/") ]));
children.push(ref([ t("Sánchez, O. (2026). "), it("Análisis y Procesamiento Inteligente de Textos "), t("[Material de clase, Sesiones 2.1–3.1-2]. Facultad de Ingeniería, UNAM.") ]));
children.push(ref([ t("Tadesse, M. M., Lin, H., Xu, B., & Yang, L. (2019). Detection of Depression-Related Posts in Reddit Social Media Forum. "), it("IEEE Access, "), t("7, 44883–44893. https://doi.org/10.1109/ACCESS.2019.2909180") ]));
children.push(ref([ t("Teferra, B. G., Rueda, A., Pang, H., Valenzano, R., Samavi, R., Krishnan, S., & Bhat, V. (2024). Screening for Depression Using Natural Language Processing: Literature Review. "), it("Interactive Journal of Medical Research, "), t("13, e55067. https://doi.org/10.2196/55067") ]));
children.push(ref([ t("Conjuntos de datos empleados (fuentes públicas, Kaggle): "), it("Reddit Dataset: r/Depression and r/SuicideWatch "), t("(https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch); y conjuntos complementarios de publicaciones neutrales, de tendencias suicidas y de detección de esquizofrenia basada en Reddit, unificados y balanceados para este trabajo.") ]));

// ============================ DOCUMENTO ============================
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Calibri", color: "2E5E8C" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Calibri", color: "404040" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ]
  },
  numbering: { config: [ { reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 600, hanging: 280 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 1080, hanging: 280 } } } }
  ] } ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [ new Paragraph({ alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 4 } },
      children: [new TextRun({ text: "Clasificación Multiclase de Trastornos de Salud Mental · APIT 2026-2", size: 16, color: "888888" })] }) ] }) },
    footers: { default: new Footer({ children: [ new Paragraph({ alignment: AlignmentType.CENTER,
      children: [ new TextRun({ text: "Página ", size: 16, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" }) ] }) ] }) },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/proyecto_apit/reports/Reporte_APIT.docx", buffer);
  console.log("DOCX ampliado generado");
});
