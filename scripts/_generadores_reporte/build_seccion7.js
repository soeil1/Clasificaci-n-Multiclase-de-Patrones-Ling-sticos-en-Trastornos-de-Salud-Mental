const fs = require("fs");
const U = require("./report_utils.js");
const {
  H1, H2, H3, P, bullet, b, t, it, code, caption, imgRatio, makeTable,
  Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat,
  BorderStyle, Header, Footer, PageNumber, ACCENT
} = U;

const children = [];

children.push(new Paragraph({ spacing: { after: 80 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "FRAGMENTO NUEVO PARA INTEGRAR AL REPORTE", bold: true, size: 20, color: "888888" })] }));
children.push(new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Sección 7 — añadida tras la retroalimentación del profesor (baseline Naïve Bayes y validación cruzada). Insertar entre el actual Análisis de Resultados y la Discusión, renumerando las secciones siguientes.", italics: true, size: 18, color: "888888" })] }));

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

children.push(new Paragraph({ spacing: { before: 280, after: 80 },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 6 } },
  children: [new TextRun({ text: "Notas de integración", bold: true, size: 22, color: ACCENT })] }));
children.push(bullet([ t("Insertar esta Sección 7 entre el actual «Análisis de Resultados» y la «Discusión».") ]));
children.push(bullet([ t("Renumerar las secciones siguientes: Discusión → 8, Conclusiones → 9, Referencias → 10.") ]));
children.push(bullet([ t("En el Abstract (Métodos) y en la «Descripción de la estructura» de la Introducción, mencionar la validación cruzada y el baseline. (Ya integrado en el reporte completo.)") ]));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 160 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Calibri", color: "2E5E8C" },
        paragraph: { spacing: { before: 200, after: 100 } } },
    ]
  },
  numbering: { config: [ { reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 600, hanging: 280 } } } }
  ] } ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/proyecto_apit/reports/Seccion7_NUEVA.docx", buffer);
  console.log("Word solo-nuevo generado");
});
