const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, ImageRun, PageNumber, Header, Footer
} = require("docx");

const FIG = "/home/claude/proyecto_apit/reports/figuras/";
const ACCENT = "1F4E79";

// ---------- helpers ----------
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
function P(runs, opts = {}) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({
    children, alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after === undefined ? 160 : opts.after, line: 276 }, ...opts.extra
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bul", level },
    children: Array.isArray(text) ? text : [new TextRun(text)],
    spacing: { after: 60, line: 264 }, alignment: AlignmentType.JUSTIFIED
  });
}
const b = (t) => new TextRun({ text: t, bold: true });
const t = (txt) => new TextRun({ text: txt });
const it = (txt) => new TextRun({ text: txt, italics: true });
const code = (txt) => new TextRun({ text: txt, font: "Consolas", size: 20 });
function caption(txt) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: txt, italics: true, size: 18, color: "555555" })] });
}
function imgRatio(file, widthPx, ratio, cap) {
  const data = fs.readFileSync(FIG + file);
  const parts = [ new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * ratio) } })] }) ];
  if (cap) parts.push(caption(cap));
  return parts;
}
function makeTable(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerRow = new TableRow({ tableHeader: true, children: headers.map((htext, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: ACCENT, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: htext, bold: true, color: "FFFFFF", size: 20 })] })] })) });
  const bodyRows = rows.map((r, ri) => new TableRow({ children: r.map((cell, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: ri % 2 ? "F2F7FB" : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20, bold: (i === 0) })] })] })) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}
// referencia con sangría francesa
function ref(runs) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 },
    indent: { left: 360, hanging: 360 }, children: runs });
}

module.exports = { H1, H2, H3, P, bullet, b, t, it, code, caption, imgRatio, makeTable, ref,
  Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  Header, Footer, PageNumber, ACCENT };
