// Markdown → .docx conversion for the session download button. Parses the same
// Markdown string the .md download produces (remark, GFM) and walks the mdast
// tree into docx paragraphs/tables, so both formats always carry identical
// content. Loaded via dynamic import() so the docx lib stays out of the main bundle.
import {
  BorderStyle,
  Document,
  ExternalHyperlink,
  HeadingLevel,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from 'docx';
import type {
  BlockContent,
  DefinitionContent,
  List,
  PhrasingContent,
  Root,
  RootContent,
  Table as MdTable,
} from 'mdast';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import { unified } from 'unified';

type DocxBlock = Paragraph | Table;

interface InlineStyle {
  bold?: boolean;
  italics?: boolean;
  strike?: boolean;
  code?: boolean;
  hyperlink?: boolean;
}

const HEADING_LEVELS = [
  HeadingLevel.HEADING_1,
  HeadingLevel.HEADING_2,
  HeadingLevel.HEADING_3,
  HeadingLevel.HEADING_4,
  HeadingLevel.HEADING_5,
  HeadingLevel.HEADING_6,
] as const;

const MONO_FONT = 'Courier New';
const INDENT_PER_LEVEL = 360; // twips (0.25")

function textRun(text: string, style: InlineStyle): TextRun {
  return new TextRun({
    text,
    bold: style.bold,
    italics: style.italics,
    strike: style.strike,
    font: style.code ? MONO_FONT : undefined,
    style: style.hyperlink ? 'Hyperlink' : undefined,
  });
}

function inlineRuns(
  nodes: PhrasingContent[],
  style: InlineStyle = {},
): (TextRun | ExternalHyperlink)[] {
  const runs: (TextRun | ExternalHyperlink)[] = [];
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        runs.push(textRun(node.value, style));
        break;
      case 'strong':
        runs.push(...inlineRuns(node.children, { ...style, bold: true }));
        break;
      case 'emphasis':
        runs.push(...inlineRuns(node.children, { ...style, italics: true }));
        break;
      case 'delete':
        runs.push(...inlineRuns(node.children, { ...style, strike: true }));
        break;
      case 'inlineCode':
        runs.push(textRun(node.value, { ...style, code: true }));
        break;
      case 'break':
        runs.push(new TextRun({ break: 1 }));
        break;
      case 'link': {
        const children = inlineRuns(node.children, { ...style, hyperlink: true })
          // ExternalHyperlink children must be plain runs — flatten nested links
          .filter((r): r is TextRun => r instanceof TextRun);
        runs.push(new ExternalHyperlink({ link: node.url, children }));
        break;
      }
      case 'image':
        if (node.alt) runs.push(textRun(node.alt, { ...style, italics: true }));
        break;
      default:
        // html / footnotes etc. — render any nested text, otherwise skip
        if ('value' in node && typeof node.value === 'string') {
          runs.push(textRun(node.value, style));
        }
    }
  }
  return runs;
}

function tableToDocx(node: MdTable): Table {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: node.children.map((row, rowIdx) =>
      new TableRow({
        children: row.children.map(cell =>
          new TableCell({
            shading: rowIdx === 0
              ? { type: ShadingType.CLEAR, fill: 'F3F4F6' }
              : undefined,
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [
              new Paragraph({
                children: inlineRuns(cell.children, rowIdx === 0 ? { bold: true } : {}),
              }),
            ],
          }),
        ),
      }),
    ),
  });
}

function listToDocx(node: List, out: DocxBlock[], level: number): void {
  let index = node.start ?? 1;
  for (const item of node.children) {
    const marker = node.ordered ? `${index}. ` : '• ';
    index += 1;
    let firstParagraphDone = false;
    for (const child of item.children) {
      if (child.type === 'paragraph') {
        out.push(new Paragraph({
          children: [
            // Manual markers instead of docx numbering: predictable restarts,
            // no shared-numbering-instance bleed between separate lists
            new TextRun({ text: firstParagraphDone ? '' : marker }),
            ...inlineRuns(child.children),
          ],
          indent: { left: INDENT_PER_LEVEL * (level + 1), hanging: firstParagraphDone ? 0 : INDENT_PER_LEVEL },
          spacing: { after: 60 },
        }));
        firstParagraphDone = true;
      } else if (child.type === 'list') {
        listToDocx(child, out, level + 1);
      } else {
        blockToDocx(child, out, level + 1);
      }
    }
  }
}

function blockToDocx(
  node: RootContent | BlockContent | DefinitionContent,
  out: DocxBlock[],
  level = 0,
): void {
  switch (node.type) {
    case 'heading':
      out.push(new Paragraph({
        heading: HEADING_LEVELS[Math.min(node.depth, 6) - 1],
        children: inlineRuns(node.children),
        spacing: { before: 240, after: 120 },
      }));
      break;
    case 'paragraph':
      out.push(new Paragraph({
        children: inlineRuns(node.children),
        indent: level > 0 ? { left: INDENT_PER_LEVEL * level } : undefined,
        spacing: { after: 120 },
      }));
      break;
    case 'list':
      listToDocx(node, out, level);
      break;
    case 'blockquote':
      for (const child of node.children) {
        if (child.type === 'paragraph') {
          out.push(new Paragraph({
            children: inlineRuns(child.children, { italics: true }),
            indent: { left: INDENT_PER_LEVEL * (level + 1) },
            border: { left: { style: BorderStyle.SINGLE, size: 12, color: 'D1D5DB' } },
            spacing: { after: 120 },
          }));
        } else {
          blockToDocx(child, out, level + 1);
        }
      }
      break;
    case 'code': {
      const lines = node.value.split('\n');
      out.push(new Paragraph({
        children: lines.flatMap((line, i) => [
          ...(i > 0 ? [new TextRun({ break: 1 })] : []),
          new TextRun({ text: line, font: MONO_FONT, size: 18 }),
        ]),
        shading: { type: ShadingType.CLEAR, fill: 'F3F4F6' },
        spacing: { after: 120 },
      }));
      break;
    }
    case 'thematicBreak':
      out.push(new Paragraph({
        children: [],
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'D1D5DB' } },
        spacing: { after: 120 },
      }));
      break;
    case 'table':
      out.push(tableToDocx(node));
      out.push(new Paragraph({ children: [], spacing: { after: 60 } }));
      break;
    default:
      break; // html, definitions, etc. — not produced by our session markdown
  }
}

/** Convert a Markdown document to a .docx Blob ready for download. */
export async function markdownToDocxBlob(markdown: string): Promise<Blob> {
  const tree = unified().use(remarkParse).use(remarkGfm).parse(markdown) as Root;
  const children: DocxBlock[] = [];
  for (const node of tree.children) blockToDocx(node, children);

  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: 'Calibri', size: 22 } }, // 11pt body
      },
    },
    sections: [{ children }],
  });
  return Packer.toBlob(doc);
}
