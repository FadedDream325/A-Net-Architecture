"""Render the generated DOCX summary to PDF with fpdf2 (no Word/Edge)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from fpdf import FPDF
from fpdf.fonts import FontFace

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "A-Net_Experiment_Summary.docx"
PDF = ROOT / "A-Net_Experiment_Summary.pdf"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
FONT = r"C:\Windows\Fonts\simhei.ttf"


def clean(s: str) -> str:
    return s.replace("•", "-").replace("µ", "u").replace("−", "-")


def text_of(node) -> str:
    return clean("".join(t.text or "" for t in node.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")))


def blocks():
    with zipfile.ZipFile(DOCX) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for child in root.find("w:body", NS):
        tag = child.tag.split("}")[-1]
        if tag == "p":
            style = child.find("w:pPr/w:pStyle", NS)
            style_val = style.get("{%s}val" % NS["w"]) if style is not None else ""
            if any(br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page"
                   for br in child.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br")):
                out.append(("pagebreak", ""))
                continue
            out.append((style_val or "Normal", text_of(child)))
        elif tag == "tbl":
            rows = []
            for tr in child.findall("w:tr", NS):
                rows.append([text_of(tc) for tc in tr.findall("w:tc", NS)])
            out.append(("table", rows))
    return out


class Pdf(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("CJK", "", 7.5)
        self.set_text_color(120, 120, 120)
        self.set_y(8)
        self.cell(0, 4, "A-Net 实验总结 · 结果、优势与代价", align="R")
        self.set_text_color(0, 0, 0)
        self.set_y(15)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-10)
        self.set_font("CJK", "", 7.5)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def main() -> None:
    pdf = Pdf(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("CJK", "", FONT)
    pdf.add_page()
    body: list[tuple[str, object]] = blocks()

    # Title page: collect the first paragraphs until the first page break.
    i = 0
    title_done = False
    while i < len(body):
        style, value = body[i]
        if style == "pagebreak":
            i += 1
            title_done = True
            break
        if style == "Title":
            pdf.set_font("CJK", "", 24)
            pdf.set_text_color(31, 56, 100)
            pdf.ln(52)
            pdf.set_x(15)
            pdf.multi_cell(0, 12, str(value), align="C")
        elif style == "Subtitle":
            pdf.set_font("CJK", "", 10.5)
            pdf.set_text_color(89, 89, 89)
            pdf.set_x(15)
            pdf.multi_cell(0, 6, str(value), align="C")
        i += 1
    pdf.set_text_color(0, 0, 0)
    if not title_done:
        pdf.add_page()

    # Remaining content.
    while i < len(body):
        style, value = body[i]
        i += 1
        if style == "pagebreak":
            pdf.add_page()
            continue
        if style == "Title":
            pdf.set_font("CJK", "", 24)
            pdf.set_text_color(31, 56, 100)
            pdf.set_x(15)
            pdf.multi_cell(0, 12, str(value), align="C")
            continue
        if style == "Subtitle":
            pdf.set_font("CJK", "", 10.5)
            pdf.set_text_color(89, 89, 89)
            pdf.set_x(15)
            pdf.multi_cell(0, 6, str(value), align="C")
            continue
        if style == "Heading1":
            if pdf.get_y() > 245:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font("CJK", "", 15)
            pdf.set_text_color(31, 78, 121)
            pdf.set_x(15)
            pdf.multi_cell(0, 8, str(value))
            pdf.set_draw_color(217, 226, 243)
            pdf.line(15, pdf.get_y() + 1, 195, pdf.get_y() + 1)
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
            continue
        if style == "Heading2":
            if pdf.get_y() > 255:
                pdf.add_page()
            pdf.ln(2)
            pdf.set_font("CJK", "", 12)
            pdf.set_text_color(46, 116, 181)
            pdf.set_x(15)
            pdf.multi_cell(0, 7, str(value))
            pdf.set_text_color(0, 0, 0)
            continue
        if style == "table":
            rows = value
            if not rows:
                continue
            # Normalize column widths by rough text length, with bounds.
            ncol = max(len(r) for r in rows)
            weights = []
            for c in range(ncol):
                maxlen = max((len(r[c]) if c < len(r) else 0) for r in rows)
                weights.append(max(12, min(maxlen, 70)))
            total = sum(weights)
            widths = [180 * w / total for w in weights]
            pdf.set_font("CJK", "", 7.5)
            with pdf.table(
                col_widths=widths,
                width=180,
                text_align="LEFT",
                first_row_as_headings=True,
                headings_style=FontFace(emphasis="", fill_color=(217, 226, 243)),
                line_height=3.8,
                padding=1.2,
                repeat_headings=1,
            ) as table:
                for row in rows:
                    tr = table.row()
                    for c in range(ncol):
                        tr.cell(row[c] if c < len(row) else "")
            pdf.ln(2)
            continue
        # Normal / bullet paragraphs.
        text = str(value)
        if not text.strip():
            pdf.ln(1.5)
            continue
        pdf.set_font("CJK", "", 10.2)
        if text.startswith("- "):
            pdf.set_x(19)
            pdf.multi_cell(176, 5.2, text, align="L")
        else:
            pdf.set_x(15)
            pdf.multi_cell(0, 5.2, text, align="L")

    pdf.output(str(PDF))
    print(PDF, PDF.stat().st_size)


if __name__ == "__main__":
    main()
