"""Render the finished Markdown research report to a readable, linked PDF."""

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)


def build(source, output):
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    font_root = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [
        ("Research", "DejaVuSans.ttf"),
        ("Research-Bold", "DejaVuSans-Bold.ttf"),
        ("Research-Italic", "DejaVuSans-Oblique.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(font_root / file)))
    pdfmetrics.registerFontFamily(
        "Research",
        normal="Research",
        bold="Research-Bold",
        italic="Research-Italic",
        boldItalic="Research-Bold",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName="Research",
            fontSize=9.4,
            leading=14.1,
            spaceAfter=7,
            textColor=colors.HexColor("#243442"),
            alignment=TA_LEFT,
            allowWidows=0,
            allowOrphans=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleReport",
            parent=styles["Body"],
            fontName="Research-Bold",
            fontSize=23,
            leading=28,
            spaceAfter=14,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionReport",
            parent=styles["Body"],
            fontName="Research-Bold",
            fontSize=13,
            leading=18,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor("#087e8b"),
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubsectionReport",
            parent=styles["Body"],
            fontName="Research-Bold",
            fontSize=10.4,
            leading=15,
            spaceBefore=10,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionReport",
            parent=styles["Body"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#51636e"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CellReport", parent=styles["Body"], fontSize=7.5, leading=10.2, spaceAfter=0
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeReport",
            fontName="Courier",
            fontSize=7.2,
            leading=10,
            backColor=colors.HexColor("#f1f5f6"),
            borderPadding=7,
            spaceAfter=9,
        )
    )
    width = A4[0] - 84

    def inline(text):
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)

        def link(match):
            label, url = match.groups()
            if url.startswith("https://"):
                return f'<link href="{url}" color="#087e8b">{label}</link>'
            return label

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)

    story = []
    lines = source.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(html.escape(lines[i]))
                i += 1
            story.append(XPreformatted("\n".join(code), styles["CodeReport"]))
            i += 1
            continue
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\-\s]+", cell) for cell in row):
                    table.append(row)
                i += 1
            count = len(table[0])
            weights = [1 / count] * count
            if count >= 5:
                weights = [0.34] + [0.66 / (count - 1)] * (count - 1)
            elif count == 3:
                weights = [0.48, 0.26, 0.26]
            elif count == 4:
                weights = [0.43, 0.19, 0.19, 0.19]
            cells = [
                [Paragraph(inline(cell), styles["CellReport"]) for cell in row] for row in table
            ]
            t = Table(cells, colWidths=[width * w for w in weights], repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceff0")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f5f7f8")],
                        ),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#087e8b")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([t, Spacer(1, 10)])
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            path = source.parent / image_match.group(2)
            img = Image(str(path))
            img.drawHeight *= width / img.drawWidth
            img.drawWidth = width
            items = [img, Spacer(1, 5)]
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("Figure "):
                text = []
                while j < len(lines) and lines[j].strip():
                    text.append(lines[j].strip())
                    j += 1
                items.append(Paragraph(inline(" ".join(text)), styles["CaptionReport"]))
                i = j
            else:
                i += 1
            story.append(KeepTogether(items))
            continue
        level = len(line) - len(line.lstrip("#"))
        if level:
            style_name = (
                "TitleReport"
                if level == 1
                else "SectionReport"
                if level == 2
                else "SubsectionReport"
            )
            story.append(Paragraph(inline(line.lstrip("#").strip()), styles[style_name]))
            i += 1
            continue
        if re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + inline(line[2:]), styles["Body"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while (
            i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|```|[-*] )", lines[i])
        ):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["Body"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d8e4e7"))
        canvas.line(42, 37, A4[0] - 42, 37)
        canvas.setFont("Research", 7)
        canvas.setFillColor(colors.HexColor("#61727e"))
        canvas.drawString(42, 25, "QCSE research | Hamiltonian encoders | 16 September 2026")
        canvas.drawRightString(A4[0] - 42, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=40,
        bottomMargin=49,
        title="Hamiltonian encoders for causal quantum semantics",
        author="QCSE research study",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "REPORT.md")
    parser.add_argument(
        "--output", type=Path, default=Path("output/pdf/hamiltonian_encoder_research.pdf")
    )
    args = parser.parse_args()
    build(args.source, args.output)
