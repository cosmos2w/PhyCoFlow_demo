"""Normalize the publication SI source without wrapping prose paragraphs."""

from __future__ import annotations

import re
from pathlib import Path


SOURCE = Path(__file__).with_name("CURRENT_Supplementary Info_FINAL.tex")
PRESERVED_ENVIRONMENTS = {"center", "longtable", "tabular", "tabularx", "equation", "equation*", "align", "align*"}
STRUCTURAL = re.compile(
    r"^\\(?:documentclass|usepackage|setlength|renewcommand|newcommand|newcolumntype|captionsetup|raggedbottom|clubpenalty|widowpenalty|setcounter|begin|end|section|section\*|subsection|subsection\*|sisubsection|label|FloatBarrier|Needspace|clearpage|vspace|begingroup|endgroup|pdfbookmark|tableofcontents|phantomsection|addcontentsline|centering|suppgraphic|toprule|midrule|bottomrule|endfirsthead|endhead|endfoot|endlastfoot|multicolumn|small|footnotesize|\[|\])"
)


def brace_delta(line: str) -> int:
    clean = re.sub(r"\\[{}]", "", line)
    return clean.count("{") - clean.count("}")


def collapse_balanced_commands(lines: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("\\caption{"):
            parts = [line.strip()]
            balance = brace_delta(line)
            index += 1
            while balance > 0 and index < len(lines):
                parts.append(lines[index].strip())
                balance += brace_delta(lines[index])
                index += 1
            result.append(" ".join(part for part in parts if part))
            continue
        result.append(line.rstrip())
        index += 1
    return result


def format_source(text: str) -> str:
    lines = [line for line in text.splitlines() if not re.match(r"^\s*%", line)]
    lines = collapse_balanced_commands(lines)
    output: list[str] = []
    paragraph: list[str] = []
    environment_stack: list[str] = []
    in_bibliography = False

    def flush() -> None:
        if paragraph:
            output.append(" ".join(part.strip() for part in paragraph if part.strip()))
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        begin = re.match(r"^\\begin\{([^}]+)\}", stripped)
        end = re.match(r"^\\end\{([^}]+)\}", stripped)

        if begin and begin.group(1) == "thebibliography":
            flush()
            output.append(stripped)
            in_bibliography = True
            continue
        if end and end.group(1) == "thebibliography":
            flush()
            output.append(stripped)
            in_bibliography = False
            continue
        if in_bibliography:
            if stripped.startswith("\\bibitem"):
                flush()
                paragraph.append(stripped)
            elif not stripped:
                flush()
                output.append("")
            else:
                paragraph.append(stripped)
            continue

        if environment_stack:
            output.append(line.rstrip())
            if end and end.group(1) == environment_stack[-1]:
                environment_stack.pop()
            elif begin and begin.group(1) in PRESERVED_ENVIRONMENTS:
                environment_stack.append(begin.group(1))
            continue

        if begin and begin.group(1) in PRESERVED_ENVIRONMENTS:
            flush()
            output.append(stripped)
            environment_stack.append(begin.group(1))
            continue

        if not stripped:
            flush()
            if output and output[-1] != "":
                output.append("")
            continue
        if stripped in {"{", "}", "{\\small", "{\\footnotesize"} or STRUCTURAL.match(stripped):
            flush()
            output.append(stripped)
            continue
        if stripped.startswith("\\caption{"):
            flush()
            output.append(stripped)
            continue
        paragraph.append(stripped)

    flush()
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output) + "\n"


if __name__ == "__main__":
    SOURCE.write_text(format_source(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
