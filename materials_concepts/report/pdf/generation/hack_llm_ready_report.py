"""The goal of this script is to convert the LLM ready report .tex file into a
PDF file w/o the scores.

Furthermore, the the content should be distilled to the following sections:
- 3.1 Possible new combinations of your own concepts
      (which, to our knowledge, have not yet been combined)
- 3.2 Possible new combinations of your concepts with concepts that,
      to our knowledge, you have not yet worked on (at least not according
      to the abstract data of the last years)
- 5.  Concepts that have been proposed and that are a little further away from your 
    previous work and a little more "exotic"
"""

from collections import defaultdict
import os
from pathlib import Path
import re

from loguru import logger

latex_header = r"""
\documentclass{article}%
\usepackage[T1]{fontenc}%
\usepackage[utf8]{inputenc}%
\usepackage{lmodern}%
\usepackage{textcomp}%
\usepackage{lastpage}%
\usepackage{graphicx}%
\usepackage{float}%
\usepackage{graphicx}%
%
\title{Research Keywords Analysis}%
\author{Prediction Model}%
\date{\today}%
%
\begin{document}%
\normalsize%
\maketitle%
"""

latex_footer = r"""
\end{document}%
"""


def extract_sections(lines: list[str]) -> defaultdict[str, list[str]]:
    section_name = "<none>"
    sections = defaultdict(list)
    section_regex = r"\\section{(.+?)}"

    for line in lines:
        if re.match(section_regex, line):
            section_name = re.findall(section_regex, line)[0]
            logger.debug(f"Found section: {section_name}")

        sections[section_name].append(line)

    return sections


def write_lines_to_file(lines: list[str], output_file: Path):
    # at the end, write the content to the output file
    content = latex_header + "%\n" * 2 + "\n".join(lines)
    if latex_footer not in content:
        # last section contains the footer, but we might want to exclude it
        content += latex_footer

    output_file.write_text(content)


def total_replace_regex(lines: list[str], regex: str) -> list[str]:
    return [re.sub(regex, "", line) for line in lines]


def process_tex_file(input_file: Path, output_folder: Path):
    lines = input_file.read_text().split("\n")
    sections = extract_sections(lines)

    plain_scores = r": \d\.\d+%"
    scores_and_distances = r": \d\.\d+ \(distance: \d\.\d+\)%"

    # Section 3
    logger.info("Processing section 3")
    lines_plain_sugs = sections["Suggestions of Research Directions"]
    lines_plain_sugs = total_replace_regex(lines_plain_sugs, plain_scores)
    lines_plain_sugs = total_replace_regex(lines_plain_sugs, r" \(Best 25\)")
    plain_output_file = output_folder / "plain_suggestions.tex"
    write_lines_to_file(lines_plain_sugs, plain_output_file)
    logger.info(f"Creating PDF for section 3: {plain_output_file}")
    os.system(f"pdflatex -output-directory={output_folder} {plain_output_file}")

    # Section 5
    logger.info("Processing section 5")
    lines_exotic_sugs = sections[
        "Top 50 Concepts that have other concept with degree <= 100 and 3 <= distance <= 6"
    ]
    lines_exotic_sugs = total_replace_regex(lines_exotic_sugs, scores_and_distances)
    lines_exotic_sugs = total_replace_regex(lines_exotic_sugs, "Top 50 ")
    exotic_output_file = output_folder / "exotic_suggestions.tex"
    write_lines_to_file(lines_exotic_sugs, exotic_output_file)
    logger.info(f"Creating PDF for section 5: {plain_output_file}")
    os.system(f"pdflatex -output-directory={output_folder} {exotic_output_file}")


base_path = Path("materials_concepts/report/pdf/generation")
# list all folders
dirs = [path for path in base_path.iterdir() if path.is_dir()]

for dir in dirs:
    # find .tex in dir
    tex_files = list(dir.glob("*.tex"))
    if len(tex_files) == 0:
        logger.warning(f"No .tex files found in {dir}")
        continue

    assert len(tex_files) == 1, f"Multiple .tex files found in {dir}"

    tex_file = tex_files[0]
    logger.info(f"Found {tex_file}")

    output_folder = dir / "distilled"
    output_folder.mkdir(exist_ok=True)

    process_tex_file(tex_file, output_folder)
