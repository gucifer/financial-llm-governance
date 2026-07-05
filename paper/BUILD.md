# Building the Preprint PDF

Source: [`preprint.tex`](preprint.tex) — no `.bib`. Seven figures live in [`figures/`](figures/) and are referenced via `\graphicspath{{figures/}}`. Compiles with any modern LaTeX.

**Upload both the `.tex` AND the `figures/` folder** (7 PNGs) — they are a package. On Overleaf, drag the whole `paper/` folder in, or zip `preprint.tex` + `figures/` together.

No LaTeX toolchain is installed on this machine. Pick one path:

## Option A — Overleaf (zero install, recommended)
1. Go to https://www.overleaf.com → New Project → Upload Project (or Blank Project).
2. Upload `preprint.tex`.
3. Set compiler to **pdfLaTeX** (Menu → Compiler). Run twice (for `\maketitle` + hyperref refs).
4. Download PDF.

## Option B — Tectonic (single binary, local)
```bash
pip install tectonic        # or: cargo install tectonic
tectonic preprint.tex       # produces preprint.pdf
```

## Option C — Full TeX Live (Windows)
```bash
# Install MiKTeX (https://miktex.org) or TeX Live, then:
pdflatex preprint.tex
pdflatex preprint.tex       # second pass for cross-refs
```

## arXiv submission checklist (cs.LG)
- [ ] PDF compiles clean, 2 passes, no missing refs.
- [ ] Upload **source** (`preprint.tex` + the 7 PNGs in `figures/`) to arXiv, not the PDF — arXiv recompiles.
- [ ] Primary category: `cs.LG`; cross-list: `cs.CR`, `q-fin.RM` (optional).
- [ ] License: choose CC BY 4.0 or arXiv non-exclusive.
- [ ] Author: Arpan Parikh, sole author (first-author claim preserved).
- [ ] Title/abstract match the `.tex`.
- [ ] Companion repo public + commit pinned before submission: https://github.com/gucifer/financial-llm-governance

## Provenance
Every numeric value in the tables traces to `aml-detection/results/full_results.json`
(verified Stage 4.5 final-integrity pass, 2026-06-25). No fabricated statistics.
