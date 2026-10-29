#!/usr/bin/env bash
# Build the report from report.md: an editable Word file and a print-ready PDF.
# Needs pandoc; the PDF is printed by headless Google Chrome (no LaTeX required).
set -euo pipefail
cd "$(dirname "$0")"

pandoc report.md -o report.docx --resource-path=.:..
pandoc report.md -o report.html --standalone --embed-resources --resource-path=.:.. --css style.css \
  --metadata lang=en

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/report.pdf" \
  "file://$PWD/report.html" 2>/dev/null
rm report.html
echo "wrote report/report.docx and report/report.pdf"
