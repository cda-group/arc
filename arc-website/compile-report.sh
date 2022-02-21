#!/bin/bash

latexmk ../arc-report/main.tex -xelatex -outdir=../arc-report/target -cd -silent -bibtex -pdf > /dev/null
mv ../arc-report/target/main.pdf src/Arc-Report.pdf
