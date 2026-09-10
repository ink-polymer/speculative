PAPER=main

.PHONY: all figures clean

all: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

figures:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/adaptivetree-tbv-mpl XDG_CACHE_HOME=/tmp/adaptivetree-tbv-cache python3 figures/make_figures.py

clean:
	latexmk -C $(PAPER).tex
