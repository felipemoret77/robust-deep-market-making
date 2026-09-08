# Makefile for the robust-deep-market-making repository.
# Sets PYTHONPATH so the flat-import source modules in src/{sim,agents,glft,eval}
# resolve one another (see README "Important: flat imports need PYTHONPATH").

export PYTHONPATH := $(CURDIR)/src/sim:$(CURDIR)/src/agents:$(CURDIR)/src/glft:$(CURDIR)/src/eval:$(PYTHONPATH)
PY := python

.PHONY: help env train-a train-b train-c paper figures clean

help:
	@echo "Targets:"
	@echo "  make env       - print the PYTHONPATH export for interactive use"
	@echo "  make train-a   - train Algorithm A (stationary DQN)"
	@echo "  make train-b   - train Algorithm B (regime-aware fine-tuning)"
	@echo "  make train-c   - train Algorithm C (scenario-bandit fine-tuning)"
	@echo "  make paper     - compile paper/main.tex -> paper/main.pdf (needs tectonic)"
	@echo "  make figures   - regenerate the data-driven figures"
	@echo "  make clean     - remove LaTeX build artifacts and __pycache__"

env:
	@echo 'export PYTHONPATH="$(CURDIR)/src/sim:$(CURDIR)/src/agents:$(CURDIR)/src/glft:$(CURDIR)/src/eval"'

train-a:
	$(PY) src/agents/DeepSarsaQRunner.py

train-b:
	$(PY) src/agents/DeepSarsaQRunner_REGIME.py

train-c:
	$(PY) src/agents/DeepSarsaQRunner_RANDOM_TAU.py

paper:
	cd paper && tectonic main.tex

# Regenerates the self-contained analytical figure bundle. The remaining
# figures are produced by the scripts in src/eval/ and repro/ from the
# committed checkpoints and cached simulation outputs.
figures:
	$(PY) src/eval/generate_as_miss_bundle.py

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.pdf
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
