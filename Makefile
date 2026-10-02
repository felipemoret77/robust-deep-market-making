# Run from the repository root. PY may select an existing scientific environment.
export PYTHONPATH := $(CURDIR)/src/sim:$(CURDIR)/src/agents:$(CURDIR)/src/glft:$(CURDIR)/src/eval:$(PYTHONPATH)
PY ?= python
ROOT := $(CURDIR)
FIGURE_OUT ?= $(ROOT)/outputs/paper_figures

.PHONY: help env verify-paper train-a train-b train-c paper figures clean
help:
	@echo "make verify-paper  - verify canonical checkpoints, metadata and archived data"
	@echo "make figures       - redraw six figures from shipped data into outputs/paper_figures"
	@echo "make paper         - compile paper/main.tex (requires tectonic)"
	@echo "make train-a/b/c   - train in outputs/training/A, B or C; preserve shipped checkpoints"
	@echo "Frozen evaluations: python repro/evaluate_paper.py --help"
env:
	@echo 'export PYTHONPATH="$(ROOT)/src/sim:$(ROOT)/src/agents:$(ROOT)/src/glft:$(ROOT)/src/eval"'
verify-paper:
	$(PY) repro/verify_paper.py
train-a:
	mkdir -p outputs/training/A
	cd outputs/training/A && $(PY) $(ROOT)/src/agents/DeepSarsaQRunner.py
train-b:
	mkdir -p outputs/training/B
	cd outputs/training/B && MM_LOB_SIM_VIZ_OUTPUT_DIR=$(ROOT)/outputs/training/B/viz $(PY) $(ROOT)/src/agents/DeepSarsaQRunner_REGIME.py
train-c:
	mkdir -p outputs/training/C
	cd outputs/training/C && MM_LOB_SIM_VIZ_OUTPUT_DIR=$(ROOT)/outputs/training/C/viz $(PY) $(ROOT)/src/agents/DeepSarsaQRunner_RANDOM_TAU.py
paper:
	cd paper && tectonic main.tex
figures:
	$(PY) repro/plot_paper_sweep.py --output "$(FIGURE_OUT)"
	$(PY) repro/regen_regime_mc_figs.py --output "$(FIGURE_OUT)"
	$(PY) repro/regen_scenario_bandit_shaded.py --output "$(FIGURE_OUT)"
clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
