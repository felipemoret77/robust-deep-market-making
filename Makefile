# Run from the repository root. PY may select an existing scientific environment.
export PYTHONPATH := $(CURDIR)/src/sim:$(CURDIR)/src/agents:$(CURDIR)/src/glft:$(PYTHONPATH)
PY ?= python
ROOT := $(CURDIR)
NUMERICS_OUT ?= $(ROOT)/outputs/numerics_$(shell date +%Y%m%d_%H%M%S)
FIGURE_OUT ?= $(ROOT)/outputs/paper_figures

.PHONY: help env check-numerics verify-paper verify-uniform-control check-portability glft-calibration-check train-a train-b train-c paper figures clean
help:
	@echo "make check-numerics - test frozen C51 action stability and inventory masks"
	@echo "make verify-paper  - verify canonical checkpoints, metadata and archived data"
	@echo "make verify-uniform-control - verify the matched adaptive/uniform experiment"
	@echo "make figures       - regenerate supported figures into outputs/paper_figures"
	@echo "make check-portability - verify an isolated export with Desktop access blocked"
	@echo "make glft-calibration-check - aggregate archived calibration statistics and run a small simulation check"
	@echo "make paper         - compile paper/main.tex (requires tectonic)"
	@echo "make train-a/b/c   - train in outputs/training/A, B or C; preserve shipped checkpoints"
	@echo "Frozen evaluations: python repro/evaluate_corrected.py --help"
env:
	@echo 'export PYTHONPATH="$(ROOT)/src/sim:$(ROOT)/src/agents:$(ROOT)/src/glft"'
check-numerics:
	$(PY) repro/verify_numerical_stability.py --output "$(NUMERICS_OUT)"
verify-paper:
	$(PY) repro/verify_paper.py
verify-uniform-control:
	$(PY) repro/uniform_control.py verify
check-portability:
	$(PY) repro/check_portability.py
glft-calibration-check:
	$(PY) repro/glft_calibration.py --mode aggregate --output outputs/glft_aggregate
	$(PY) repro/glft_calibration.py --output outputs/glft_smoke
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
	$(PY) repro/plot_paper_heatmaps.py --output "$(FIGURE_OUT)" --verify-reference
	$(PY) repro/glft_figures.py --output "$(FIGURE_OUT)"
	$(PY) repro/frontier.py plot --output "$(FIGURE_OUT)"
	$(PY) repro/plot_policy_results.py --output "$(FIGURE_OUT)"
	$(PY) repro/plot_paper_episode.py --output "$(FIGURE_OUT)"
	$(PY) repro/plot_paper_diagrams.py --output "$(FIGURE_OUT)"
clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
