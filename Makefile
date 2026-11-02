# Project tasks (`make help` lists them). `play` needs a ROM you supply yourself (docs/ROM.md).
PY   := .venv/bin/python
NPROC := $(shell nproc)
RUN  := OMP_NUM_THREADS=$(NPROC) $(PY)


.PHONY: train-mario
.PHONY: verify-results verify-data figures calibration study-s study-s-reading demo-small nulls
.PHONY: help setup download download-all build-graphs build-graphs-all import-rom play ablation follow runs checkpoint stop check-env check-connectome test clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

setup:            ## create the Python 3.11 venv and install dependencies
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

download:         ## fetch the primary dataset named in the registry (CC-BY)
	$(RUN) -m flymario.connectome.download

download-all:     ## fetch both: the primary and the replication axis
	$(RUN) -c "from flymario.connectome import download; download.all_datasets()"

build-graphs:     ## raw tables -> validated ConnectomeGraph caches
	$(RUN) scripts/build_graph.py

build-graphs-all: ## build both datasets (primary + replication axis)
	$(RUN) scripts/build_graph.py --all

import-rom:       ## import your own SMB ROM from roms/ and link it to our integration
	$(RUN) scripts/import_rom.py

play:             ## run the fly in Mario with the live dashboard
	$(RUN) scripts/play.py

ablation:         ## topology comparison with matched firing rates
	$(RUN) scripts/ablation.py

follow:           ## live view of the most recent long run (make follow RUN=name)
	$(PY) tools/follow.py $(RUN_NAME)

runs:             ## every live run and its state (running / stopped / DEAD)
	$(PY) tools/follow.py --list

checkpoint:       ## ask the most recent run to save now and keep going
	$(PY) tools/runctl.py checkpoint $(RUN_NAME)

stop:             ## ask the most recent run to save and exit cleanly (resumable)
	$(PY) tools/runctl.py stop $(RUN_NAME)

check-env:        ## verify emulator, ROM, GPU and dependencies
	$(RUN) scripts/check_env.py

check-connectome: ## print stats for every cached graph
	$(RUN) scripts/check_connectome.py

test:             ## unit tests (slow ones: .venv/bin/python -m pytest tests -m slow)
	$(RUN) -m pytest tests/ -q

# --- reproducing the preprint (docs/REPRODUCIBILITY.md) ---
verify-results:   ## result and frozen-protocol files vs their recorded SHA-256 (no venv needed)
	python3 tools/verify_results.py

verify-data:      ## raw MaleCNS tables and the built graph vs their recorded SHA-256
	@grep -v '^#' results/SHA256SUMS.malecns-v1.0 | sha256sum -c -

figures:          ## the computed figures of the preprint -> artifacts/figures/ (CPU, seconds)
	$(PY) -I scripts/figures/make_figures_taches.py --donnees .
	$(PY) -I scripts/figures/make_figures_campagne.py

calibration:      ## statistical calibration by simulation, 16 scenarios x 20,000 (CPU; stoppable, resumable)
	OMP_NUM_THREADS=2 $(PY) tools/audits/statistical_20261019.py

study-s:          ## visual tasks for one graph, e.g. make study-s GRAPH=nullin:3 (one GPU, ~15 min)
	$(RUN) scripts/diagnostics/study_s.py run --graph $(GRAPH) --trials 6000

study-s-reading:  ## declared reading of the visual tasks over all graphs present
	$(PY) scripts/diagnostics/study_s_reading.py

demo-small:       ## smoke demonstration: development null 100, both tasks, 500 trials (minutes on a GPU)
	$(RUN) scripts/diagnostics/study_s.py run --graph nulllat:100 --trials 500 --tag demo

nulls:            ## null graphs for SEEDS (default 0-39), family S; FAMILY=E for in-strength
	OPENBLAS_NUM_THREADS=1 $(PY) scripts/diagnostics/build_nulls.py $(if $(filter E,$(FAMILY)),--instrength) $(or $(SEEDS),$(shell seq 0 39))

lint:
	.venv/bin/ruff check src scripts tests

clean:
	rm -rf artifacts/runs/* .pytest_cache .ruff_cache

deep-clean: clean ## also drop built graphs (raw downloads are kept)
	rm -rf data/connectome/* data/processed/*

train-mario:      ## start or resume the Mario pilot (sweep regime) in the background
	OMP_NUM_THREADS=4 setsid nohup $(PY) scripts/train_mario_sweep.py --version 2 >> artifacts/logs/mario_sweep_v2.log 2>&1 < /dev/null &
