# multiqc-foundation-gate -- MultiQC-driven QC pipeline scaffold
# Targets are deliberately small. Every repo using this scaffold should be
# reproducible end-to-end with `make data && make run && make test && make report`.

PYTHON ?= .venv/bin/python
UV ?= uv
PKG := multiqc_gate
RUN_NAME ?= demo
ARTIFACT_DIR := artifacts
DATA_DIR := data
REPORT_DIR := reports

.PHONY: help install data run test report lint clean canary

help:
	@echo "make install      Install pinned dependencies via uv"
	@echo "make data         Download + checksum-verify public inputs from data/manifest.yaml"
	@echo "make run          Run the end-to-end pipeline (audit + MLflow hooks engaged)"
	@echo "make test         Run pytest"
	@echo "make report       Render demo notebook to HTML at reports/demo.html"
	@echo "make lint         ruff check"
	@echo "make canary       Run the deterministic canary smoke test"
	@echo "make  Check the honest-scope preamble is present in README"
	@echo "make clean        Remove build artifacts (raw data left alone)"

install:
	$(UV) sync

data:
	$(PYTHON) -m $(PKG).pipeline fetch --manifest $(DATA_DIR)/manifest.yaml --out $(DATA_DIR)

run: | $(ARTIFACT_DIR)
	$(PYTHON) -m $(PKG).pipeline run --name $(RUN_NAME) --out $(ARTIFACT_DIR)

test:
	$(PYTHON) -m pytest -q

report: | $(REPORT_DIR)
	$(PYTHON) -m jupyter nbconvert --to html --output-dir $(REPORT_DIR) notebooks/demo.ipynb

lint:
	$(PYTHON) -m ruff check src tests

canary:
	$(PYTHON) -m $(PKG).canary


clean:
	rm -rf $(ARTIFACT_DIR) $(REPORT_DIR) .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

$(ARTIFACT_DIR) $(REPORT_DIR):
	mkdir -p $@
