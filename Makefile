.PHONY: install lint test baseline train train-save predict leakage fairness validation calibration plots shap reports compile app

PYTHON ?= python3
PYTHONPATH := src

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests app.py

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall src tests

baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.train_baseline

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.train_models

train-save:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.train_models --save-best-model

predict:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.predict --input-path examples/sample_applications.csv --output-path reports/sample_predictions.csv

leakage:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.leakage

fairness:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.fairness

validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.validation

calibration:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.calibration

plots:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.report_plots

shap:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m loan_status_prediction.shap_analysis

reports: leakage fairness validation calibration plots shap

app:
	PYTHONPATH=$(PYTHONPATH) streamlit run app.py
