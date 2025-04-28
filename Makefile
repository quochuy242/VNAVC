# Makefile for Python project

.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CRAWLER := tts_data_pipeline/crawler/main.py
TEXT_PROCESSING := tts_data_pipeline/pre_processing/text.py
AUDIO_PROCESSING := tts_data_pipeline/pre_processing/audio.py
ALIGNMENT := tts_data_pipeline/alignment/main.py


venv:
	@echo "Creating virtual environment..."
	@python -m venv $(VENV)
	@echo "Done"

install: venv
	@echo "Installing dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Done"

clean:
	@echo "Cleaning virtual environment..."
	@rm -rf $(VENV)
	@echo "Done"

update_deps:
	@echo "Updating dependency file..."
	@$(PIP) freeze > requirements.txt
	@echo "aeneas==1.7.3.0" >> requirements.txt
	@echo "Done"

download_all:
	@echo "Downloading all books..."
	@$(PYTHON) $(CRAWLER) --download "all"
	@echo "Done"