.PHONY: setup lint test run repro clean data build estimate docs notebooks all

setup:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check src tests

data:
	python tools/run_all.py --stage data

build:
	python tools/run_all.py --stage build

estimate:
	@echo "DEPRECATED: inference/estimation is disabled for the descriptive atlas pipeline."
	@exit 1

docs:
	python tools/run_all.py --stage docs

notebooks:
	python tools/run_all.py --stage notebooks

test:
	pytest

run:
	python tools/run_all.py --stage paper

repro:
	make clean
	make run
	make test

clean:
	rm -rf output outputs data/02_intermediate data/03_clean data/04_analysis

all:
	python tools/run_all.py --stage all
