.PHONY: data build estimate docs notebooks test all

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

all:
	python tools/run_all.py --stage all
