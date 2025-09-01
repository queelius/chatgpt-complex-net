.PHONY: setup install clean test

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

install:
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	.venv/bin/python -m pytest