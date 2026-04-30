.PHONY: setup test lint typecheck run help

help:
	@echo "Available commands:"
	@echo "  make setup     - Install dependencies"
	@echo "  make test      - Run unit tests with pytest"
	@echo "  make lint      - Check code style with ruff"
	@echo "  make format    - Format code with ruff"
	@echo "  make typecheck - Run static type checking with mypy"
	@echo "  make run       - Launch the Streamlit dashboard"

setup:
	pip install -r requirements.txt
	pip install ruff mypy pytest pytest-cov

test:
	pytest --cov=src tests/

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/

run:
	streamlit run app.py
