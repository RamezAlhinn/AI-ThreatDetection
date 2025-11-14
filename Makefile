.PHONY: setup run test clean help

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

help:
	@echo "AI Threat Detection Agent - PoC"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make run      - Start the Streamlit UI"
	@echo "  make test     - Run unit tests"
	@echo "  make clean    - Remove virtual environment and cache files"
	@echo ""
	@echo "Quick start: make setup && make run"

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete!"
	@echo "  Run 'make run' to start the application"

run:
	@echo "Starting AI Threat Detection Agent UI..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(BIN)/streamlit run src/ui_app.py

test:
	@echo "Running tests..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(BIN)/pytest tests/ -v

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"
