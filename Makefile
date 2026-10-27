PYTHON = .venv\Scripts\python.exe
KATALOGI = src/ tests/

SHELL = cmd.exe

.PHONY: install data lint format test tune train all clean

install:
	uv pip install -r requirements.txt

data:
	dvc pull

lint:
	$(PYTHON) -m ruff check $(KATALOGI)

format:
	$(PYTHON) -m ruff format $(KATALOGI)
	$(PYTHON) -m ruff check --fix $(KATALOGI)

test:
	$(PYTHON) -m pytest tests/ -v

tune: data
	$(PYTHON) -m src.tune

train: data
	$(PYTHON) -m src.train

all: lint test train

clean:
	-rmdir /s /q .pytest_cache 2>nul
	-rmdir /s /q .ruff_cache 2>nul
	-FOR /d /r . %%d in (__pycache__) DO @IF EXIST "%%d" rmdir /s /q "%%d"