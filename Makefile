VERSION := 0.0.0
PYTHON_VERSION := C:\Users\cirkunov\AppData\Local\Programs\Python\Python311\python.exe


.PHONY: run
uvicorn: venv requirements
	poetry run uvicorn --reload --log-level=info --workers 2 --host 0.0.0.0 --port 8000 server.api:app

venv:
	pip install jsonschema==4.17.3 poetry
	poetry env use ${PYTHON_VERSION}

.PHONY: pre
pre:
	poetry run ruff . --fix
	poetry run isort . --profile black
	poetry run black . --skip-string-normalization --line-length=120

.PHONY: requirements
requirements: venv
	poetry install

.PHONY: clean
clean:
	poetry env remove ${PYTHON_VERSION}
	rm -rf dist
