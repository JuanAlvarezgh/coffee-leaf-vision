.PHONY: setup test lint format run-api run-dashboard

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install
	python -c "import shutil, os; shutil.copyfile('.env.example', '.env') if not os.path.exists('.env') else print('.env exists')"

test:
	pytest tests/ -v

lint:
	black --check .
	ruff check .
	isort --check-only .

format:
	black .
	isort .
	ruff check --fix .

run-api:
	uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

run-dashboard:
	streamlit run dashboard/app.py
