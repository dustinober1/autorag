.PHONY: lint typecheck test m1

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -q

m1:
	uv run autorag smoke --input data/fixtures/pdfs --question "What is project scope?" --run-id m1
