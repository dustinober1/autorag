.PHONY: lint typecheck test m1 m2

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -q

m1:
	uv run autorag smoke --input data/fixtures/pdfs --question "What is project scope?" --run-id m1

m2:
	uv run autorag ingest --input data/raw/pdfs --run-id m2 --chunking heading_recursive
	uv run autorag index-vector --run-id m2 --embedding bge-small-en-v1.5
	uv run autorag query --run-id m2 --mode vector --question "How is scope baseline used?" --top-k 8
