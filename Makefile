.PHONY: lint typecheck test verify m1 m2 m3 m6 demo-build demo-smoke

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -q

verify: lint typecheck test

m1:
	uv run autorag smoke --input data/fixtures/pdfs --question "What is project scope?" --run-id m1

m2:
	uv run autorag ingest --input data/raw/pdfs --run-id m2 --chunking heading_recursive
	uv run autorag index-vector --run-id m2 --embedding bge-small-en-v1.5
	uv run autorag query --run-id m2 --mode vector --question "How is scope baseline used?" --top-k 8

m3:
	uv run autorag ingest --input data/raw/pdfs --run-id m3 --chunking heading_recursive
	uv run autorag build-kg --run-id m3
	uv run autorag query --run-id m3 --mode graph --question "How does scope control affect risk response?" --top-k 8

demo-build:
	bash scripts/run_m6_demo.sh

demo-smoke:
	bash scripts/run_m6_demo_smoke.sh

m6: demo-build
