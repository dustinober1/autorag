# Known-Good Run Snapshot Template

## Identity
- Date (UTC):
- Operator:
- Git branch:
- Git commit:
- Python version:
- `uv` version:

## Inputs
- Fixture pack ID: `m8_demo_fixture_pack_v1`
- Fixture pack config: `configs/demo_fixture_pack.yaml`
- Artifact root:
- Demo run ID:
- Demo mode:
- Demo question:
- Top K:

## Commands
```bash
make verify
make demo-smoke
uv run autorag doctor --run-id <run_id>
```

## Gate Results
- `make verify`: pass/fail
- `make demo-smoke`: pass/fail
- `autorag doctor`: pass/fail

## Required Artifact Checks
- `data/artifacts/<run_id>/answers.jsonl` valid JSONL + `answer_text` + `citations`
- `data/artifacts/<run_id>/demo_payload_samples.jsonl` valid JSONL + `question` + `answer_record.answer_text`
- `reports/experiments/matrix_results.json` object with `run_id`, `summary`, and non-empty `rows`
- `reports/experiments/matrix_results.csv` header + at least one data row
- `reports/milestones/m6_demo_report.md` non-empty
- `reports/experiments/leaderboard.md` non-empty

## Notes
- Variance or anomalies:
- Follow-up TODOs:
