# Evaluation Guide

The project evaluates two different systems:

1. The deterministic harness: routing validation, contracts, tools, safety, context packing, and finish gates.
2. The live agent: the harness plus the configured local model and prompts.

A green unit suite does not guarantee that every live coding prompt will pass. A live failure can expose model reasoning weakness, prompt weakness, or a missing harness capability.

## Unit Tests

Run:

```bash
python3 -m unittest discover -s tests
```

The unit suite covers:

- configuration and local Ollama endpoint restrictions
- context compaction and adaptive context retry behavior
- route and JSON action validation
- tool confirmation, workspace confinement, and shell policy
- task-contract parsing, fallback merging, evidence ledgers, and ordering constraints
- repeated action prevention and repair behavior
- coding skill selection

CI runs the unit suite and builds the Docker image on pushes and pull requests.

## Live Suites

The live evaluator creates a new temporary workspace for each prompt and runs the real configured Ollama model:

```bash
python3 scripts/evaluate_agent.py --suite agentic --timeout 300
python3 scripts/evaluate_agent.py --suite smoke
python3 scripts/evaluate_agent.py --suite medium --timeout 300
python3 scripts/evaluate_agent.py --suite hard --timeout 300
```

Equivalent Make targets:

```bash
make eval-agentic
make eval-smoke
make eval-medium
make eval-hard
```

Use `--json` for machine-readable results:

```bash
python3 scripts/evaluate_agent.py --suite agentic --timeout 300 --json
```

Run one prompt:

```bash
python3 scripts/evaluate_agent.py \
  --prompt "write hello.py, run it, and display the output" \
  --timeout 180
```

## Suite Purposes

### `agentic`

Tests orchestration and contract completion rather than algorithm difficulty:

- execute a program multiple requested times
- display source before running
- create, run, then delete a file
- create, run, update, rerun, then delete
- combine local work with a conversational response

These cases use structured checks against output and final workspace state.

### `smoke`

Tests common short coding workflows such as file creation, execution, simple algorithms, and visible results.

### `medium`

Tests multi-case algorithms and data structures with more edge cases and repair opportunities.

### `hard`

Tests more difficult algorithm generation and verification. Failures here are often model-sensitive and should be diagnosed before changing the harness.

## Structured Assertions

Agentic cases can check:

- required or forbidden output text
- case-insensitive alternatives
- minimum and maximum output occurrence counts
- files that must exist
- files that must be absent

This prevents a superficially plausible final message from passing when the requested side effects did not happen.

## How To Diagnose A Failure

Classify the failure before changing code:

1. **Protocol failure**: invalid route, contract, or action JSON.
2. **Harness failure**: unsafe action accepted, valid action rejected, premature finish, repeated loop, or missing evidence enforcement.
3. **Tool/runtime failure**: incorrect command, import path, missing dependency, timeout, or environment problem.
4. **Model reasoning failure**: incorrect implementation, brittle tests, wrong repair, or inability to choose the next useful action despite correct context.
5. **Evaluator failure**: assertion does not represent the user request or accepts false success.

Preserve the failed temporary workspace shown in the evaluator output while diagnosing it. Inspect generated files and compare the observation trace against the task contract.

## Recommended Development Loop

For changes to tools or validation:

```bash
make test
make eval-agentic
```

For changes to prompts, skills, model settings, or repair behavior:

```bash
make test
make eval-agentic
make eval-smoke
```

Run medium and hard suites when the change is intended to improve coding quality rather than only orchestration.

Track at least:

- pass rate by suite
- average steps and latency
- repeated or blocked actions
- false success claims
- command failures and repair rate
- unnecessary tool calls
- final workspace correctness

Do not hide a model-sensitive failure by weakening an evaluator assertion. Change the model, prompt/context, skill guidance, or harness only when the failure analysis supports it.
