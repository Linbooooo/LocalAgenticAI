# Evaluation

The project uses three layers. Each answers a different question.

## 1. Harness Tests

```bash
make test
```

The unit suite tests the deterministic code:

- Bash-block extraction
- linear observation flow
- recovery after a failed command
- stopping after declined approval
- context packing
- local endpoint validation
- shell confirmation and safety policy
- streamed Ollama metrics

These tests use a fake model. They prove the harness works, not that the model writes correct code.

## 2. Local Coding Benchmark

```bash
make benchmark-agent
```

`scripts/benchmark_agent.py` gives the model four tasks:

- create and run a program
- repair existing code
- implement an algorithm
- extend existing code without regression

The model is not shown the evaluator assertions. After the agent finishes, a separate hidden command imports and tests the resulting code. This avoids scoring the model's own tests or final claims.

Reported metrics:

- pass rate
- TFS: task start to first shell action
- model TTFT
- generation TPS
- end-to-end task latency
- turns and commands per task

Run one case while debugging:

```bash
python3 scripts/benchmark_agent.py --case repair-existing-code --json
```

## 3. SWE-bench Lite

[SWE-bench](https://github.com/SWE-bench/SWE-bench) evaluates patches against real GitHub issues and repository tests. SWE-bench Lite contains 300 tasks and is the selected external accuracy benchmark.

Install its optional tooling:

```bash
python3 -m pip install datasets swebench
```

Generate one local-model prediction:

```bash
python3 scripts/swebench.py \
  --instance-id sympy__sympy-20590 \
  --limit 1
```

The script:

1. loads the official dataset row
2. clones the repository and checks out `base_commit`
3. runs this local agent on the issue statement
4. captures `git diff --binary`
5. writes official prediction fields: `instance_id`, `model_name_or_path`, and `model_patch`
6. saves the linear trajectory and latency metrics separately

Score with the official Docker evaluator:

```bash
python3 scripts/swebench.py \
  --instance-id sympy__sympy-20590 \
  --limit 1 \
  --evaluate
```

Inference remains local. Dataset download, repository cloning, and evaluator image setup require network access. The official evaluator can require substantial disk, memory, and build time; its Docker result, not patch generation alone, determines resolved-task accuracy.

## Recorded SWE-bench Result

| Date | Dataset | Instance | Patch | Official result |
|---|---|---|---|---|
| 2026-06-09 | SWE-bench Lite | `sympy__sympy-20590` | empty | 0/1 resolved; classified as an empty patch |

The saved trajectory showed the model inspecting repository history but misunderstanding the issue and finishing without a code change. This is a model-accuracy result, not an evaluator or action-protocol error.

## Interpreting Failures

- Harness test failure: deterministic control or safety bug.
- Hidden-test failure with a valid trajectory: model, prompt, or context quality problem.
- No shell action: instruction-following/model capability problem.
- Command/runtime failure followed by recovery: expected agent behavior.
- SWE-bench patch generated but unresolved: valid benchmark attempt, incorrect solution.
- Evaluator setup failure: infrastructure result, not model accuracy.

Keep benchmark prompts and hidden verification stable when comparing models or architecture changes.
