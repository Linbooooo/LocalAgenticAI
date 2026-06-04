# Agent Architecture

This project is a local-only coding agent built around a small observe-act loop.

```mermaid
flowchart TD
    A["User request"] --> B["LocalAgent.run()"]

    B --> C["Exact command check"]
    C -->|direct command| D["Run shell directly"]
    C -->|not exact command| E["LLM semantic router"]

    E --> F["Validate route"]
    F -->|chat| G["Normal Ollama answer"]
    F -->|hardware| H["Gather CPU/GPU/RAM/Ollama status"]
    F -->|read/edit/shell| I["Action loop"]

    H --> G

    I --> J["Build action context"]
    J --> K["Current user request"]
    J --> L["Workspace snapshot"]
    J --> M["Structured agent state"]
    J --> N["Selected coding skills"]
    J --> O["Current-task observations"]

    K --> P["LLM chooses one JSON action"]
    L --> P
    M --> P
    N --> P
    O --> P

    P --> Q["Validate action protocol"]
    Q -->|invalid| R["Retry with protocol correction"]
    R --> P

    Q -->|valid| S["Execute local tool"]
    S --> T["Record observation"]

    T --> U{"Done?"}
    U -->|needs more work| J
    U -->|verified or finished| V["Final response"]
```

## Core Idea

The model decides the next useful step, but the harness decides whether that step is valid and safe.

```text
LLM reasoning
+ deterministic route/action validation
+ workspace-confined local tools
+ coding workflow skills
+ iterative observation loop
```

The result is a coding assistant that can inspect files, edit code, run local commands, observe failures, and continue toward a verified result without giving the model unrestricted machine access.

## Workflow

1. The user sends a request.
2. `LocalAgent` checks for an exact direct command such as `execute "nvidia-smi"`.
3. Otherwise, the model classifies the request as `chat`, `read`, `edit`, `shell`, or `hardware`.
4. The deterministic harness validates the route and downgrades risky low-confidence action routes to chat.
5. For action routes, the agent builds a self-contained context packet from the current request, workspace files, structured agent state, selected coding skills, completion requirements, and observations from the current task.
6. The model returns exactly one JSON action.
7. The harness validates the action protocol and tool permissions.
8. A local tool runs, and the result is recorded as an observation.
9. The loop continues until the task is complete, verification succeeds, safety blocks the action, or the step limit is reached.

Validation is not just JSON shape checking. The harness also rejects repeated failed command proposals, repeated identical rewrites, missing test directories for unittest discovery, Python script runs that cannot produce requested output, and success claims that lack required run/test evidence.

## Protocol Isolation

Normal chat can use conversation history. Tool planning should not depend on prior assistant prose.

Route and action calls therefore use isolated protocol messages. The protocol prompt explicitly includes:

```text
current user request
workspace snapshot
structured agent state
completion requirements
selected coding skills
current-task observations
```

Structured state carries compact references such as the last written file and last shell result. This lets the model resolve follow-ups like "test it" while avoiding a common failure mode where a previous successful command result pollutes the next JSON action decision.

## Tools Versus Skills

Tools are primitive capabilities:

```text
list_files
read_file
search_text
write_file
replace_in_file
run_shell
hardware_profile
```

Skills are procedural guidance for using those tools correctly:

```text
coding-change
project-discovery
python-testing
debugging
algorithm-verification
```

For example, `python-testing` does not run tests by itself. It reminds the model to use the correct project-root command for unittest files, while the existing shell tool and safety policy still control execution.

The same separation applies to repair work. `debugging` and `algorithm-verification` tell the model how to interpret failures, while `LocalAgent` decides whether the proposed next action is admissible and whether the loop has enough evidence to finish.
