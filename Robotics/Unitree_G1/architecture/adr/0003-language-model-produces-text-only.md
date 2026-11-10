# ADR-0003: The language model produces text only; motion is decided by fixed logic

- **Status:** Draft
- **Date:** 2026-11-06
- **Scope:** Voice dialogue and gestures

## Context

The voice dialogue chain is: on-board speech recognition -> text via DDS -> C++ program -> local language model (Ollama, qwen3.5:9b) -> text -> speech output on the robot, with matching arm gestures through the SDK arm service. A language model can produce free-form output; a humanoid robot can cause physical harm if that output is turned into motion without a check.

## Options considered

1. **Model output controls actions directly** (for example structured commands or tool calls from the model).
2. **Model output is text only (chosen).** Actions such as gestures are triggered by fixed logic in the C++ program, not by the model.

## Decision

The language model only produces text. Whether a movement follows is decided by fixed logic in the C++ program. The same principle applies to navigation (see ADR-0004).

## Consequences

- The set of possible movements is bounded by the C++ code and can be reviewed, tested and traced without evaluating model behavior.
- Model errors (wrong or unexpected text) cannot directly produce an unintended movement.
- Gestures that fit a response need a mapping in code (rules, not model output). This is less flexible than model-driven action selection.
- The model runs locally; no dialogue content leaves the development PC through the model path.

## Evidence

| Statement | Status |
|---|---|
| Response time from end of question to start of answer: 2.4 to 2.7 s | measured or confirmed by source |
| Interruption detection stops the robot mid-sentence | measured or confirmed by source |
| Model output cannot reach the motion interface other than through the fixed logic | assumed or unverified: needs a code-level review or test |
