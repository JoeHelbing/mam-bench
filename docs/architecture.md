# Source walkthrough

For either simulation, start at [main.py](../main.py): parse an explicit model file
and suite, then [config.py](../src/mam_bench/config.py) validates the complete nested
configuration. [model_setup.py](../src/mam_bench/model_setup.py) discovers the
served context and available capabilities from vLLM or OpenRouter once before
any case, and validates YAML request overrides against known constraints. It
defaults action and summary output caps to one quarter of served context, bounded
by known provider output limits. Omitted sampling and reasoning settings use
endpoint defaults. The runner records the input
config and effective setup, then creates one provider, one fresh `CaseRuntime`
and writer per case with discovered context passed separately from YAML settings,
constructs the selected simulation, and awaits
`evaluate(runtime)`. It never steps a world.

[runtime.py](../src/mam_bench/runtime.py) selects native `VLLMProvider` or
`OpenRouterProvider` profiles and applies configured request settings.
Native profiles retain their tool-choice and reasoning safeguards. The runtime
enables native OpenRouter cache controls without a separate cache policy.
Model requests receive no automatic sampling seed; simulation random streams
remain addressed by the case seed. [kludge.py](../src/mam_bench/kludge.py) collects
temporary client workarounds: vLLM repetition stops, stream capture, explicit
thinking settings, and OpenAI's fixed-array schema encoding. Each documents its
cause and removal condition there. Server-side Qwen, Gemma and Bolt corrections
are explained in the [vLLM serving guide](vllm-serving-workarounds.md), for
operators to apply to their own serving installation.
An explicit `tool_choice: auto` opts out of forcing; otherwise native output
selection decides when forcing is supported. Session history is repaired by
PydanticAI before requests, rather than rewritten after stopped turns.

For Schelling, [simulation.py](../src/mam_bench/simulations/schelling/simulation.py)
constructs two distinct worlds through the same constructor, executes its single
step loop for each, and writes initial/completed states. Its
[board](../src/mam_bench/simulations/schelling/board.py) owns spatial state and
measurements. Persistent [agents](../src/mam_bench/simulations/schelling/agents.py)
choose actions; the simulation validates/reserves and settles them. Its
[result](../src/mam_bench/simulations/schelling/results.py) calculates signed lift.

For Civil Violence,
[simulation.py](../src/mam_bench/simulations/civil_violence/simulation.py) follows the
same evaluate/run/step path. One citizen-then-police sequence serves both worlds.
Persistent [agents](../src/mam_bench/simulations/civil_violence/agents.py) use
citizen and police role bases with ordinary and model-controlled variants.
Ordinary proposals remain combined;
model-controlled agents choose activity or an arrest target and move or stay in
either order, then `submit` the combined action. Jailed citizens `defer`.
The simulation applies custody and atomic phase settlement before measuring
participation and checking revolution.
Its [result](../src/mam_bench/simulations/civil_violence/results.py) calculates the
participation and revolution components through the same `score` property.

Both use [sessions.py](../src/mam_bench/sessions.py) for individual model turns,
private history/memory, and communication. Instructions carry identity, goal,
and standing rules; user inputs contain changing JSON observations. Each
simulation owns legality and settlement. [scheduling.py](../src/mam_bench/scheduling.py)
shares cohort ordering, concurrent admission, failure cleanup, and
addressed random streams. [artifacts.py](../src/mam_bench/artifacts.py) writes supplied
records and preserves native archives. The runner retains case
results, and `BenchmarkResult.total_score` sums their calculated scores.

At run-directory creation, the CLI enters [telemetry.py](../src/mam_bench/telemetry.py)
in the task that owns the benchmark. It configures native Pydantic AI
instrumentation and OpenTelemetry providers/exporters for local traces, logs,
and metrics, with optional Logfire delivery. Harness remains the full conversation
archive; native call IDs connect exported traces to saved messages. Native
exception details are retained in traces and archives; package logs also retain
provider error bodies, including streaming errors.
