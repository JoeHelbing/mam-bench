# Source walkthrough

For either simulation, start at [main.py](../main.py): parse an explicit model file
and suite, then [config.py](../src/mam_bench/config.py) validates the complete nested
configuration. [BenchmarkRunner](../src/mam_bench/runner.py) creates one provider,
one fresh `CaseRuntime` and writer per case, constructs the selected simulation,
and awaits `evaluate(runtime)`. It never steps a world.

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
shares cohort ordering, concurrent admission, claim gates, failure cleanup, and
addressed random streams. [artifacts.py](../src/mam_bench/artifacts.py) writes supplied
records and preserves native archives. The runner retains case
results, and `BenchmarkResult.total_score` sums their calculated scores.

There are no reset/reinitialization paths, alternate constructors, old YAML
adapters, simulation-model matrices, base-simulation framework, or reference
replay dependency. The obsolete 60-step/two-agent calibration selector and tests for
retired interfaces are removed. Historical result files are untouched.
