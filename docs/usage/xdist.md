# Parallel Execution with pytest-xdist

RAMPART supports parallel test execution via `pytest-xdist`, producing a **single unified report** even when tests run across multiple worker processes.

---

## Quick Start

```bash
pip install pytest-xdist
pytest -n 4
```

With `-n 4`, pytest spawns 4 worker processes that execute tests in parallel. RAMPART intercepts each worker's results, ships them to the controller process, and emits **one consolidated report** at the end of the session.

---

## How It Works

```
Worker 1                    Worker 2                    Controller
─────────                   ─────────                   ──────────
collect results             collect results
    │                           │
pytest_sessionfinish        pytest_sessionfinish
    │                           │
serialize → workeroutput    serialize → workeroutput
    │                           │
    └───────────┬───────────────┘
                ▼
        pytest_testnodedown (per worker)
        deserialize + merge into
        controller's RampartSession
                │
                ▼
        pytest_sessionfinish (controller)
        aggregate trials → evaluate gates → emit sinks
                │
                ▼
        Single unified TestRunReport
```

- **Workers** collect [`Result`][rampart.core.result.Result] objects normally and hand them to the controller. Workers do **not** emit reports.
- **Controller** receives each worker's results via the `pytest_testnodedown` hook, merges them into its own [`RampartSession`][rampart.pytest_plugin._session.RampartSession], and emits sinks once at session end.

The result: **one** `JsonFileReportSink` output file, **one** call to `MyCustomSink.emit_async`, and accurate population statistics over the full result set.

### Result transport

How a worker's results reach the controller depends on the run topology:

- **Durable shard transport (local `popen` workers).** For a plain `pytest -n N`
  run — or any `--tx` topology where every gateway is a local `popen` — each
  worker streams every finished result to its own on-disk JSONL shard,
  flushing after each write. The controller reads the shards back in
  `pytest_testnodedown`. Because results hit disk as soon as they finish, a
  worker that is killed mid-run **keeps every result it had already produced**,
  and the size cap applies **per record** (see [Size cap](#size-cap)).
- **Inline fallback (remote or proxied workers).** When any gateway is remote
  (`--tx=ssh=…`, `--tx=socket=…`) or routed through a `via` proxy, the
  controller cannot read a worker-local shard file, so RAMPART falls back to the
  legacy transport: each worker serializes its full result set into
  `config.workeroutput` once at its clean `pytest_sessionfinish`. This is the
  batch-at-end behavior — a worker killed before session finish contributes
  nothing, and the size cap applies to the whole payload.

Both transports feed the same merge-and-emit path, so reports are identical
apart from the durability characteristics above. Shard files live in a private
temporary directory and are deleted at the end of the run (pass
`--rampart-keep-shards` to retain them for debugging).

---

## Trial Tests with xdist

`@pytest.mark.trial(n=, threshold=)` clones a test into N independent runs. Under xdist, clones may be distributed across workers depending on the `--dist` mode.

| `--dist` mode | Trial behavior |
|---------------|----------------|
| `loadgroup` | All trial clones for one test pinned to the same worker |
| `load` (default) | Trial clones distributed across all workers |
| `loadscope` / `loadfile` | Grouped by class/module/file |

**Correctness is preserved regardless of mode** — the controller aggregates trial groups from the merged result set and evaluates each group's threshold against the full population. You'll see a warning if you use `@trial` markers without `--dist=loadgroup`:

```text
RAMPART @trial markers present with --dist=load. Trial clones may be
split across workers. Aggregation remains correct (controller merges
all results), but using --dist=loadgroup keeps trial clones co-located
on one worker for better locality.
```

This warning is **informational, not a correctness signal** — see below for when it's safe to ignore.

### Choosing `loadgroup` vs `load`

**Both modes produce an identical, correct report.** The controller merges per-worker
partials into one population and evaluates each trial's threshold against the full
group either way. The choice is about *execution*, not correctness:

- **`load` (default)** spreads a test's trial clones across **all** workers, so a
  20-clone trial keeps every worker busy. It is usually the **fastest** option and is
  the right default when trial clones are **independent** (no shared per-group state).
- **`loadgroup`** pins all clones of one trial group to a **single** worker. Prefer it
  only when a trial group needs **cohesion** — e.g. clones share a session-scoped
  fixture, a per-group cache/connection, or other worker-local state that must not be
  split across processes. The trade-off is less parallelism, so it can run slower.

**Rule of thumb:** independent trials → plain `pytest -n 4` (faster); trials that
share per-group worker state → `pytest -n 4 --dist=loadgroup`.

As an illustration, one 22-item suite containing a 20-clone trial measured:

| Mode | Command | Wall time | Reports | `total_runs` |
|------|---------|-----------|---------|--------------|
| Serial | `pytest -n 0` | 203.4s | 1 | 22 |
| Parallel, loadgroup | `pytest -n 4 --dist=loadgroup` | 165.5s | 1 | 22 |
| Parallel, default load | `pytest -n 4` | **113.8s** | 1 | 22 |

All three emit the same single report and the same trial verdict; `load` is fastest
here because the 20 clones fan out across the 4 workers instead of being pinned to one.

---

## Registering Sinks: the `pytest_rampart_sinks` hook

The **recommended** way to register report sinks is the `pytest_rampart_sinks`
hook. It is resolved on the controller — which never executes fixtures — so it
behaves identically in single-process and xdist runs, and (unlike the fixture
path) supports sinks that need configuration.

Implement it in your `conftest.py`:

```python
# conftest.py
from pathlib import Path

from rampart.reporting import JsonFileReportSink


def pytest_rampart_sinks(config):
    return [JsonFileReportSink(output_dir=Path(".report"))]
```

- Multiple implementations are supported; RAMPART emits to the **union** of every
  returned sink.
- An implementation may return an empty list to contribute none.
- Non-`ReportSink` items (or a non-list return) are dropped with a warning, so one
  malformed implementation cannot break emission.

### Precedence vs the `rampart_sinks` fixture

The legacy `rampart_sinks` fixture is still supported as a **single-process
fallback**. The rule is:

- If **any** `pytest_rampart_sinks` hook implementation exists, the hook is
  authoritative and the fixture path is skipped entirely (so a project that
  defines both does **not** double-register).
- If **no** hook implementation exists, RAMPART falls back to the fixture. On the
  xdist controller this fallback scans registered conftest modules for a
  `rampart_sinks` attribute.

### Fixture fallback constraints (no hook present)

When you rely on the fixture fallback under xdist, pytest's fixture machinery
does not run on the controller. RAMPART therefore unwraps a **parameterless**
`rampart_sinks` fixture and calls its underlying function directly, so these
shapes resolve:

```python
# Parameterless session fixture — resolves single-process AND on the
# xdist controller.
@pytest.fixture(scope="session")
def rampart_sinks():
    return [JsonFileReportSink(output_dir=Path(".report"))]

# Plain list assigned at module level — resolved on the xdist controller
# only. Single-process discovery looks up a *fixture* named rampart_sinks,
# so a bare module-level list is silently ignored there; use the fixture
# form above (or the hook) for single-process runs.
rampart_sinks = [JsonFileReportSink(output_dir=Path(".report"))]
```

A **fixture with dependencies** cannot be resolved on the controller and is
skipped with a warning:

```python
# Not resolvable on the controller — use the hook instead
@pytest.fixture(scope="session")
def rampart_sinks(my_sink_config, db_connection):
    return [DatabaseSink(connection=db_connection)]
```

If your sinks need dependencies, **use the `pytest_rampart_sinks` hook** — it
receives the `pytest.Config` and runs on the controller, so you can build sinks
from `config` values or environment variables there.

---

## Trust Boundary & Security

Worker payloads cross a process boundary via `execnet` and may contain attacker-controlled content (agent responses, payload text, evaluator rationale). RAMPART's serialization defends against:

- **Arbitrary code execution** — strict JSON-safe primitives only; no `pickle`, `marshal`, or custom `__reduce__`.
- **Schema drift** — payloads with missing or unknown schema versions are rejected fail-closed.
- **Memory exhaustion** — result payloads are capped (64 MB by default). Under the durable shard transport the cap is applied **per result record**; under the inline fallback it caps the whole worker payload.
- **Terminal/log injection** — ANSI escape sequences are stripped from free-form text at the deserialization boundary.
- **Path traversal** — worker-local artifact paths are stored as opaque strings in metadata; the controller never accesses worker files.

### Size cap

The default 64 MB cap can be overridden via the pytest CLI option or an ini setting:

```bash
pytest -n 4 --rampart-xdist-max-bytes=134217728
```

Or in `pytest.ini` / `pyproject.toml`:

```ini
[pytest]
rampart_xdist_max_bytes = 134217728
```

**Under the durable shard transport** (local `popen` workers) the cap is applied
**per result record**: a single oversized result is written to the shard as a
truncation marker and dropped, while every other result from the same worker is
recovered normally. The controller records the run as incomplete in
`TestRunReport.metadata`, naming how many records were recovered and dropped.

**Under the inline fallback** (remote/proxied workers) the cap applies to the
worker's whole serialized payload: exceeding it drops the entire payload and
marks that worker incomplete.

---

## Incomplete Runs

If a worker crashes, runs out of time, or hits the size cap, the controller marks the run as incomplete:

```python
report.metadata["incomplete"]            # True if any worker failed
report.metadata["incomplete_reasons"]    # list[str] — one per failure
```

Reports are still emitted with whatever data was collected. For safety-critical CI, sinks or post-processing should check the `incomplete` flag and fail the build accordingly.

---

## Run-Mode Metadata

Reports produced under xdist include:

```python
report.metadata["xdist_active"]   # True
report.metadata["worker_count"]   # int
report.metadata["dist_mode"]      # "load", "loadgroup", etc.
```

---

## Durability

For local `pytest -n N` runs (and any all-`popen` `--tx` topology), RAMPART uses
a **durable per-worker shard transport**: each worker streams every finished
result to its own on-disk JSONL shard, flushing after each write. This gives two
guarantees the earlier batch-at-end transport could not:

- **A worker killed mid-run keeps its already-finished results.** Because each
  result hits disk the moment it completes, a worker that crashes, is killed
  (OOM, timeout, `-x` shutdown), or otherwise never reaches
  `pytest_sessionfinish` still contributes every test it had finished. The
  controller recovers those results from the shard and marks the run incomplete
  (see [Incomplete Runs](#incomplete-runs)), rather than losing the worker's
  entire contribution.
- **The size cap drops only the oversized record.** When one result exceeds
  `--rampart-xdist-max-bytes`, just that record is dropped; the worker's other
  results are recovered normally (see [Size cap](#size-cap)).

### Remote topologies fall back to the inline transport

The shard transport requires the controller to read worker-local files, so it is
used **only** when every gateway is a local `popen` process. Remote (`--tx=ssh`,
`--tx=socket`) or `via`-proxied topologies transparently fall back to the inline
`workeroutput` transport, which retains the earlier characteristics: results ship
in a single batch at a worker's clean `pytest_sessionfinish` (a worker killed
before then contributes nothing), and the size cap drops the whole worker
payload. Size your cap to your largest expected payload when running remote
workers.

---

## Limitations

- Sinks discovered through the **fixture fallback** on the controller cannot depend
  on other pytest fixtures — use the `pytest_rampart_sinks` hook instead (see
  [Registering Sinks](#registering-sinks-the-pytest_rampart_sinks-hook)).
- On **local** (`popen`) workers, results survive a worker killed mid-run and the
  size cap is per-record. On **remote/proxied** workers RAMPART falls back to the
  inline transport, where a worker that dies before `pytest_sessionfinish` is lost
  and an over-cap payload is dropped wholesale (see [Durability](#durability)).
- Mixed RAMPART versions across controller and workers are unsupported; install the
  same version everywhere.
- `pytest-xdist` itself does not support interactive debugging (`--pdb`, `--trace`);
  use single-process mode for debugging.
