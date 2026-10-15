# Predictability Horizon

**How far ahead can a learned model predict a chaotic system, and is that limit set by the physics or by the model?**

This project measures learned models of the double pendulum in units of **Lyapunov time**, and separates two sources of error: error that chaos forces on any predictor, and error the model adds. It climbs a four-stage ladder, from validated physics to a learned world model used for planning. The full plan is in [`docs/BRIEF.md`](docs/BRIEF.md).

## Status

Stage 1 (physics ground truth, no ML): milestones 1 to 4 of 7 are done.

- A double-pendulum simulator whose equations are derived from the Lagrangian and checked against an independent SymPy and mpmath oracle ([derivation](docs/derivations/double_pendulum.md)).
- A fixed-step RK4 in JAX and a SciPy DOP853 reference, with fourth-order convergence checked against four exact solutions and against the pendulum, and a predeclared agreement gate ([derivation](docs/derivations/integrators.md)).
- A physics validation suite: energy drift over `t = 1000`, the small-angle FFT against the normal-mode frequencies, and time reversibility ([derivation](docs/derivations/physics_validation.md)).

Next: two independent Lyapunov-exponent estimators (milestone 5), energy regimes and finite-time exponents (milestone 6), and the figures and the report (milestone 7).

**No scientific results exist yet.** What has been measured so far validates the numerical tools; it is not a finding about predictability. This section will gain the hero figure and the finding when Stage 1 is tagged `stage1-v1.0`.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) 0.8.22 or newer and Python 3.12 or newer (`.python-version` pins 3.12; uv fetches it if needed).

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

`uv run --locked pytest` runs every test, the unchanged gates included. The suite has 613 tests and takes 3 to 4 minutes on a laptop, and between 3 and 10 minutes on the CI runner, depending on the runner's speed (GitHub Actions, Ubuntu 24.04, one pinned Python version).

### Test tiers

Every test is marked `fast` or `full`; the marker is assigned at collection from [`tests/tiers.py`](tests/tiers.py), so the gate test files carry no decorators and are unchanged.

```bash
uv run --locked pytest -m fast        # the fast tier: about 30 s on a laptop
uv run --locked pytest                # everything: the full tier
uv run --locked pytest -n auto        # optional: spread the tests over the cores (pytest-xdist)
```

The **fast tier is a smoke tier and never a gate.** It holds the cheap tests and reduced versions of the heavy gates (fewer states, shorter horizons, the same constants from `horizon.tolerances`, each reduced control still failing); it can miss what only the long horizons and the full ensembles show. Every claim of the project rests on the full tier. Details are in D-066 to D-068 of the decision log.

In CI (`.github/workflows/ci.yml`), a push to a branch other than `main` runs the job *fast tier (smoke)*: lint, format check and `pytest -m fast`, plus a guard that fails the job if any fast test, setup or teardown takes 15 s or more. A pull request to `main`, a push to `main` and a manual run (`workflow_dispatch`) run the job *full tier (gates)*: lint, format check and `pytest` with no marker filter. Both jobs run with `-n auto` (one worker per core, the default `--dist load`): the fast job as D-067's rule for `-n` decided from laptop measurements (D-068), the full job at the owner's choice (D-069). Pull requests to a base other than `main` get no CI.

The two scripts below print the measured numbers of the validation tests next to their predeclared thresholds, with a provenance header, and write no files:

```bash
uv run --locked python -m experiments.stage1.integrator_validation   # convergence and agreement
uv run --locked python -m experiments.stage1.physics_validation      # all five validation tests, about 4 minutes
```

`--locked` makes uv fail, instead of silently re-resolving dependencies, if `uv.lock` is out of date. There is a single locked stack, so every committed result comes from the same library versions. See D-027 in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## How the repository is made

The equations and the numerical mathematics are derived and checked by the repository owner. The code is implemented by an AI coding assistant from those derivations, and the owner reviews it. Each design decision is logged with its alternatives in [`docs/DECISIONS.md`](docs/DECISIONS.md), and some were reviewed by a second model before any code was written. Every threshold is declared in [`src/horizon/tolerances.py`](src/horizon/tolerances.py) before the code it governs is tested, and a gate that fails is reported and not adjusted. Reported results come from committed scripts; the numbers behind the thresholds come from calibration runs that are not committed and are labelled as such in the log.

## Layout

- [`src/horizon/`](src/horizon): the package: the double-pendulum mechanics (`systems/double_pendulum.py`), the integrators (`integrators.py`) and the declared tolerances (`tolerances.py`).
- [`tests/`](tests): the tests, the independent oracle (`tests/oracles/`) and the measurement code shared by the tests and the scripts.
- [`experiments/stage1/`](experiments/stage1): the scripts that print the validation numbers.
- [`docs/`](docs): the brief, the decision log, the derivation notes and the plan.

## Documents

- [`docs/BRIEF.md`](docs/BRIEF.md): research brief and working rules
- [`docs/DECISIONS.md`](docs/DECISIONS.md): decision log (choice, alternatives, why)
- [`docs/derivations/`](docs/derivations): the derivation notes, each with a table that maps every claim to the test that checks it
- [`docs/plans/stage1.md`](docs/plans/stage1.md): the Stage 1 plan and its milestones
- [`PROGRESS.md`](PROGRESS.md): what is done, what is next, open questions

## License

MIT. See [`LICENSE`](LICENSE).
