# Predictability Horizon

**How far ahead can a learned model predict a chaotic system, and is that limit set by the physics or by the model?**

This project measures learned models of the double pendulum in units of **Lyapunov time**, and separates two sources of error: error that chaos forces on any predictor, and error the model adds. It climbs a four-stage ladder, from validated physics to a learned world model used for planning. The full plan is in [`docs/BRIEF.md`](docs/BRIEF.md).

## Status

Stage 1 (physics ground truth, no ML), milestone 1 of 7: repository baseline.

**No scientific results exist yet.** This section will gain the hero figure and the finding when Stage 1 is tagged `stage1-v1.0`.

## Reproduce the environment

Requires [uv](https://docs.astral.sh/uv/) 0.8.22 or newer.

```bash
uv sync --locked --python 3.12
uv run --locked --python 3.12 ruff check .
uv run --locked --python 3.12 ruff format --check .
uv run --locked --python 3.12 pytest
```

`--locked` makes uv fail, instead of silently re-resolving dependencies, if `uv.lock` is out of date.

Python 3.12 is the **reference environment**: all committed results and figures are generated there. The project supports Python 3.11 as well, but the lockfile resolves an older JAX/NumPy/SciPy stack for it (JAX 0.11 requires Python 3.12 or newer), so CI runs both. See D-024 in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Documents

- [`docs/BRIEF.md`](docs/BRIEF.md): research brief and working rules
- [`docs/DECISIONS.md`](docs/DECISIONS.md): decision log (choice, alternatives, why)
- [`PROGRESS.md`](PROGRESS.md): what is done, what is next, open questions

## License

MIT. See [`LICENSE`](LICENSE).
