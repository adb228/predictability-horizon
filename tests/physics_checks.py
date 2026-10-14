"""Shared definitions for the milestone 4 validations: seeded states, measurements, helpers.

The test modules and ``experiments/stage1/physics_validation.py`` both import from here, so a
gate and the number printed for it have one implementation. Every value comes from
``horizon.tolerances`` (D-052 to D-060); nothing is chosen or tuned in this module. The claims
and the tests that check them are tabulated in ``docs/derivations/physics_validation.md``,
section 6.

Measuring and checking are separate, as in ``tests/integrator_checks.py``: a measurement returns
numbers and never raises on a bad value, so the script can print them. The check helpers
(``check_at_most`` and the others) are reused from that module.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from horizon import integrators
from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from tests import integrator_checks as chk

DEFAULT = dp.DEFAULT_PARAMS
FLIP = np.array([1.0, 1.0, -1.0, -1.0])
"""``S``: flips the two velocities (section 1 of the note)."""

Field = Callable[..., jax.Array]


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


# ---- seeded states (D-054, D-055, D-057) ---------------------------------------------------


def _regular_rest_states(stream: int, n: int) -> np.ndarray:
    """``n`` states at rest, both angles uniform in ``[-REGULAR_THETA_MAX, REGULAR_THETA_MAX]``."""
    rng = np.random.default_rng([tol.CASES_SEED, stream])
    theta = rng.uniform(-tol.REGULAR_THETA_MAX, tol.REGULAR_THETA_MAX, size=(n, 2))
    return _frozen(np.hstack([theta, np.zeros_like(theta)]))


def _chaotic_rest_states(stream: int, n: int) -> np.ndarray:
    """``n`` chaotic-ish states at rest, the recipe of D-046(a) under the default parameters."""
    rng = np.random.default_rng([tol.CASES_SEED, stream])
    return _frozen(chk.draw_rest_states(rng, DEFAULT, n))


@functools.cache
def e1_regular_states() -> np.ndarray:
    """The regular gate states of E1, shape ``(E1_N_REGULAR, 4)``."""
    return _regular_rest_states(tol.E1_STREAM_REGULAR, tol.E1_N_REGULAR)


@functools.cache
def e1_chaotic_states() -> np.ndarray:
    """The chaotic-ish gate states of E1, shape ``(E1_N_CHAOTIC, 4)``."""
    return _chaotic_rest_states(tol.E1_STREAM_CHAOTIC, tol.E1_N_CHAOTIC)


@functools.cache
def e2_regular_states() -> np.ndarray:
    """The regular states of E2, shape ``(E2_N_REGULAR, 4)``."""
    return _regular_rest_states(tol.E2_STREAM_REGULAR, tol.E2_N_REGULAR)


@functools.cache
def e2_chaotic_states() -> np.ndarray:
    """The chaotic-ish states of E2, shape ``(E2_N_CHAOTIC, 4)``."""
    return _chaotic_rest_states(tol.E2_STREAM_CHAOTIC, tol.E2_N_CHAOTIC)


@functools.cache
def v_regular_states() -> np.ndarray:
    """The regular states of V1 and V2, shape ``(V_N_REGULAR, 4)``."""
    return _regular_rest_states(tol.V_STREAM_REGULAR, tol.V_N_REGULAR)


@functools.cache
def v_chaotic_states() -> np.ndarray:
    """The chaotic-ish states of V1 and V2, shape ``(V_N_CHAOTIC, 4)``."""
    return _chaotic_rest_states(tol.V_STREAM_CHAOTIC, tol.V_N_CHAOTIC)


@functools.cache
def moving_states() -> np.ndarray:
    """The moving states the script reports and no test gates, shape ``(N_MOVING, 4)``.

    One draw of angles, uniform in ``[-CHAOTIC_THETA_MAX, CHAOTIC_THETA_MAX]``, then one draw of
    velocities, uniform in ``[-MOVING_OMEGA_MAX, MOVING_OMEGA_MAX]`` (D-054).
    """
    rng = np.random.default_rng([tol.CASES_SEED, tol.MOVING_STREAM])
    theta = rng.uniform(-tol.CHAOTIC_THETA_MAX, tol.CHAOTIC_THETA_MAX, size=(tol.N_MOVING, 2))
    omega = rng.uniform(-tol.MOVING_OMEGA_MAX, tol.MOVING_OMEGA_MAX, size=(tol.N_MOVING, 2))
    return _frozen(np.hstack([theta, omega]))


# ---- energy (D-053 to D-055) ---------------------------------------------------------------

_energy_of_states = jax.jit(jax.vmap(dp.energy, in_axes=(0, None)))


def energies_along(rows: np.ndarray, params: dp.Params) -> np.ndarray:
    """``E`` (``double_pendulum.energy``) of every state in ``rows``, shape ``rows.shape[:-1]``."""
    states = np.asarray(rows, dtype=np.float64)
    flat = jnp.asarray(states.reshape(-1, states.shape[-1]))
    return np.asarray(_energy_of_states(flat, params)).reshape(states.shape[:-1])


def drift(energies: np.ndarray, params: dp.Params, *, up_to: int | None = None) -> np.ndarray:
    """``max over t <= up_to of |E(t) - E(0)| / E_s``, one value per trajectory (D-053).

    ``energies`` has shape ``(..., n_t)`` with outputs at the integer times ``0, 1, ...``. The
    scale ``E_s`` comes from ``params`` alone (``energy_scale``): not from the state, and not
    from the energies, so shifting every energy by a constant changes nothing.
    """
    e = np.asarray(energies, dtype=np.float64)
    stop = e.shape[-1] if up_to is None else up_to + 1
    return np.max(np.abs(e[..., :stop] - e[..., :1]), axis=-1) / chk.energy_scale(params)


def literal_ratio(energies: np.ndarray, params: dp.Params) -> float | None:
    """The literal ``max_t |E(t) - E(0)| / |E(0)|`` of one trajectory, for the report only.

    ``None`` when ``|E(0)| < E_LITERAL_FLOOR * E_s``: there the ratio would measure ``|E(0)|``
    and not the integrator (D-053).
    """
    e = np.asarray(energies, dtype=np.float64)
    if abs(e[0]) < tol.E_LITERAL_FLOOR * chk.energy_scale(params):
        return None
    return float(np.max(np.abs(e - e[0])) / abs(e[0]))


@dataclass(frozen=True)
class DriftMeasurement:
    """Everything the script prints about the drift of one trajectory."""

    e0: float
    """``E(0)``."""
    drift: float
    """The statistic of D-053 up to the horizon."""
    growth: tuple[float, ...]
    """The statistic up to each of ``E_GROWTH_TIMES``: the growth law."""
    literal: float | None
    """The literal ratio where ``|E(0)|`` is not near zero, else ``None``."""


def measure_drift(energies: np.ndarray, params: dp.Params) -> DriftMeasurement:
    """Measure one trajectory's energy drift."""
    return DriftMeasurement(
        e0=float(energies[0]),
        drift=float(drift(energies, params)),
        growth=tuple(float(drift(energies, params, up_to=t)) for t in tol.E_GROWTH_TIMES),
        literal=literal_ratio(energies, params),
    )


def damped_field(gamma: float) -> Field:
    """The pendulum field with a damping term ``-gamma * omega`` added (controls of E1 and V1)."""

    def field(y: jax.Array, params: dp.Params) -> jax.Array:
        return dp.vector_field(y, params) + jnp.array([0.0, 0.0, -gamma * y[2], -gamma * y[3]])

    return field


def dop853_energies(
    state: np.ndarray,
    params: dp.Params = DEFAULT,
    *,
    field: Field = dp.vector_field,
    rtol: float | None = None,
) -> np.ndarray:
    """``E`` along DOP853 at the integer times up to ``E_HORIZON``, shape ``(E_HORIZON + 1,)``.

    ``rtol`` sets ``rtol = atol`` when given; otherwise the defaults of ``solve_reference`` (the
    brief's ``1e-12``) apply.
    """
    times = np.arange(tol.E_HORIZON + 1, dtype=np.float64)
    options = {} if rtol is None else {"rtol": rtol, "atol": rtol}
    rows = integrators.solve_reference(field, state, params, times, **options)
    return energies_along(rows, params)


def rk4_energies(
    states: np.ndarray,
    params: dp.Params = DEFAULT,
    dt: float = tol.AGREE_DT,
    trajectory: chk.Trajectory | None = None,
) -> np.ndarray:
    """``E`` along RK4 at the integer times up to ``E_HORIZON``, shape ``(n, E_HORIZON + 1)``."""
    rows = chk.rk4_pendulum(states, params, dt, tol.E_HORIZON, chk.steps_per_unit(dt), trajectory)
    return energies_along(rows, params)


# ---- small-angle FFT (D-056) ---------------------------------------------------------------

F_SETS = {"F1": DEFAULT, "F2": dp.Params(*tol.F2_PARAMS)}
"""The two small-angle parameter sets of D-056: the defaults, and ``F2_PARAMS``."""

MODE_DIRECTIONS = {
    "mode 1": (1.0, math.sqrt(2.0)),
    "mode 2": (1.0, -math.sqrt(2.0)),
    "generic": (1.0, 0.0),
}
"""Initial angles in units of ``A``: the two eigenvectors for equal masses and lengths, and a
state with both modes (section 3.1 of the note)."""


def analytic_frequencies(g: float, length: float) -> tuple[float, float]:
    """``sqrt((g/L)(2 - sqrt 2))`` and ``sqrt((g/L)(2 + sqrt 2))``, written here from the brief.

    They are not taken from the code under test, which is the point of comparing with them.
    """
    ratio = g / length
    return math.sqrt(ratio * (2.0 - math.sqrt(2.0))), math.sqrt(ratio * (2.0 + math.sqrt(2.0)))


PEAK_COUNT = {"mode 1": 1, "mode 2": 1, "generic": 2}
"""How many peaks the spectrum of each initial state has: one per mode that it excites."""


def expected_peaks(mode: str, params: dp.Params) -> list[float]:
    """The analytic frequencies the peaks of ``mode`` must match, in increasing order.

    The brief's formula holds for equal masses and equal lengths only, so any other set is
    refused instead of being compared with a formula that does not apply to it (D-063).
    """
    if params.m1 != params.m2 or params.L1 != params.L2:
        raise ValueError("the formula (g/L)(2 -+ sqrt 2) needs m1 = m2 and L1 = L2")
    low, high = analytic_frequencies(params.g, params.L1)
    return {"mode 1": [low], "mode 2": [high], "generic": [low, high]}[mode]


def spectrum_peaks(series: np.ndarray, h: float, n_peaks: int) -> list[float]:
    """The ``n_peaks`` largest separated peaks of the spectrum of ``series``, as angular frequency.

    Remove the mean, multiply by a Hann window, zero-pad to a power of two of at least
    ``F_PAD`` times the length, take the magnitude, and pick the largest local maxima that are
    more than ``F_PEAK_SEPARATION_BINS`` padded bins from every earlier pick. Each is refined by
    a parabola through the logarithms of the magnitudes of the peak bin and its two neighbours
    (section 3.3 of the note). The expected frequencies are not used. Sorted increasingly.
    """
    n = len(series)
    windowed = (series - series.mean()) * np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * tol.F_PAD)))
    magnitude = np.abs(np.fft.rfft(windowed, nfft))
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=h)
    inner = magnitude[1:-1]
    maxima = np.flatnonzero((inner > magnitude[:-2]) & (inner >= magnitude[2:])) + 1
    chosen: list[int] = []
    for k in maxima[np.argsort(-magnitude[maxima])]:
        if all(abs(int(k) - j) > tol.F_PEAK_SEPARATION_BINS for j in chosen):
            chosen.append(int(k))
        if len(chosen) == n_peaks:
            break
    peaks = []
    for k in sorted(chosen):
        a, b, c = np.log(magnitude[k - 1]), np.log(magnitude[k]), np.log(magnitude[k + 1])
        offset = 0.5 * (a - c) / (a - 2.0 * b + c)
        peaks.append(float(omega[k] + offset * (omega[1] - omega[0])))
    return peaks


def fft_peaks(
    params: dp.Params, mode: str, integrator: str, *, amplitude: float = tol.F_AMPLITUDE
) -> list[float]:
    """Simulate ``theta1(t)`` from small angles of ``mode`` and return its measured peaks.

    ``integrator`` is ``"dop853"`` or ``"rk4"``. Sampling interval ``F_SAMPLE_DT t_s`` for
    ``F_N_SAMPLES`` samples; RK4 steps ``AGREE_DT t_s`` with ``F_RK4_STRIDE`` steps per sample.
    Frequencies are in radians per physical time unit.
    """
    t_s = math.sqrt(params.L1 / params.g)
    h = tol.F_SAMPLE_DT * t_s
    direction = MODE_DIRECTIONS[mode]
    y0 = np.array([amplitude * direction[0], amplitude * direction[1], 0.0, 0.0])
    if integrator == "dop853":
        rows = integrators.solve_reference(
            dp.vector_field, y0, params, h * np.arange(tol.F_N_SAMPLES)
        )
    elif integrator == "rk4":
        rows = np.asarray(
            integrators.rk4_trajectory(
                dp.vector_field,
                jnp.asarray(y0),
                params,
                tol.AGREE_DT * t_s,
                tol.F_N_SAMPLES - 1,
                tol.F_RK4_STRIDE,
            )
        )
    else:
        raise ValueError(f"unknown integrator {integrator!r}")
    return spectrum_peaks(rows[:, 0], h, PEAK_COUNT[mode])


def relative_errors(measured: list[float], expected: list[float]) -> list[float]:
    """``|measured - expected| / expected`` for each pair."""
    return [abs(m - e) / e for m, e in zip(measured, expected, strict=True)]


# ---- reversibility (D-057) -----------------------------------------------------------------


def dop853_round_trip_error(
    state: np.ndarray,
    horizon: float,
    params: dp.Params = DEFAULT,
    *,
    field: Field = dp.vector_field,
) -> float:
    """Forward ``horizon``, flip ``omega``, forward ``horizon``, flip back; distance from ``state``.

    The largest absolute difference over the four components, with DOP853 at its defaults.
    """
    times = np.array([float(horizon)])
    forward = integrators.solve_reference(field, state, params, times)[0]
    back = integrators.solve_reference(field, forward * FLIP, params, times)[0]
    return chk.max_abs_diff(back * FLIP, state)


def rk4_round_trip_errors(
    states: np.ndarray,
    horizon: float,
    params: dp.Params = DEFAULT,
    dt: float = tol.AGREE_DT,
    trajectory: chk.Trajectory | None = None,
) -> np.ndarray:
    """The same round trip with RK4 at ``dt``, one error per state, shape ``(n,)``."""
    steps = round(horizon / dt)
    forward = np.asarray(chk.rk4_pendulum(states, params, dt, 1, steps, trajectory))[:, -1]
    back = np.asarray(chk.rk4_pendulum(forward * FLIP, params, dt, 1, steps, trajectory))[:, -1]
    return np.max(np.abs(back * FLIP - states), axis=1)


def rk4_negative_dt_pair(
    states: np.ndarray, horizon: float, params: dp.Params = DEFAULT, dt: float = tol.AGREE_DT
) -> tuple[np.ndarray, np.ndarray]:
    """The two sides of V3: RK4 with ``-dt`` from ``y``, and ``S`` RK4 with ``dt`` from ``S y``.

    Equal, to rounding, for a field with ``f(S y) = -S f(y)`` (section 4.2 of the note).
    """
    steps = round(horizon / dt)
    negative = np.asarray(chk.rk4_pendulum(states, params, -dt, 1, steps))[:, -1]
    flipped = np.asarray(chk.rk4_pendulum(states * FLIP, params, dt, 1, steps))[:, -1] * FLIP
    return negative, flipped


# ---- summaries for the script --------------------------------------------------------------


def summary(values: np.ndarray) -> tuple[float, float, float]:
    """``(maximum, median, minimum)`` of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.max(v)), float(np.median(v)), float(np.min(v))
