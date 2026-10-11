"""Tests T1 to T11 of the owner-written double-pendulum mechanics (D-030 to D-032).

They stay red until the owner implements ``horizon.systems.double_pendulum``. Each test names
its tolerance class from ``horizon.tolerances`` and says in its docstring what it catches and
what it is blind to. Every comparison goes through ``check_close``, the single rule
``|x - x_ref| <= ATOL + RTOL * s``, so no test can quietly use a looser one.

The functions are called on one state at a time, eagerly, as in the contract of D-028.
``jit`` and ``vmap`` are exercised separately, in T11, so that a function that breaks only under
them fails one test with a clear message instead of failing everything.

Which tests rest on the oracle: T1 to T4 and T8 compare with it, so they are exactly as
trustworthy as it is (anchored by O1, O3 and O4). T5, T6, T7, T9, T10 and T11 check identities
stated in ``docs/derivations/double_pendulum.md`` and hold whatever the oracle says.
"""

import inspect
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from horizon import tolerances as tol
from horizon.systems import double_pendulum as dp
from horizon.tolerances import check_close
from tests import cases
from tests.oracles import double_pendulum as oracle

MODERATE = cases.moderate_cases()
LARGE_ANGLE = cases.large_angle_cases()

EQUILIBRIA = tuple(itertools.product((0.0, math.pi), repeat=2))
"""The four equilibria ``(theta1, theta2)`` with ``theta_i`` in ``{0, pi}`` (at rest)."""

CONTRACT = (
    ("mass_matrix", "theta", (2, 2)),
    ("kinetic_energy", "state", ()),
    ("potential_energy", "state", ()),
    ("energy", "state", ()),
    ("acceleration", "state", (2,)),
    ("vector_field", "state", (4,)),
)
"""Each function, whether its first argument is ``theta`` or the state, and its output shape."""


# ---- helpers ------------------------------------------------------------------------------


def _case_id(case: cases.Case) -> str:
    return f"set-{case.index:02d}"


def _large_id(case: cases.Case) -> str:
    return f"large-angle-{case.index}"


def _params(case: cases.Case) -> dp.Params:
    return dp.Params(*case.params)


def _arr(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _impl(x: object, x_ref: object, what: str, scale: object = None) -> None:
    """Compare with the oracle: class IMPL."""
    check_close(_arr(x), x_ref, rtol=tol.IMPL_RTOL, atol=tol.IMPL_ATOL, scale=scale, what=what)


def _identity(x: object, x_ref: object, what: str, scale: object = None) -> None:
    """Compare against an exact identity: class IDENTITY."""
    check_close(
        _arr(x), x_ref, rtol=tol.IDENTITY_RTOL, atol=tol.IDENTITY_ATOL, scale=scale, what=what
    )


def _acceleration(state: np.ndarray, params: dp.Params) -> np.ndarray:
    return _arr(dp.acceleration(jnp.asarray(state), params))


# ---- T1: mass matrix ----------------------------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t1_mass_matrix_matches_the_oracle(case: cases.Case) -> None:
    """T1 (class IMPL): ``mass_matrix`` equals the oracle's ``M(theta)``.

    Catches a wrong entry, a wrong coupling angle (``cos(theta1 + theta2)``), a swapped
    mass or length, and a missing factor. Blind to mistakes that live only in the forcing
    ``b`` or in the energies, and to anything smaller than the IMPL tolerance.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        theta = state[:2]
        _impl(
            dp.mass_matrix(jnp.asarray(theta), params),
            oracle.mass_matrix(theta, case.params),
            f"mass_matrix, set {case.index}, state {k}",
        )


# ---- T2: energies -------------------------------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t2_kinetic_energy_matches_the_oracle(case: cases.Case) -> None:
    """T2 (class IMPL): ``kinetic_energy`` equals the oracle's Cartesian ``T``.

    Catches a missing 1/2, a wrong cross term or coupling angle, and swapped parameters.
    Blind to mistakes in the potential or in the accelerations.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        _impl(
            dp.kinetic_energy(jnp.asarray(state), params),
            oracle.kinetic_energy(state, case.params),
            f"kinetic_energy, set {case.index}, state {k}",
        )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t2_potential_energy_matches_the_oracle(case: cases.Case) -> None:
    """T2 (class IMPL): ``potential_energy`` equals the oracle's Cartesian ``V``.

    Catches a sign error, a wrong reference height (the zero is the pivot) and swapped
    parameters. Blind to mistakes in the kinetic energy or in the accelerations.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        _impl(
            dp.potential_energy(jnp.asarray(state), params),
            oracle.potential_energy(state, case.params),
            f"potential_energy, set {case.index}, state {k}",
        )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t2_energy_matches_the_oracle(case: cases.Case) -> None:
    """T2 (class IMPL): ``energy`` equals the oracle's ``T + V`` (summed at high precision).

    Catches ``energy`` not being the sum of the two parts. Blind to the same things as the
    two tests above.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        _impl(
            dp.energy(jnp.asarray(state), params),
            oracle.energy(state, case.params),
            f"energy, set {case.index}, state {k}",
        )


# ---- T3: accelerations --------------------------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t3_acceleration_matches_the_oracle(case: cases.Case) -> None:
    """T3 (class IMPL): ``acceleration`` equals the oracle's solved accelerations.

    The complete check of the equations of motion at generic states, with distinct non-default
    parameters, so a sign, a swapped parameter, a missing term or the wrong coupling all show
    up. Blind to nothing in the dynamics, but only as trustworthy as the oracle (O1, O3, O4).
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        _impl(
            _acceleration(state, params),
            oracle.acceleration(state, case.params),
            f"acceleration, set {case.index}, state {k}",
        )


@pytest.mark.parametrize("case", LARGE_ANGLE, ids=_large_id)
def test_t3_acceleration_matches_the_oracle_at_large_unwrapped_angles(case: cases.Case) -> None:
    """T3 (class IMPL): the same comparison at angles up to 1e3 radians (D-032).

    The exact function is ``2 pi``-periodic, so agreement with it at large unwrapped angles
    implies periodicity to the IMPL tolerance, and also correctness there. Catches wrapping
    done wrongly and loss of precision in the angle arguments. Blind to the same things as
    the test above.
    """
    params = _params(case)
    state = case.states[0]
    _impl(
        _acceleration(state, params),
        oracle.acceleration(state, case.params),
        f"acceleration at large angles, state {case.index}",
    )


# ---- T4: Euler-Lagrange residual (the gate's residual check) ------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t4_euler_lagrange_residual_of_the_acceleration_vanishes(case: cases.Case) -> None:
    """T4 (class IMPL, scale = sum of the monomials): the residual check of the gate.

    The owner's float64 accelerations are substituted into the Euler-Lagrange equations
    derived independently by SymPy from the geometry, and the residual is evaluated at 60
    digits. It must vanish to rounding error relative to the size of the terms that cancel
    (the "natural scale"), which is a backward-error check: it does not amplify rounding the
    way a comparison of ``alpha`` itself can. Blind to a wrong convention shared with the
    oracle (that is what O1, O3 and O4 guard).
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        residual, scale = oracle.el_residual(state, _acceleration(state, params), case.params)
        _impl(
            residual,
            np.zeros(2),
            f"Euler-Lagrange residual, set {case.index}, state {k}",
            scale=scale,
        )


# ---- T5: energy is stationary along the flow ----------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t5_energy_is_stationary_along_the_vector_field(case: cases.Case) -> None:
    """T5 (class IMPL, scale = sum of the four terms): ``grad(E) . f = 0``, with no integration.

    ``jax.grad`` of the owner's ``energy`` is paired with the owner's ``vector_field``.
    Catches any inconsistency between them (a sign error in one Coriolis term, a missing
    term). Blind to a mistake made consistently in both, for example equations derived from
    the same wrong energy: only T1 to T4 and T8 catch those.
    """
    params = _params(case)
    grad_energy = jax.grad(dp.energy)
    for k, state in enumerate(case.states):
        s = jnp.asarray(state)
        terms = _arr(grad_energy(s, params)) * _arr(dp.vector_field(s, params))
        where = f"set {case.index}, state {k}"
        assert np.all(np.isfinite(terms)), f"non-finite term in grad(E).f, {where}"
        _impl(np.sum(terms), 0.0, f"grad(E).f, {where}", scale=np.sum(np.abs(terms)))


# ---- T6: symmetries -----------------------------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t6_acceleration_is_even_in_omega_and_odd_in_theta(case: cases.Case) -> None:
    """T6 (class IDENTITY): ``alpha`` is even in ``omega`` and odd in ``theta``.

    That is ``alpha(theta, -omega) = alpha(theta, omega)`` and
    ``alpha(-theta, omega) = -alpha(theta, omega)`` (time reversal and reflection).
    Catches a velocity term that is not quadratic (``omega`` for ``omega**2``) and a gravity
    or coupling term with the wrong parity (``sin`` for ``cos``). Blind to parity-preserving
    mistakes: ``cos(theta1 + theta2)`` for ``cos(theta1 - theta2)``, wrong constants, and
    swapped parameters.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        theta, omega = state[:2], state[2:]
        base = _acceleration(state, params)
        where = f"set {case.index}, state {k}"
        _identity(
            _acceleration(np.concatenate([theta, -omega]), params),
            base,
            f"acceleration with omega reversed, {where}",
        )
        _identity(
            _acceleration(np.concatenate([-theta, omega]), params),
            -base,
            f"acceleration with theta reflected, {where}",
        )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t6_vector_field_is_2pi_periodic_in_each_angle(case: cases.Case) -> None:
    """T6 (class IDENTITY): shifting either angle by ``2 pi k`` (``k = -1, +1``) changes nothing.

    The shift is done in float64. Catches a vector field that uses an angle other than
    through ``sin`` and ``cos`` (for example ``theta`` itself, or a wrap with the wrong
    period). Moderate angles only: at ``|theta|`` near 1e3 the rounding of ``theta1 - theta2``
    would make this identity tolerance unfair, so large angles are covered by T3 (D-032).
    Blind to errors that are themselves periodic, such as the wrong coupling angle.
    """
    params = _params(case)
    for k, state in enumerate(case.states):
        base = _arr(dp.vector_field(jnp.asarray(state), params))
        for angle, shift in itertools.product((0, 1), tol.PERIODICITY_SHIFTS):
            shifted = np.array(state)
            shifted[angle] = state[angle] + shift * 2.0 * math.pi
            _identity(
                dp.vector_field(jnp.asarray(shifted), params),
                base,
                f"vector_field with theta{angle + 1} shifted by {shift} * 2 pi, "
                f"set {case.index}, state {k}",
            )


# ---- T7: equilibria and the energy floor --------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t7_the_four_equilibria_have_zero_vector_field(case: cases.Case) -> None:
    """T7 (class IDENTITY): ``f = 0`` at ``theta_i`` in ``{0, pi}`` with ``omega = 0``.

    Catches a flipped gravity sign or ``sin`` and ``cos`` confused in the gravity terms,
    either of which moves the equilibria. Blind to ``cos(theta1 + theta2)`` in place of
    ``cos(theta1 - theta2)``: every equilibrium angle is a multiple of ``pi``, where the two
    coincide. Also blind to any error in terms that vanish at rest.
    """
    params = _params(case)
    for theta1, theta2 in EQUILIBRIA:
        state = jnp.array([theta1, theta2, 0.0, 0.0])
        _identity(
            dp.vector_field(state, params),
            np.zeros(4),
            f"vector_field at the equilibrium ({theta1:.4f}, {theta2:.4f}), set {case.index}",
        )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t7_hanging_energy_is_minus_the_gravity_scale(case: cases.Case) -> None:
    """T7 (class IDENTITY): ``E(hanging, at rest) = -[(m1 + m2) g L1 + m2 g L2]``.

    Catches a wrong sign or reference height of the potential, and parameter mistakes in the
    gravity terms. Blind to the kinetic energy (it is zero at rest) and to the coupling angle.
    """
    m1, m2, L1, L2, g = case.params
    _identity(
        dp.energy(jnp.zeros(4), _params(case)),
        -((m1 + m2) * g * L1 + m2 * g * L2),
        f"energy at the hanging state, set {case.index}",
    )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t7_energy_is_bounded_below_by_the_hanging_energy(case: cases.Case) -> None:
    """T7 (class IDENTITY, one-sided): ``E >= -[(m1 + m2) g L1 + m2 g L2]`` at every state.

    ``V`` is at its minimum when both rods hang and ``T >= 0`` because ``M`` is positive
    definite. Catches a kinetic energy that can go negative (a sign error in the cross term
    large enough to break positive definiteness) and a potential below its true minimum.
    Blind to errors that keep ``E`` above the floor.
    """
    m1, m2, L1, L2, g = case.params
    floor = -((m1 + m2) * g * L1 + m2 * g * L2)
    slack = tol.IDENTITY_ATOL + tol.IDENTITY_RTOL * max(abs(floor), 1.0)
    params = _params(case)
    for k, state in enumerate(case.states):
        e = float(dp.energy(jnp.asarray(state), params))
        assert e >= floor - slack, (
            f"energy {e!r} is below the hanging energy {floor!r} (slack {slack:.1e}), "
            f"set {case.index}, state {k}"
        )


# ---- T8: Jacobian by autodiff -------------------------------------------------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t8_autodiff_jacobian_matches_the_oracle(case: cases.Case) -> None:
    """T8 (class IMPL): ``jax.jacfwd(vector_field)`` equals the oracle's SymPy Jacobian.

    Milestone 5 needs autodiff Jacobians for the tangent-space method. Catches a function
    that is not differentiable (``stop_gradient``, NumPy calls, value-dependent branches) and
    an error in the equations that shows in the derivatives. Blind to what T3 already misses.
    """
    params = _params(case)
    jacobian = jax.jacfwd(dp.vector_field)
    for k, state in enumerate(case.states):
        _impl(
            jacobian(jnp.asarray(state), params),
            oracle.jacobian(state, case.params),
            f"jacfwd(vector_field), set {case.index}, state {k}",
        )


# ---- T9: the gate: normal modes at the hanging state --------------------------------------


def _hanging_jacobian() -> np.ndarray:
    return _arr(jax.jacfwd(dp.vector_field)(jnp.zeros(4), dp.DEFAULT_PARAMS))


def test_t9_hanging_jacobian_has_the_expected_blocks() -> None:
    """T9 (class IDENTITY): the Jacobian at the hanging state is ``[[0, I], [-M(0)^-1 K, 0]]``.

    Checks the top-left block (zero), the top-right block (the identity) and the bottom-right
    block (zero, because ``alpha`` is quadratic in ``omega``) directly, so a wrong block cannot
    hide behind correct eigenvalues. Uses the default parameters only, so it cannot catch a
    swapped parameter, and it is blind to ``cos(theta1 + theta2)``.
    """
    jacobian = _hanging_jacobian()
    _identity(jacobian[:2, :2], np.zeros((2, 2)), "J[:2, :2]")
    _identity(jacobian[:2, 2:], np.eye(2), "J[:2, 2:]")
    _identity(jacobian[2:, 2:], np.zeros((2, 2)), "J[2:, 2:]")


def test_t9_stiffness_block_eigenvalues_are_two_minus_and_plus_sqrt_two() -> None:
    """T9 (class IDENTITY): the eigenvalues of ``-J[2:, :2] = M(0)^-1 K`` are ``2 -+ sqrt(2)``.

    That matrix is not symmetric, so the test uses ``numpy.linalg.eigvals`` (a symmetric
    solver would silently read one triangle), asserts the imaginary parts are negligible and
    compares the sorted real parts. Same blind spots as the block test.
    """
    eigenvalues = np.linalg.eigvals(-_hanging_jacobian()[2:, :2])
    largest_imaginary = float(np.max(np.abs(eigenvalues.imag)))
    assert largest_imaginary <= tol.IDENTITY_ATOL, f"eigenvalues are not real: {eigenvalues}"
    _identity(
        np.sort(eigenvalues.real),
        np.array([2.0 - math.sqrt(2.0), 2.0 + math.sqrt(2.0)]),
        "eigenvalues of M(0)^-1 K",
    )


def test_t9_hanging_jacobian_eigenvalues_are_plus_minus_i_sqrt_two_plus_minus_sqrt_two() -> None:
    """T9 (class IDENTITY): the 4x4 eigenvalues are ``+-i sqrt(2 +- sqrt(2))``.

    The real parts must vanish (a Hamiltonian system at a stable equilibrium) and the sorted
    imaginary parts must match. Secondary to the two tests above, which are sharper.
    """
    eigenvalues = np.linalg.eigvals(_hanging_jacobian())
    largest_real = float(np.max(np.abs(eigenvalues.real)))
    assert largest_real <= tol.IDENTITY_ATOL, f"eigenvalues are not imaginary: {eigenvalues}"
    fast, slow = math.sqrt(2.0 + math.sqrt(2.0)), math.sqrt(2.0 - math.sqrt(2.0))
    _identity(
        np.sort(eigenvalues.imag),
        np.array([-fast, -slow, slow, fast]),
        "imaginary parts of the Jacobian eigenvalues",
    )


# ---- T10: scaling invariances (what the nondimensionalization claims) ---------------------


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t10_scaling_both_masses_keeps_acceleration_and_scales_energy_linearly(
    case: cases.Case,
) -> None:
    """T10 (class IDENTITY): scaling both masses by ``c`` keeps ``alpha`` and scales ``E`` by ``c``.

    Factors 0.5 and 3.0. Catches a mass that enters ``alpha`` or ``E`` with the wrong power or
    an additive constant. Blind to a swap of ``m1`` and ``m2`` in one term, and to the
    coupling angle, since those preserve homogeneity.
    """
    params = _params(case)
    for factor in tol.SCALE_FACTORS:
        scaled = params._replace(m1=factor * params.m1, m2=factor * params.m2)
        for k, state in enumerate(case.states):
            where = f"c = {factor}, set {case.index}, state {k}"
            _identity(
                _acceleration(state, scaled),
                _acceleration(state, params),
                f"acceleration, masses scaled, {where}",
            )
            _identity(
                dp.energy(jnp.asarray(state), scaled),
                factor * _arr(dp.energy(jnp.asarray(state), params)),
                f"energy, masses scaled, {where}",
            )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t10_scaling_lengths_and_gravity_keeps_acceleration_and_scales_energy_by_c_squared(
    case: cases.Case,
) -> None:
    """T10 (class IDENTITY): scaling ``L1, L2, g`` by ``c`` keeps ``alpha``, ``E`` goes by ``c^2``.

    Factors 0.5 and 3.0. ``g / L`` is unchanged, so ``alpha`` is; ``T`` goes as ``L**2`` and
    ``V`` as ``g L``. Catches a length or ``g`` with the wrong power. Blind to a swap of
    ``L1`` and ``L2`` in one term and to the coupling angle.
    """
    params = _params(case)
    for factor in tol.SCALE_FACTORS:
        scaled = params._replace(L1=factor * params.L1, L2=factor * params.L2, g=factor * params.g)
        for k, state in enumerate(case.states):
            where = f"c = {factor}, set {case.index}, state {k}"
            _identity(
                _acceleration(state, scaled),
                _acceleration(state, params),
                f"acceleration, lengths and g scaled, {where}",
            )
            _identity(
                dp.energy(jnp.asarray(state), scaled),
                factor**2 * _arr(dp.energy(jnp.asarray(state), params)),
                f"energy, lengths and g scaled, {where}",
            )


@pytest.mark.parametrize("case", MODERATE, ids=_case_id)
def test_t10_gravity_time_scaling_multiplies_acceleration_by_k(case: cases.Case) -> None:
    """T10 (class IDENTITY): ``alpha(theta, sqrt(k) omega; k g) = k alpha(theta, omega; g)``.

    This is the statement that time is measured in units of ``sqrt(L1 / g)``: without a test
    behind it the document's nondimensionalization would be unchecked. Factors 0.5 and 3.0.
    Catches ``g`` entering the velocity terms or ``omega`` entering gravity terms. Blind to the
    coupling angle and to swaps that preserve the scaling.
    """
    params = _params(case)
    for factor in tol.SCALE_FACTORS:
        scaled = params._replace(g=factor * params.g)
        for k, state in enumerate(case.states):
            fast = np.concatenate([state[:2], math.sqrt(factor) * state[2:]])
            _identity(
                _acceleration(fast, scaled),
                factor * _acceleration(state, params),
                f"acceleration, time rescaled, k = {factor}, set {case.index}, state {k}",
            )


# ---- T11: contract: signature, shapes, dtype, jit and vmap --------------------------------


def _first_argument(kind: str, states: np.ndarray) -> np.ndarray:
    return states[..., :2] if kind == "theta" else states


def _params_from(row: np.ndarray) -> dp.Params:
    return dp.Params(*(float(v) for v in row))


@pytest.mark.parametrize("name", [entry[0] for entry in CONTRACT])
def test_t11_params_is_a_required_argument(name: str) -> None:
    """T11: every function takes ``params`` with no default (D-028).

    A default would let a test or an experiment run on the all-ones parameters, which hide
    swapped-parameter bugs. Checked from the signature, so no call is needed. Blind to
    everything numerical.
    """
    parameters = inspect.signature(getattr(dp, name)).parameters
    assert "params" in parameters, f"{name} must take an argument named 'params'"
    assert parameters["params"].default is inspect.Parameter.empty, (
        f"{name}: 'params' must be required"
    )


def test_t11_params_fields_and_defaults() -> None:
    """T11: ``Params`` is ``(m1, m2, L1, L2, g)`` and the defaults are all ones (D-028)."""
    assert dp.Params._fields == ("m1", "m2", "L1", "L2", "g")
    assert dp.Params(1.0, 1.0, 1.0, 1.0, 1.0) == dp.DEFAULT_PARAMS


@pytest.mark.parametrize(("name", "kind", "shape"), CONTRACT, ids=[e[0] for e in CONTRACT])
def test_t11_output_has_the_documented_shape_and_is_float64(
    name: str, kind: str, shape: tuple[int, ...]
) -> None:
    """T11: each function returns a ``jax.Array`` of the documented shape and ``float64``.

    Catches float32 truncation and a wrong shape (for example a ``(1,)`` where a scalar is
    documented). Blind to everything numerical.
    """
    case = MODERATE[0]
    first = jnp.asarray(_first_argument(kind, case.states[0]))
    out = getattr(dp, name)(first, _params(case))
    assert isinstance(out, jax.Array), f"{name} returned {type(out).__name__}, not a jax.Array"
    assert out.shape == shape, f"{name} has shape {out.shape}, expected {shape}"
    assert out.dtype == jnp.float64, f"{name} has dtype {out.dtype}, expected float64"


@pytest.mark.parametrize(
    ("name", "kind"), [(e[0], e[1]) for e in CONTRACT], ids=[e[0] for e in CONTRACT]
)
def test_t11_jit_and_vmap_with_traced_params_equal_the_per_case_result(
    name: str, kind: str
) -> None:
    """T11 (class IDENTITY): ``jit`` + ``vmap`` agree with the plain per-case calls.

    Two forms, both with ``params`` passed as a traced argument and never closed over or
    static: (A) all 500 states with 500 different ``Params`` batched together; (B) the states
    of one set batched with that set's shared ``Params``, one compiled function reused for all
    twenty sets. Catches Python branching on a value of ``params`` or ``state``, which an
    eager call cannot, and results that depend on the execution mode. Blind to numerical
    correctness: it only checks consistency with the eager results.
    """
    function = getattr(dp, name)
    states = np.concatenate([c.states for c in MODERATE])
    rows = np.concatenate([np.tile(c.params, (len(c.states), 1)) for c in MODERATE])
    inputs = _first_argument(kind, states)
    per_case = np.stack(
        [_arr(function(jnp.asarray(inputs[i]), _params_from(rows[i]))) for i in range(len(inputs))]
    )

    batched_params = dp.Params(*(jnp.asarray(rows[:, j]) for j in range(5)))
    form_a = _arr(jax.jit(jax.vmap(function))(jnp.asarray(inputs), batched_params))
    assert form_a.shape == per_case.shape
    for i in range(len(inputs)):
        _identity(form_a[i], per_case[i], f"{name}, jit+vmap with batched Params, case {i}")

    run = jax.jit(jax.vmap(function, in_axes=(0, None)))
    for c in MODERATE:
        first = c.index * len(c.states)
        form_b = _arr(run(jnp.asarray(_first_argument(kind, c.states)), _params(c)))
        for k in range(len(c.states)):
            _identity(
                form_b[k],
                per_case[first + k],
                f"{name}, jit+vmap with shared traced Params, set {c.index}, state {k}",
            )
