"""Tripwire for the milestone 4 register (K1), its seeded-case guards (C) and helper tests (K2).

K1 pins the declared values in ``src/horizon/tolerances.py`` (D-052 to D-060). The claims and the
tests that check them are tabulated in ``docs/derivations/physics_validation.md``, section 6.
"""

from horizon import tolerances as tol

# ---- K1: the register ---------------------------------------------------------------------


def test_register_holds_the_declared_physics_values() -> None:
    """K1 (class exact). Tripwire: these literals are the values logged in D-052 to D-060.

    Catches: an edit to the milestone 4 section of ``src/horizon/tolerances.py`` without a
    matching edit here, so every change shows up in two places in the diff. Changing a value
    needs Opus 5.5's approval first (D-025). Blind to: a declared value that is itself wrong. That
    is what the review in the log is for.
    """
    # D-054: E1, the DOP853 energy-drift gate, and what is reported beside it.
    assert (tol.E_HORIZON, tol.E_DOP_CEILING) == (1000, 1e-9)
    assert (tol.E1_STREAM_REGULAR, tol.E1_N_REGULAR) == (3000, 2)
    assert (tol.E1_STREAM_CHAOTIC, tol.E1_N_CHAOTIC) == (3001, 5)
    assert tol.E_TIMEOUT_S == 300
    assert tol.E_GROWTH_TIMES == (100, 300, 1000)
    assert tol.E_LITERAL_FLOOR == 0.1
    assert (tol.MOVING_STREAM, tol.N_MOVING, tol.MOVING_OMEGA_MAX) == (3006, 8, 2.0)

    # D-055: E2, the RK4 energy-drift gate.
    assert (tol.E2_STREAM_REGULAR, tol.E2_N_REGULAR, tol.E2_CEILING_REGULAR) == (3002, 12, 1e-7)
    assert (tol.E2_STREAM_CHAOTIC, tol.E2_N_CHAOTIC, tol.E2_CEILING_CHAOTIC) == (3003, 30, 5e-4)

    # D-056: F, the small-angle FFT.
    assert (tol.F_AMPLITUDE, tol.F_SAMPLE_DT, tol.F_N_SAMPLES) == (1e-3, 0.05, 20000)
    assert (tol.F_RK4_STRIDE, tol.F_PAD, tol.F_PEAK_SEPARATION_BINS) == (5, 8, 40)
    assert tol.F_TOL == 1e-5
    assert tol.F2_PARAMS == (1.3, 1.3, 0.7, 0.7, 9.81)

    # D-057: V, time reversibility.
    assert (tol.V_REGULAR_T, tol.V_CHAOTIC_T) == (100, 8)
    assert (tol.V_STREAM_REGULAR, tol.V_N_REGULAR) == (3004, 12)
    assert (tol.V_STREAM_CHAOTIC, tol.V_N_CHAOTIC) == (3005, 30)
    assert (tol.V1_CEILING_REGULAR, tol.V1_CEILING_CHAOTIC) == (5e-9, 5e-8)
    assert (tol.V2_CEILING_REGULAR, tol.V2_CEILING_CHAOTIC) == (1e-6, 2e-3)

    # D-058: the controls.
    assert (tol.E_CONTROL_RTOL, tol.E_CONTROL_N_STATES) == (1e-10, 2)
    assert (tol.E_CONTROL_DAMPING, tol.E_CONTROL_RK4_DT) == (1e-6, 0.05)
    assert tol.F_CONTROL_M2_FACTOR == 1.01


def test_register_streams_are_distinct_and_disjoint_from_the_earlier_ones() -> None:
    """K1 (class exact): streams 3000 to 3006 are distinct and do not reuse an earlier stream.

    The earlier ones are the parameter-set streams ``k = 0`` to ``19``, 1000 (D-032) and 2000 to
    2003 (D-044 to D-047), as Opus 5.5 checked (D-059). Catches a copy-paste of a stream number,
    which would make two gates draw the same states. Blind to the content of the streams.
    """
    streams = (
        tol.E1_STREAM_REGULAR,
        tol.E1_STREAM_CHAOTIC,
        tol.E2_STREAM_REGULAR,
        tol.E2_STREAM_CHAOTIC,
        tol.V_STREAM_REGULAR,
        tol.V_STREAM_CHAOTIC,
        tol.MOVING_STREAM,
    )
    assert sorted(streams) == list(range(3000, 3007))
    earlier = {*range(20), 1000, 2000, 2001, 2002, 2003}
    assert earlier.isdisjoint(streams)


def test_register_sampling_of_the_fft_record_is_consistent() -> None:
    """K1 (class exact): the stride times the RK4 step is the sampling interval, and the horizon.

    RK4 outputs every ``F_RK4_STRIDE`` steps of ``AGREE_DT``, so the two integrators must sample
    at the same times; the record spans ``T_total = 1000 t_s``. The ``E_GROWTH_TIMES`` are
    increasing and end at the horizon. Catches an edit of one of these numbers that leaves the
    others behind. Blind to whether the numbers are good ones.
    """
    assert abs(tol.F_RK4_STRIDE * tol.AGREE_DT - tol.F_SAMPLE_DT) < 1e-15
    assert abs(tol.F_N_SAMPLES * tol.F_SAMPLE_DT - 1000.0) < 1e-9
    assert list(tol.E_GROWTH_TIMES) == sorted(tol.E_GROWTH_TIMES)
    assert tol.E_GROWTH_TIMES[-1] == tol.E_HORIZON
