"""Predictability Horizon: learned dynamics measured in units of Lyapunov time.

Importing this package enables 64-bit floats in JAX (``jax_enable_x64``). Every
numerical result in this repository depends on it: the reference integrator runs
at tolerances near 1e-12, far below float32 resolution (about 1.2e-7).

Import ``horizon`` before creating any JAX array. Arrays created earlier stay
float32. See ``docs/DECISIONS.md`` (D-015).
"""

import jax

jax.config.update("jax_enable_x64", True)
