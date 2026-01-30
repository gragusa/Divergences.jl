# Mathematical Theory

This section provides the mathematical foundation for divergence measures as implemented in Divergences.jl.

## Definition of Divergence

A divergence between vectors ``a \in \mathbb{R}^n`` and ``b \in \mathbb{R}^n`` is defined as

```math
D(a, b) = \sum_{i=1}^n \gamma(a_i / b_i) \, b_i
```

where ``\gamma: (a_\gamma, +\infty) \to \mathbb{R}_+`` is a strictly convex, twice continuously differentiable function.

### Normalization

The divergence function ``\gamma`` is normalized to satisfy:
- ``\gamma(1) = 0``
- ``\gamma'(1) = 0``
- ``\gamma''(1) = 1``

These normalizations ensure that:
- ``D(a, a) = 0`` (identity of indiscernibles)
- The divergence is locally quadratic around equal distributions

### Extended Real-Valued Functions

It is convenient to view ``\gamma`` as an extended real-valued function defined on ``\mathbb{R}`` taking values in ``[a_\gamma, +\infty]``. The convex function ``\gamma``, initially defined on ``(a_\gamma, +\infty)``, can be extended outside its domain by setting ``\gamma(u) = +\infty`` for all ``u \in (-\infty, a_\gamma)``.

For the boundary value, we let ``\gamma(a_\gamma) = \lim_{u \to a_\gamma^+} \gamma(u)``, knowing that this limit is possibly ``\infty``. This ensures that the extension of ``\gamma`` is lower-semicontinuous on ``\mathbb{R}``.

## Gradient and Hessian

The gradient of the divergence with respect to ``a`` is:

```math
\nabla_a D(a, b) = \left( \gamma'(a_1/b_1), \ldots, \gamma'(a_n/b_n) \right)
```

The Hessian matrix is diagonal:

```math
\nabla_a^2 D(a, b) = \text{diag}\left( \frac{\gamma''(a_1/b_1)}{b_1}, \ldots, \frac{\gamma''(a_n/b_n)}{b_n} \right)
```

Given the normalization ``\gamma'(1) = 0`` and ``\gamma''(1) = 1``, we have:

```math
\nabla_a D(a, a) = 0, \quad \nabla_a^2 D(a, a) = \text{diag}(1/a_1, \ldots, 1/a_n)
```

## Conjugate Function

The conjugate (or Legendre-Fenchel transform) of ``\gamma`` is defined as:

```math
\psi(v) = \gamma^*(v) = \sup_{u \in \mathbb{R}} \left\{ u v - \gamma(u) \right\}
```

The conjugate of a convex extended real-valued function is itself a convex lower semi-continuous function. Key properties:

- ``\gamma^*`` is increasing on ``\mathbb{R}``
- The domain of ``\gamma^*`` depends on the asymptotic behavior of ``\gamma``
- For many divergences, ``(\gamma^*)'(v) = (\gamma')^{-1}(v)``

Define:
```math
d = \lim_{u \to +\infty} \gamma(u)/u
```

Then the effective domain of ``\gamma^*`` is ``\overline{\text{dom}\,\gamma^*} = (-\infty, d)``.

## Modified Divergences

For many divergences, the effective domain of their conjugate ``\gamma^*`` does not span ``\mathbb{R}`` since ``\gamma(u)/u \to l < +\infty`` as ``u \to +\infty``. This can be problematic in optimization applications.

### Upper Modification

For some ``\vartheta > 0``, let ``u_\vartheta \equiv 1 + \vartheta``. The **modified divergence** ``\gamma_\vartheta`` is defined as:

```math
\gamma_\vartheta(u) = \begin{cases}
  \gamma(u_\vartheta) + \gamma'(u_\vartheta)(u-u_\vartheta) + \frac{1}{2}\gamma''(u_\vartheta)(u-u_\vartheta)^2, & u \geq u_\vartheta \\
  \gamma(u), & u \in (a_\gamma, u_\vartheta) \\
  \lim_{u \to 0^+} \gamma(u), & u = 0 \\
  +\infty, & u < 0
\end{cases}
```

This modified divergence still satisfies all requirements and normalization of ``\gamma``. Furthermore:

```math
\lim_{u \to \infty} \frac{\gamma_\vartheta(u)}{u} = +\infty, \quad \text{and} \quad \lim_{u \to \infty} \frac{u \gamma'_\vartheta(u)}{\gamma_\vartheta(u)} = 2
```

The first limit implies that the image of ``\gamma'_\vartheta`` is the real line and thus ``\overline{\text{dom}\,\gamma^*_\vartheta} = (-\infty, +\infty)``.

### Conjugate of Modified Divergence

The conjugate is obtained by applying the Legendre-Fenchel transform:

```math
\gamma_\vartheta^*(v) = \begin{cases}
  a_\vartheta v^2 + b_\vartheta v + c_\vartheta, & v > \gamma'(u_\vartheta) \\
  \gamma^*(v), & v \leq \gamma'(u_\vartheta)
\end{cases}
```

where:

- ``a_\vartheta = 1/(2\gamma''(u_\vartheta))``
- ``b_\vartheta = u_\vartheta - 2 a_\vartheta \gamma'(u_\vartheta)``
- ``c_\vartheta = -\gamma(u_\vartheta) + a_\vartheta (\gamma'(u_\vartheta))^2``

## Fully Modified Divergences

For some ``\vartheta > 0`` and ``0 < \varphi < 1 - a_\gamma``, the **fully modified divergence** extends the domain on both ends:

```math
\gamma_{\varphi, \vartheta}(u) = \begin{cases}
  \gamma(u_\vartheta) + \gamma'(u_\vartheta)(u-u_\vartheta) + \frac{1}{2}\gamma''(u_\vartheta)(u-u_\vartheta)^2, & u \geq u_\vartheta \\
  \gamma(u), & u \in (u_\varphi, u_\vartheta) \\
  \gamma(u_\varphi) + \gamma'(u_\varphi)(u-u_\varphi) + \frac{1}{2}\gamma''(u_\varphi)(u-u_\varphi)^2, & u \leq u_\varphi
\end{cases}
```

where ``u_\vartheta = 1 + \vartheta`` and ``u_\varphi = a_\gamma + \varphi``.

This divergence satisfies all requirements of ``\gamma`` while being defined on all of ``\mathbb{R}``.

