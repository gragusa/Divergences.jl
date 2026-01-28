# Divergence Types

This section lists all divergence types implemented in Divergences.jl along with their mathematical formulas.

## Basic Divergences

### Kullback-Leibler Divergence

**Type:** [`KullbackLeibler`](@ref)

**Mathematical Definition:**

| Function | Formula |
|:---------|:--------|
| ``\gamma(u)`` | ``u \log(u) - u + 1`` |
| ``\gamma'(u)`` | ``\log(u)`` |
| ``\gamma''(u)`` | ``1/u`` |
| ``\psi(v)`` | ``e^v - 1`` |
| ``\psi'(v)`` | ``e^v`` |
| Domain of ``\gamma`` | ``u > 0`` |
| Domain of ``\psi`` | ``v \in \mathbb{R}`` |

### Reverse Kullback-Leibler Divergence

**Type:** [`ReverseKullbackLeibler`](@ref)

**Mathematical Definition:**

| Function | Formula |
|:---------|:--------|
| ``\gamma(u)`` | ``-\log(u) + u - 1`` |
| ``\gamma'(u)`` | ``1 - 1/u`` |
| ``\gamma''(u)`` | ``1/u^2`` |
| ``\psi(v)`` | ``-\log(1 - v)`` |
| ``\psi'(v)`` | ``1/(1 - v)`` |
| Domain of ``\gamma`` | ``u > 0`` |
| Domain of ``\psi`` | ``v < 1`` |

### Chi-Squared Divergence

**Type:** [`ChiSquared`](@ref)

**Mathematical Definition:**

| Function | Formula |
|:---------|:--------|
| ``\gamma(u)`` | ``(u - 1)^2 / 2`` |
| ``\gamma'(u)`` | ``u - 1`` |
| ``\gamma''(u)`` | ``1`` |
| ``\psi(v)`` | ``v^2/2 + v`` |
| ``\psi'(v)`` | ``v + 1`` |
| Domain of ``\gamma`` | ``u \in \mathbb{R}`` |
| Domain of ``\psi`` | ``v \in \mathbb{R}`` |

### Hellinger Divergence

**Type:** [`Hellinger`](@ref)

**Mathematical Definition:**

| Function | Formula |
|:---------|:--------|
| ``\gamma(u)`` | ``2(\sqrt{u} - 1)^2 = 2u - 4\sqrt{u} + 2`` |
| ``\gamma'(u)`` | ``2 - 2/\sqrt{u}`` |
| ``\gamma''(u)`` | ``1/u^{3/2}`` |
| ``\psi(v)`` | ``2v/(2 - v)`` |
| ``\psi'(v)`` | ``4/(2 - v)^2`` |
| Domain of ``\gamma`` | ``u > 0`` |
| Domain of ``\psi`` | ``v < 2`` |

### Cressie-Read Family

**Type:** [`CressieRead`](@ref)

**Mathematical Definition:**

| Function | Formula |
|:---------|:--------|
| ``\gamma_\alpha(u)`` | ``\frac{u^{1+\alpha} - 1}{\alpha(1+\alpha)} - \frac{u - 1}{\alpha}`` |
| ``\gamma'_\alpha(u)`` | ``\frac{u^\alpha - 1}{\alpha}`` |
| ``\gamma''_\alpha(u)`` | ``u^{\alpha - 1}`` |
| ``\psi_\alpha(v)`` | ``u^* v - \gamma_\alpha(u^*)`` where ``u^* = (1 + \alpha v)^{1/\alpha}`` |
| ``\psi'_\alpha(v)`` | ``(1 + \alpha v)^{1/\alpha}`` |
| Domain of ``\gamma`` | ``u \geq 0`` (for ``\alpha > 0``), ``u > 0`` (for ``\alpha < 0``) |
| Domain of ``\psi`` | ``1 + \alpha v > 0`` |

**Special Cases:**

| ``\alpha`` | Divergence |
|:-----------|:-----------|
| ``\alpha \to 0`` | Kullback-Leibler |
| ``\alpha \to -1`` | Reverse Kullback-Leibler |
| ``\alpha = -0.5`` | Hellinger |
| ``\alpha = 1`` | Chi-squared |

## Modified Divergences

Modified divergences extend the domain of the base divergence by replacing the tail with a quadratic function.

### Modified Divergence

**Type:** [`ModifiedDivergence`](@ref)

**Construction:**

For a base divergence ``\gamma`` and threshold ``\rho > 1``:

```math
\gamma_\rho(u) = \begin{cases}
  \gamma(\rho) + \gamma'(\rho)(u-\rho) + \frac{1}{2}\gamma''(\rho)(u-\rho)^2, & u > \rho \\
  \gamma(u), & u \leq \rho
\end{cases}
```

**Example:**
```@example modified
using Divergences
# Create a modified Kullback-Leibler divergence
md = ModifiedDivergence(KullbackLeibler(), 1.5)
# The divergence behaves like KL for u ≤ 1.5, quadratic for u > 1.5
md(2.0)  # Uses quadratic extension
```

### Fully Modified Divergence

**Type:** [`FullyModifiedDivergence`](@ref)

**Construction:**

For a base divergence ``\gamma``, lower threshold ``\varphi \in (0, 1)``, and upper threshold ``\rho > 1``:

```math
\gamma_{\varphi, \rho}(u) = \begin{cases}
  \gamma(\rho) + \gamma'(\rho)(u-\rho) + \frac{1}{2}\gamma''(\rho)(u-\rho)^2, & u > \rho \\
  \gamma(u), & \varphi \leq u \leq \rho \\
  \gamma(\varphi) + \gamma'(\varphi)(u-\varphi) + \frac{1}{2}\gamma''(\varphi)(u-\varphi)^2, & u < \varphi
\end{cases}
```

**Example:**
```@example fullymodified
using Divergences
# Create a fully modified Kullback-Leibler divergence
fmd = FullyModifiedDivergence(KullbackLeibler(), 0.5, 1.5)
# Uses quadratic extension for u < 0.5 and u > 1.5
fmd(0.3)  # Uses lower quadratic extension
```

## Summary Table

| Type | Constructor | ``\gamma(u)`` | Domain |
|:-----|:------------|:--------------|:-------|
| Kullback-Leibler | `KullbackLeibler()` | ``u \log u - u + 1`` | ``u > 0`` |
| Reverse KL | `ReverseKullbackLeibler()` | ``-\log u + u - 1`` | ``u > 0`` |
| Hellinger | `Hellinger()` | ``2(\sqrt{u} - 1)^2`` | ``u > 0`` |
| Chi-Squared | `ChiSquared()` | ``(u-1)^2/2`` | ``u \in \mathbb{R}`` |
| Cressie-Read | `CressieRead(α)` | ``\frac{u^{1+\alpha} - 1}{\alpha(1+\alpha)} - \frac{u-1}{\alpha}`` | Depends on ``\alpha`` |

## Conjugate Functions Summary

| Type | ``\psi(v)`` | Domain |
|:-----|:------------|:-------|
| Kullback-Leibler | ``e^v - 1`` | ``v \in \mathbb{R}`` |
| Reverse KL | ``-\log(1-v)`` | ``v < 1`` |
| Hellinger | ``2v/(2-v)`` | ``v < 2`` |
| Chi-Squared | ``v^2/2 + v`` | ``v \in \mathbb{R}`` |
| Cressie-Read | See above | ``1 + \alpha v > 0`` |
