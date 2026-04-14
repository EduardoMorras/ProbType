# ProbType - Probabilistic Types for Chapel

**Proposal Package** | Chapel Improvement Proposal (pending number)  
**Status:** Draft  
**Author:** Eduardo ()  
**Requires:** Chapel 2.8+ | No external dependencies

---

## What This Is

A proposal to add a `ProbType` module to the Chapel standard library,
providing `prob` - a value-type record representing probability distributions
as first-class values.

This package contains:

- **`docs/CHIP-prob.md`** - the formal proposal document
- **`src/ProbType.chpl`** - reference implementation (Chapel 2.8, ~600 lines)
- **`examples/`** - four worked examples demonstrating the design
- **`tests/test_probtype.chpl`** - ~90 test assertions

---

## Quick Start

```sh
# Run all examples
make

# Run the test suite
make test

# Individual examples
make 01   # basic distributions and moments
make 02   # exact arithmetic dispatch
make 03   # Bayesian updating
make 04   # parallel sampling (set CHPL_RT_NUM_THREADS_PER_LOCALE)
```

No external dependencies. `ProbType.chpl` uses only `Math`, `Random`,
`CTypes`, and `List` from the Chapel standard library.

---

## The Core Idea in 10 Lines

```chapel
use ProbType;

var x = gaussian(0.0, 1.0);          // N(0,1) - value type, 48 bytes
var y = gaussian(3.0, 0.5);

var z = x + y;                        // N(3.0, √1.25) - EXACT, not approximate
writeln(z.name());                    // "Gaussian(mu=3.0, sigma=1.118)"

var p = x ** 2;                       // E[X²] = Var(X) + E[X]² = 1 - exact
var post = condition(x, obs=2.5);    // Bayesian update - conjugate Gaussian

// Parallel sampling - no shared state, linear scaling
var samples = sampleBatch(z, 1_000_000, seed=42);
```

---

## Why This Belongs in the Standard Library

**1. Value semantics.** `prob` is a record - stack-allocated, copyable, storable
in arrays. No boxing, no heap allocation per distribution. An array of
1 million `prob` values occupies 48 MB.

**2. Exact arithmetic.** When mathematical closure holds, operators return the
correct distributional family:

| Expression | Result |
|---|---|
| `N(μ₁,σ₁) + N(μ₂,σ₂)` | `N(μ₁+μ₂, √(σ₁²+σ₂²))` |
| `Poisson(λ₁) + Poisson(λ₂)` | `Poisson(λ₁+λ₂)` |
| `Exp(λ) + Exp(λ)` | `Gamma(2, λ)` |

**3. `forall` integration.** Parallel Monte Carlo is the natural expression:

```chapel
// N independent posterior samples, each task has its own RNG
forall i in 0..#N with (var rng = new randomStream(real, seed=i)) do
  samples[i] = myDist.sample(rng);
```

**4. No language changes required.** The implementation uses only existing
Chapel 2.8 features: `extern {}` inline C, `private var`, operator overloading,
and the `interface` keyword. A future CHIP may propose native tagged unions
to make the implementation even cleaner.

---

## Distributions

| Family | Constructor | Support | Properties |
|---|---|---|---|
| Gaussian | `gaussian(μ, σ)` | ℝ | Closed under +, -, × |
| Beta | `beta(α, β)` | (0,1) | Conjugate prior for Bernoulli |
| Gamma | `gamma(k, λ)` | ℝ⁺ | Closed under +; generalizes Exp |
| Exponential | `exponential(λ)` | ℝ⁺ | `Exp+Exp => Gamma` exact |
| Bernoulli | `bernoulli(p)` | {0,1} | Binary logic foundation |
| Poisson | `poisson(λ)` | ℕ | `Poisson+Poisson => Poisson` exact |
| Dirichlet | `dirichlet(α[])` | simplex | Prior over probability vectors |

---

## Files

```
ProbType-Proposal/
├ docs/
│   └ CHIP-prob.md                  ← formal proposal
├ src/
│   └ ProbType.chpl                 ← reference implementation
├ examples/
│   ├ 01_basic_distributions.chpl   ← constructors, moments, logProb
│   ├ 02_exact_arithmetic.chpl      ← exact dispatch demonstration
│   ├ 03_bayesian_updating.chpl     ← condition, Dirichlet, convergence
│   └ 04_parallel_sampling.chpl     ← forall, coforall MCMC preview
├ tests/
│   ├ TestRunner.chpl               ← minimal TAP framework
│   └ test_probtype.chpl            ← ~90 assertions
├ Makefile
└ README.md                         ← this file
```

---

## Relationship to Chapel's Existing Libraries

`ProbType` does not overlap with `Random` (which provides RNG primitives) or
`Math` (transcendental functions). It sits one level above: using `Random` for
sampling and `Math` for logProb computations, while providing the distributional
abstractions that neither library addresses.

---

## Submitting Feedback

This is a draft proposal. Discussion is welcome at this repository GitHub issues

The reference implementation has been validated on Chapel 2.8 / FreeBSD 14+.
