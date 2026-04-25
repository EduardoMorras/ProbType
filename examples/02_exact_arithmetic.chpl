/*
 * 02_exact_arithmetic.chpl
 *
 * Demonstrates exact arithmetic dispatch in ProbType.
 *
 * When the mathematical closure property holds, arithmetic on prob values
 * produces the correct distributional family - not just correct moments.
 * This is a key differentiator from ad-hoc (mean, variance) representations.
 *
 * Compile:
 *   chpl src/ProbType.chpl examples/02_exact_arithmetic.chpl -o 02_arithmetic
 *   ./02_arithmetic
 */

use ProbType;
use Math;

proc header(title: string) {
  writeln();
  writeln(" ", title, " ");
}

proc showOp(expr: string, result: prob, expected: string) {
  const kindStr = if result.isGaussian()    then "Gaussian"
                  else if result.isPoisson() then "Poisson"
                  else if result.isGamma()   then "Gamma"
                  else if result.isDerived() then "Derived"
                  else result.distKind():string;
  writef("  %<35s => %<12s  %s\n", expr, kindStr, result.name());
  writef("    mean=%.4r  std=%.4r  (expected family: %s)\n",
         result.mean(), result.std(), expected);
}

proc main() {
  writeln("ProbType - Exact Arithmetic Dispatch");

  header("Gaussian closure under +, -, ×");
  // N(μ₁,σ₁) + N(μ₂,σ₂) = N(μ₁+μ₂, √(σ₁²+σ₂²))
  showOp("N(1,2) + N(3,4)",
         gaussian(1.0,2.0) + gaussian(3.0,4.0),
         "Gaussian");

  // N(μ,σ) - N(ν,τ) = N(μ-ν, √(σ²+τ²))
  showOp("N(5,2) - N(1,1)",
         gaussian(5.0,2.0) - gaussian(1.0,1.0),
         "Gaussian");

  // N(μ,σ) × c = N(μ·c, σ·|c|)
  showOp("N(1,1) * 3.0",
         gaussian(1.0,1.0) * 3.0,
         "Gaussian");

  // N(μ,σ) + c = N(μ+c, σ)
  showOp("N(0,1) + 5.0",
         gaussian(0.0,1.0) + 5.0,
         "Gaussian");

  // Negation
  showOp("-N(2,1)",
         -gaussian(2.0,1.0),
         "Gaussian");

  header("Poisson closure under +");
  // Poisson(λ₁) + Poisson(λ₂) = Poisson(λ₁+λ₂)
  // Proof: sum of independent Poissons is Poisson with summed rates
  showOp("Poisson(2) + Poisson(3)",
         poisson(2.0) + poisson(3.0),
         "Poisson");
  showOp("Poisson(1) + Poisson(1) + Poisson(1)",
         poisson(1.0) + poisson(1.0) + poisson(1.0),
         "Poisson(3)");

  header("Exponential => Gamma (Erlang)");
  // Exp(λ) + Exp(λ) = Gamma(2, λ) - Erlang-2
  // This is the waiting time for the second event in a Poisson process
  showOp("Exp(2) + Exp(2)",
         exponential(2.0) + exponential(2.0),
         "Gamma(2,2)");

  header("Moment propagation (DerivedDist)");
  // When no exact closure applies, moments are still correct
  // but the distributional family is approximated as Gaussian
  showOp("N(0,1) + Beta(2,5)",
         gaussian(0.0,1.0) + beta(2.0,5.0),
         "Derived (moment propagation)");
  showOp("N(0,1) + Poisson(3)",
         gaussian(0.0,1.0) + poisson(3.0),
         "Derived (moment propagation)");
  showOp("Exp(2) + Exp(3)  [different rates]",
         exponential(2.0) + exponential(3.0),
         "Derived (not Erlang)");

  header("Power rule: moments of X^n");
  // For X ~ N(1, 1): E[X²] = Var(X) + E[X]² = 1 + 1 = 2
  var x = gaussian(1.0, 1.0);
  var x2 = x ** 2;
  writef("  N(1,1)^2: E[X²] = %.4r  (exact: Var+E² = %r)\n",
         x2.mean(), x.variance() + x.mean()**2);

  header("Chaining: (N(0,1) + 2.0) * 3.0");
  var chain = (gaussian(0.0, 1.0) + 2.0) * 3.0;
  // Should be N(6, 3) - exact through two operations
  writeln("  Result: ", chain.name());
  writef("  mean=%.4r (expected 6.0)  std=%.4r (expected 3.0)\n",
         chain.mean(), chain.std());
}
