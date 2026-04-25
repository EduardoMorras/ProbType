/*
 * 01_basic_distributions.chpl
 *
 * Demonstrates the six distribution families in ProbType.
 * Verifies analytical moments (mean, variance) against empirical estimates.
 *
 * Compile:
 *   chpl src/ProbType.chpl examples/01_basic_distributions.chpl -o 01_basic
 *   ./01_basic
 */

use ProbType;
use Random;
use Math;

const N = 200_000;

proc empirical(d: prob, n: int): (real, real) {
  var rng = new randomStream(real, seed=42);
  var sum = 0.0; var sum2 = 0.0;
  var dc  = d;
  for 1..n { const s = dc.sample(rng); sum += s; sum2 += s*s; }
  const m = sum / n:real;
  return (m, sum2/n:real - m*m);
}

proc check(d: prob, name: string) {
  const (empM, empV) = empirical(d, N);
  const tol          = 5e-2;
  const mOK          = abs(empM - d.mean())     / (abs(d.mean()) + 1e-6)     < tol;
  const vOK          = abs(empV - d.variance()) / (abs(d.variance()) + 1e-6) < tol;
  writef(" %<30s  E[X] theory=%8.4dr empirical=%8.4dr %s  "
       + "  \t   Var[X] theory=%8.4dr empirical=%8.4dr %s\n",
         name,
         d.mean(),     empM, if mOK then "" else "",
         d.variance(), empV, if vOK then "" else "");
}

proc main() {
  writeln("ProbType - Distribution Families");
  writeln("N = ", N, " samples per distribution");
  writeln();

  //  Continuous families 

  writeln("Continuous:");
  check(gaussian(2.0, 3.0),     "Gaussian(mu=2, sigma=3)");
  check(beta(3.0, 7.0),         "Beta(alpha=3, beta=7)");
  check(gamma(4.0, 2.0),        "Gamma(shape=4, rate=2)");
  check(exponential(0.5),       "Exponential(rate=0.5)");
  writeln();

  //  Discrete families 

  writeln("Discrete:");
  check(bernoulli(0.3),         "Bernoulli(p=0.3)");
  check(poisson(7.0),           "Poisson(lambda=7)");
  writeln();

  //  logProb verification 

  writeln("logProb values:");
  // N(0,1): logProb(0) = -0.5·log(2Pi) ≈ -0.9189
  var g = gaussian(0.0, 1.0);
  writef("  Gaussian(0,1).logProb(0.0) = %.6r  (expected %.6r)\n",
         g.logProb(0.0), -0.5*log(2.0*pi));

  // Exp(1): logProb(0) = log(1) - 1*0 = 0
  var e = exponential(1.0);
  writef("  Exponential(1).logProb(0.0) = %.6r  (expected 0.0)\n",
         e.logProb(0.0));

  // Beta(2,2) at x=0.5 - the mode
  var b = beta(2.0, 2.0);
  writef("  Beta(2,2).logProb(0.5) = %.6r\n", b.logProb(0.5));
  writef("  Beta(2,2).logProb(0.0) = %.6r  (expected -inf)\n",
         b.logProb(0.0));
  writeln();

  //  Dirichlet 

  writeln("Dirichlet:");
  var dr    = dirichlet([2.0, 3.0, 5.0]);
  writeln("  ", dr.name());
  writef("  E[k] = %.4r  (expected %.4r = sum(i*alpha_i)/sum(alpha))\n",
         dr.mean(), (0.0*2+1.0*3+2.0*5)/10.0);
  var rng42 = new randomStream(real, seed=42);
  writeln("  Sample vectors (should sum to 1.0):");
  for 1..3 {
    const vec = sampleDirichlet(dr, rng42);
    writef("    [%.3r, %.3r, %.3r]  sum=%.6r\n",
           vec[0], vec[1], vec[2], + reduce vec);
  }
  writeln();

  //  Distribution summary 

  writeln("Summary output:");
  gaussian(1.5, 2.0).summary();
  poisson(5.0).summary();
  dirichlet([1.0, 1.0, 1.0]).summary();
}
