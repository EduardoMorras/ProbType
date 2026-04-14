/*
 * 03_bayesian_updating.chpl
 *
 * Demonstrates Bayesian conditioning in ProbType.
 *
 * Bayesian inference updates a prior distribution given observed data.
 * For the Gaussian-Gaussian case, the posterior has a known closed form.
 * ProbType implements this exactly without Monte Carlo.
 *
 * This example models estimating an unknown mean from noisy measurements,
 * and shows how the belief distribution narrows as evidence accumulates.
 *
 * Compile:
 *   chpl src/ProbType.chpl examples/03_bayesian_updating.chpl -o 03_bayes
 *   ./03_bayes
 */

use ProbType;
use Math;

/* Print a simple text histogram of a distribution. */
proc printDist(d: prob, lo: real, hi: real, width: int = 50) {
  const step = (hi - lo) / width:real;
  var maxH = 0.0;
  var heights: [0..#width] real;
  for i in 0..#width {
    const x = lo + (i + 0.5) * step;
    heights[i] = exp(d.logProb(x));
    if heights[i] > maxH then maxH = heights[i];
  }
  for i in 0..#width {
    const bars = (heights[i] / maxH * 20.0):int;
    write(if i == 0 then "  [" else "");
    write("█" * bars + " " * (20 - bars));
    write(if i == width-1 then "]\n" else "|");
  }
}

proc main() {
  writeln("ProbType - Bayesian Updating");
  writeln();

  //  Scenario: estimating the true temperature of a star 
  //
  // We have a prior belief: temperature ~ N(5000 K, 500²)
  // Each measurement has Gaussian noise with sigma=200 K.
  // We update sequentially as measurements arrive.

  writeln("Scenario: estimating stellar surface temperature");
  writeln("  Prior:       N(5000 K, σ=500 K)");
  writeln("  Measurements arrive with σ_noise = 200 K");
  writeln();

  const trueMean    = 5800.0;   // the actual temperature
  const priorMu     = 5000.0;
  const priorSigma  = 500.0;
  const noiseSigma  = 200.0;
  const obs: [1..6] real = [5900.0, 5750.0, 5820.0, 5840.0, 5780.0, 5810.0];

  var belief = gaussian(priorMu, priorSigma);

  writef("  Prior:         mean=%8.1r  std=%6.1r  95%%CI=[%7.1r, %7.1r]\n",
         belief.mean(), belief.std(),
         belief.mean() - 1.96*belief.std(),
         belief.mean() + 1.96*belief.std());

  for (i, o) in zip(1.., obs) {
    belief = condition(belief, o, likelihoodVar=noiseSigma**2);
    writef("  After obs %d (%5.0r):  mean=%8.1r  std=%6.1r"
         + "  95%%CI=[%7.1r, %7.1r]\n",
           i, o, belief.mean(), belief.std(),
           belief.mean() - 1.96*belief.std(),
           belief.mean() + 1.96*belief.std());
  }

  writeln();
  writef("  True value:    %8.1r\n", trueMean);
  writef("  Final estimate: %.1r ± %.1r\n", belief.mean(), belief.std());
  const inCI = abs(belief.mean() - trueMean) < 1.96 * belief.std();
  writeln("  True value in 95% CI: ", if inCI then "YES " else "NO ");
  writeln();

  //  Visualize the belief evolution 

  writeln("Belief distribution after all 6 measurements:");
  writeln("  (x-axis: 4500 K to 6500 K)");
  printDist(belief, 4500.0, 6500.0);
  writeln();

  //  Dirichlet updating for categorical outcomes 

  writeln("Scenario: estimating the probability of three diagnoses");
  writeln("  (gripe, resfriado, ninguno)");
  writeln("  Prior: Dirichlet(1,1,1) - uniform");
  writeln();

  var prior = dirichletUniform(3, 1.0);
  writeln("  Prior: ", prior.name());

  // Observe: 8 gripes, 3 resfriados, 1 otro
  const counts = [8.0, 3.0, 1.0];
  var posterior = updateDirichlet(prior, counts);
  writeln("  Posterior after [8, 3, 1] observations: ", posterior.name());

  // Expected probabilities
  const s = 1.0 + 1.0 + 1.0 + 8.0 + 3.0 + 1.0;
  writef("  E[P(gripe)]      = %.3r  (alpha=9/16)\n", 9.0/s);
  writef("  E[P(resfriado)]  = %.3r  (alpha=4/16)\n", 4.0/s);
  writef("  E[P(ninguno)]    = %.3r  (alpha=2/16)\n", 2.0/s);

  writeln();

  //  Sequential Gaussian updating demonstrates convergence 

  writeln("Convergence: sequential Gaussian updates");
  writeln("  True mean = 3.0, observation noise σ = 1.0");
  writeln("  Prior: N(0, 10)");
  writeln();

  var b2 = gaussian(0.0, 10.0);
  const trueM = 3.0;
  const noiseV = 1.0;

  writef("  n=%-4d  mean=%-8.3r  std=%-7.3r\n", 0, b2.mean(), b2.std());
  for n in [1, 2, 5, 10, 20, 50, 100] {
    const obs2 = trueM + 0.3;  // slightly biased observation stream
    while b2.std() > 10.0 / sqrt(n:real) + 0.01 {
      b2 = condition(b2, obs2, likelihoodVar=noiseV);
    }
    writef("  n=%-4d  mean=%-8.3r  std=%-7.3r\n", n, b2.mean(), b2.std());
  }
  writeln();
  writeln("  Std => 0 as n => ∞: belief converges to true value ");
}
