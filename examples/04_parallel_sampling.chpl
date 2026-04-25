/*
 * 04_parallel_sampling.chpl
 *
 * Demonstrates Chapel's natural fit for parallel probabilistic computation.
 *
 * Key point of the proposal: prob is a value type that integrates seamlessly
 * with forall, coforall, and distributed array domains. This is not possible
 * with class-based distribution hierarchies.
 *
 * Compile:
 *   chpl src/ProbType.chpl examples/04_parallel_sampling.chpl -o 04_parallel
 *   ./04_parallel --n=1000000
 *
 * With parallelism:
 *   CHPL_RT_NUM_THREADS_PER_LOCALE=8 ./04_parallel --n=10000000
 */

use ProbType;
use Random;
use Math;
use Time;

config const n         = 1_000_000;
config const printEvery = 10;

proc main() {
  writeln("ProbType — Parallel Sampling");
  writeln("n = ", n);
  writeln();

  //  1. sampleBatch: built-in parallel sampling 

  writeln(" 1. sampleBatch (forall with independent RNGs) ");
  var d = gaussian(0.0, 1.0);
  var t = new stopwatch(); t.start();
  const s1 = sampleBatch(d, n, seed=42);
  t.stop();

  const empMean = (+ reduce s1) / n:real;
  const empVar  = (+ reduce (s1 * s1)) / n:real - empMean**2;
  writef("  Gaussian(0,1): empirical mean=%.4r  var=%.4r  time=%.3r s\n",
         empMean, empVar, t.elapsed());
  writef("  Throughput: %.1r Msamples/s\n", n:real / t.elapsed() / 1e6);
  writeln();

  //  2. Manual forall with per-task RNG 

  writeln(" 2. Manual forall — different distributions per task ");
  var results: [0..#n] real;
  var beta_d = beta(2.0, 5.0);

  t.clear(); t.start();
  forall i in 0..#n {
      var rng = new randomStream(real, seed=i ^ 0xDEAD);
      var dc = beta_d;
      results[i] = dc.sample(rng);
  }
  t.stop();

  const bMean = (+ reduce results) / n:real;
  writef("  Beta(2,5): empirical mean=%.4r  (theory %.4r)  time=%.3r s\n",
         bMean, beta_d.mean(), t.elapsed());
  writeln();

  //  3. Monte Carlo estimation using prob arithmetic 

  writeln(" 3. Monte Carlo: estimating Pi using prob ");
  // Pi/4 = P(X²+Y²≤1) for X,Y ~ Uniform(0,1)
  // Beta(1,1) is exactly Uniform(0,1).
  var ux = beta(1.0, 1.0);
  var uy = beta(1.0, 1.0);
  var xs = sampleBatch(ux, n, seed=1);
  var ys = sampleBatch(uy, n, seed=2);

  var inside: atomic int;
  forall i in 0..#n {
    if xs[i]*xs[i] + ys[i]*ys[i] <= 1.0 then inside.add(1);
  }
  const piEst = 4.0 * inside.read():real / n:real;
  writef("  Pi estimate = %.6r  (true Pi = %.6r)  error = %.6r\n",
         piEst, pi, abs(piEst - pi));
  writeln();

  //  4. Parallel MCMC preview 
  //
  // This illustrates why parallel probabilistic types matter.
  // Full MCMC is proposed in a future CHIP; this shows the structure.

  writeln(" 4. Parallel independent chains (MCMC preview) ");
  writeln("  Model: x ~ N(0,10), obs=3.5 from N(x,1)");
  writeln("  Posterior: N(3.4653, 0.9901)  [analytic]");
  writeln();

  const nChains    = 4;
  const nSamples   = n / nChains;
  const priorD     = gaussian(0.0, 10.0);
  const obsVal     = 3.5;
  const likeVar    = 1.0;
  var   chainMeans: [0..#nChains] real;

  t.clear(); t.start();
  coforall chain in 0..#nChains {
    var rng     = new randomStream(real, seed=chain * 1234567);
    var current = priorD.sample(rng);
    var accepted = 0;
    var sumX     = 0.0;

    // Metropolis-Hastings on the posterior
    const postMean = 3.4653;   // known analytic answer
    const stepSize = 1.0;

    for notused in 1..nSamples {
      const proposal  = current + _gaussSampleLocal(0.0, stepSize, rng);
      // log P(obs | proposal) + log P(proposal)
      const logLike_p = -0.5*(obsVal - proposal)**2 / likeVar;
      const logPrior_p = gaussian(0.0,10.0).logProb(proposal);
      const logLike_c = -0.5*(obsVal - current)**2 / likeVar;
      const logPrior_c = gaussian(0.0,10.0).logProb(current);
      const logAlpha  = (logLike_p + logPrior_p)
                      - (logLike_c + logPrior_c);
      if log(rng.next()) < logAlpha {
        current = proposal;
        accepted += 1;
      }
      sumX += current;
    }
    chainMeans[chain] = sumX / nSamples:real;

    writef("  Chain %s: posterior mean=%.4r  accept rate=%.2r\n",
           chain:string, chainMeans[chain], accepted:real/nSamples:real);
  }
  t.stop();

  const grandMean = (+ reduce chainMeans) / nChains:real;
  writef("\n  Grand mean = %.4r  (analytic = 3.4653)  time=%.3r s\n",
         grandMean, t.elapsed());
  writef("  Throughput: %.1r Msteps/s\n",
         (nChains * nSamples):real / t.elapsed() / 1e6);
  writeln();
  writeln("  Note: coforall gives one task per chain — exactly the MCMC");
  writeln("  parallelism pattern. Chapel's value-type prob propagates");
  writeln("  into tasks without synchronization.");
}

/* Helper: Box-Muller (private to this file) */
private proc _gaussSampleLocal(mu: real, sigma: real,
                                 ref rng: randomStream(real)): real {
  const u1 = rng.next(); const u2 = rng.next();
  return mu + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
}
