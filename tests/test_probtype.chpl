/*
 * test_probtype.chpl - Comprehensive tests for ProbType
 *
 * Covers:
 *   1. Constructors - moments, kind, validation
 *   2. logProb - known values, support boundaries, -inf
 *   3. Exact arithmetic - N+N, Poisson+Poisson, Exp+Exp=>Gamma
 *   4. Moment propagation - DerivedDist correctness
 *   5. Sampling - empirical mean/variance vs analytic
 *   6. sampleBatch - parallel correctness
 *   7. condition - Gaussian conjugate exact posterior
 *   8. Dirichlet - sampleDirichlet, updateDirichlet, toCategorical
 *   9. Type safety - private fields, invariant preservation
 *  10. Distribution interface - Distribution requirement
 *
 * Compile:
 *   chpl src/ProbType.chpl tests/TestRunner.chpl tests/test_probtype.chpl \
 *        -o tests/_test_probtype
 *   ./tests/_test_probtype
 */

use ProbType;
use TestRunner;
use Random;
use Math;

var rng = new randomStream(real, seed=47634);
const N     = 50_000;
const TOL_S = 0.05;   // statistical tolerance for empirical tests

proc empMoments(d: prob, n: int): (real, real) {
  var s = 0.0; var s2 = 0.0;
  var dc = d;
  for 1..n { const x = dc.sample(rng); s += x; s2 += x*x; }
  const m = s / n:real;
  return (m, s2/n:real - m*m);
}

// =========================================================================
// Suite 1 - Constructors and analytic moments
// =========================================================================

proc testConstructors() {
  beginSuite("Constructors and analytic moments");

  // Gaussian
  var g = gaussian(2.0, 3.0);
  assertEqual(g.mean(),     2.0, 1e-9, "Gaussian mean = mu");
  assertEqual(g.variance(), 9.0, 1e-9, "Gaussian variance = sigma^2");
  assertEqual(g.std(),      3.0, 1e-9, "Gaussian std = sigma");
  assertTrue(g.isGaussian(),            "isGaussian() = true");
  assertFalse(g.isDerived(),            "isDerived() = false for Gaussian");

  // Beta
  var b = beta(3.0, 7.0);
  assertNear(b.mean(), 3.0/10.0, 1e-9, "Beta mean = alpha/(a+b)");
  assertNear(b.variance(), (3.0*7.0)/(100.0*11.0), 1e-9, "Beta variance");
  assertTrue(b.isBeta(), "isBeta() = true");

  // Gamma
  var gm = gamma(4.0, 2.0);
  assertEqual(gm.mean(),     2.0,   1e-9, "Gamma mean = shape/rate");
  assertEqual(gm.variance(), 1.0,   1e-9, "Gamma variance = shape/rate^2");

  // Exponential
  var e = exponential(2.0);
  assertEqual(e.mean(),     0.5,   1e-9, "Exp mean = 1/rate");
  assertEqual(e.variance(), 0.25,  1e-9, "Exp variance = 1/rate^2");

  // Bernoulli
  var br = bernoulli(0.3);
  assertEqual(br.mean(),     0.3,     1e-9, "Bernoulli mean = p");
  assertEqual(br.variance(), 0.3*0.7, 1e-9, "Bernoulli variance = p(1-p)");

  // Poisson
  var po = poisson(5.0);
  assertEqual(po.mean(),     5.0, 1e-9, "Poisson mean = lambda");
  assertEqual(po.variance(), 5.0, 1e-9, "Poisson variance = lambda");

  // Dirichlet - scalar mean
  var dr = dirichlet([2.0, 3.0, 5.0]);
  // Scalar mean = sum(i * alpha_i / sum(alpha)) = (0*2+1*3+2*5)/10 = 1.3
  assertNear(dr.mean(), 1.3, 1e-9, "Dirichlet scalar mean");
  assertTrue(dr.isDirichlet(), "isDirichlet() = true");

  endSuite();
}

// =========================================================================
// Suite 2 - logProb: known values and support
// =========================================================================

proc testLogProb() {
  beginSuite("logProb - known values and support");

  // Gaussian N(0,1): logProb(0) = -0.5*log(2*pi)
  var g = gaussian(0.0, 1.0);
  assertNear(g.logProb(0.0), -0.5*log(2.0*pi), 1e-9,
             "N(0,1) logProb(0) = -0.5*log(2*pi)");
  assertTrue(isFinite(g.logProb(100.0)),
             "N(0,1) logProb finite on all reals");

  // Beta: out-of-support => -inf
  var b = beta(2.0, 2.0);
  assertEqual(b.logProb(0.0), -Math.inf, 1e-9, "Beta logProb(0) = -inf");
  assertEqual(b.logProb(1.0), -Math.inf, 1e-9, "Beta logProb(1) = -inf");
  assertEqual(b.logProb(-0.1),-Math.inf, 1e-9, "Beta logProb(-0.1) = -inf");
  assertTrue(isFinite(b.logProb(0.5)),           "Beta logProb(0.5) finite");

  // Gamma: x ≤ 0 => -inf
  var gm = gamma(2.0, 1.0);
  assertEqual(gm.logProb(-0.1), -Math.inf, 1e-9, "Gamma logProb(-0.1) = -inf");
  assertEqual(gm.logProb(0.0),  -Math.inf, 1e-9, "Gamma logProb(0) = -inf");
  assertTrue(isFinite(gm.logProb(1.0)),           "Gamma logProb(1.0) finite");

  // Exponential: logProb(0) = log(rate)
  var ex = exponential(2.0);
  assertNear(ex.logProb(0.0), log(2.0), 1e-9, "Exp(2) logProb(0) = log(2)");
  assertEqual(ex.logProb(-0.1), -Math.inf, 1e-9,"Exp logProb(-0.1) = -inf");

  // Bernoulli: logProb(1) = log(p)
  var br = bernoulli(0.3);
  assertNear(br.logProb(1.0), log(0.3), 1e-9, "Bernoulli logProb(1) = log(p)");
  assertNear(br.logProb(0.0), log(0.7), 1e-9, "Bernoulli logProb(0) = log(1-p)");

  // Poisson: logProb(0) = -lambda
  var po = poisson(3.0);
  assertNear(po.logProb(0.0), -3.0, 1e-9, "Poisson(3) logProb(0) = -lambda");

  endSuite();
}

// =========================================================================
// Suite 3 - Exact arithmetic dispatch
// =========================================================================

proc testExactArithmetic() {
  beginSuite("Exact arithmetic dispatch");

  // N + N => Gaussian
  var s = gaussian(1.0, 2.0) + gaussian(3.0, 4.0);
  assertTrue(s.isGaussian(),                   "N+N => Gaussian");
  assertEqual(s.mean(),     4.0,     1e-9,     "N+N mean = mu1+mu2");
  assertNear(s.variance(),  4.0+16.0,1e-9,     "N+N var = s1^2+s2^2");

  // N - N => Gaussian
  var d = gaussian(5.0, 2.0) - gaussian(1.0, 1.0);
  assertTrue(d.isGaussian(),                   "N-N => Gaussian");
  assertEqual(d.mean(), 4.0, 1e-9,             "N-N mean = mu1-mu2");

  // N × c => Gaussian
  var sc = gaussian(1.0, 1.0) * 3.0;
  assertTrue(sc.isGaussian(),                  "N*c => Gaussian");
  assertEqual(sc.mean(),     3.0,   1e-9,      "N*c mean = mu*c");
  assertEqual(sc.variance(), 9.0,   1e-9,      "N*c var = sigma^2*c^2");

  // N + c => Gaussian
  var sh = gaussian(1.0, 2.0) + 5.0;
  assertTrue(sh.isGaussian(),                  "N+c => Gaussian");
  assertEqual(sh.mean(), 6.0,       1e-9,      "N+c mean = mu+c");
  assertEqual(sh.variance(), 4.0,   1e-9,      "N+c var unchanged");

  // -N => Gaussian
  var ng = -gaussian(2.0, 1.0);
  assertTrue(ng.isGaussian(),                  "-N => Gaussian");
  assertEqual(ng.mean(), -2.0,      1e-9,      "-N mean = -mu");

  // Commutativity: c + N = N + c
  var ca = 5.0 + gaussian(1.0, 2.0);
  assertNear(ca.mean(), sh.mean(), 1e-9,       "c+N = N+c (commutative)");

  // Poisson + Poisson => Poisson
  var pp = poisson(2.0) + poisson(3.0);
  assertTrue(pp.isPoisson(),                   "Poisson+Poisson => Poisson");
  assertEqual(pp.mean(), 5.0,       1e-9,      "Poisson sum lambda=5");

  // Exp + Exp (same rate) => Gamma(2,λ)
  var eg = exponential(2.0) + exponential(2.0);
  assertTrue(eg.isGamma(),                     "Exp+Exp same rate => Gamma");
  assertEqual(eg.mean(), 1.0,       1e-9,      "Gamma(2,2) mean = 1.0");

  // Exp + Exp (different rates) => DerivedDist
  var edd = exponential(2.0) + exponential(3.0);
  assertTrue(edd.isDerived(),                  "Exp+Exp diff rates => Derived");

  // X^2: E[X^2] = Var(X) + E[X]^2
  var x = gaussian(1.0, 1.0);
  assertNear(( x**2 ).mean(), x.variance() + x.mean()**2, 1e-9,
             "X^2 mean = Var+E^2 (exact second moment)");

  endSuite();
}

// =========================================================================
// Suite 4 - Moment propagation correctness
// =========================================================================

proc testMomentPropagation() {
  beginSuite("Moment propagation for DerivedDist");

  // N + Beta: mean and variance should still be correct
  var a = gaussian(1.0, 1.0);
  var b = beta(3.0, 7.0);
  var d = a + b;
  assertTrue(d.isDerived(),                          "N+Beta => Derived");
  assertNear(d.mean(), a.mean() + b.mean(), 1e-9,    "Derived mean = E[a]+E[b]");
  assertNear(d.variance(), a.variance() + b.variance(), 1e-9,
             "Derived var = Var[a]+Var[b] (independent)");

  // Scalar multiplication
  var sc = poisson(3.0) * 2.0;
  assertNear(sc.mean(),     6.0,    1e-9, "Poisson*2 mean");
  assertNear(sc.variance(), 12.0,   1e-9, "Poisson*2 var = lambda*4");

  // Chaining: (N(0,1) + 2) * 3 should be N(6,3)
  var chain = (gaussian(0.0, 1.0) + 2.0) * 3.0;
  assertTrue(chain.isGaussian(),                "Chain stays Gaussian");
  assertNear(chain.mean(), 6.0,    1e-9,        "Chain mean = 6");
  assertNear(chain.std(),  3.0,    1e-9,        "Chain std = 3");

  endSuite();
}

// =========================================================================
// Suite 5 - Sampling: empirical vs analytic
// =========================================================================

proc testSampling() {
  beginSuite("Sampling - empirical moments vs analytic");

  proc check(d: prob, name: string) {
    const (em, ev) = empMoments(d, N);
    assertInRange(em, d.mean() - TOL_S, d.mean() + TOL_S,
                  name + " empirical mean ≈ analytic");
    assertInRange(ev,
                  d.variance() * (1.0 - TOL_S*3),
                  d.variance() * (1.0 + TOL_S*3),
                  name + " empirical variance ≈ analytic");
  }

  check(gaussian(2.0, 1.5),   "Gaussian(2, 1.5)");
  check(beta(3.0, 7.0),       "Beta(3, 7)");
  check(gamma(4.0, 2.0),      "Gamma(4, 2)");
  check(exponential(3.0),     "Exponential(3)");
  check(bernoulli(0.3),       "Bernoulli(0.3)");
  check(poisson(5.0),         "Poisson(5)");

  // Exponential: all samples must be ≥ 0
  var ec = exponential(1.0);
  var hasNeg = false;
  for 1..1000 { if ec.sample(rng) < 0.0 then hasNeg = true; }
  assertFalse(hasNeg, "Exponential samples always ≥ 0");

  // Beta: all samples in (0,1)
  var bc = beta(2.0, 2.0);
  var outOfRange = false;
  for 1..1000 {
    const s = bc.sample(rng);
    if s <= 0.0 || s >= 1.0 then outOfRange = true;
  }
  assertFalse(outOfRange, "Beta samples always in (0,1)");

  endSuite();
}

// =========================================================================
// Suite 6 - sampleBatch parallel correctness
// =========================================================================

proc testSampleBatch() {
  beginSuite("sampleBatch - parallel sampling");

  var d = gaussian(3.0, 1.5);
  const samples = sampleBatch(d, N, seed=42);

  assertEqual(samples.size, N, "sampleBatch returns N samples");

  const empM = (+ reduce samples) / N:real;
  const empV = (+ reduce (samples * samples)) / N:real - empM**2;

  assertInRange(empM, 3.0 - TOL_S, 3.0 + TOL_S,
                "sampleBatch empirical mean ≈ 3.0");
  assertInRange(empV, 2.25*(1-TOL_S*3), 2.25*(1+TOL_S*3),
                "sampleBatch empirical var ≈ 2.25");

  // Reproducibility: same seed => same samples
  const s2 = sampleBatch(d, 100, seed=42);
  const s3 = sampleBatch(d, 100, seed=42);
  var same = true;
  for i in 0..#100 { if abs(s2[i] - s3[i]) > 1e-12 then same = false; }
  assertTrue(same, "sampleBatch reproducible with same seed");

  // Different seed => different samples
  const s4 = sampleBatch(d, 100, seed=99);
  var diff = false;
  for i in 0..#100 { if abs(s2[i] - s4[i]) > 1e-6 then diff = true; }
  assertTrue(diff, "sampleBatch different with different seeds");

  endSuite();
}

// =========================================================================
// Suite 7 - condition: Bayesian updating
// =========================================================================

proc testCondition() {
  beginSuite("condition - Gaussian conjugate posterior");

  // Prior N(0, 10²), likelihood N(mu, 1), obs=3.5
  // Posterior:
  //   prior_prec = 1/100 = 0.01
  //   like_prec  = 1/1   = 1.0
  //   post_prec  = 1.01
  //   post_var   = 1/1.01 ≈ 0.9901
  //   post_mean  = 0.9901 * (0*0.01 + 3.5*1.0) ≈ 3.4653

  var prior = gaussian(0.0, 10.0);
  var post  = condition(prior, 3.5, likelihoodVar=1.0);

  assertTrue(post.isGaussian(),                   "condition(Gaussian) => Gaussian");
  assertNear(post.mean(),     3.4653, 1e-3,       "posterior mean ≈ 3.4653 (analytic)");
  assertNear(post.variance(), 0.9901, 1e-3,       "posterior var ≈ 0.9901 (analytic)");
  assertLT(post.variance(), prior.variance(),     "posterior narrower than prior");

  // Multiple sequential updates converge
  var belief = gaussian(0.0, 5.0);
  const obsArr = [3.0, 3.2, 2.8, 3.1, 2.9];
  for o in obsArr do belief = condition(belief, o, likelihoodVar=1.0);
  assertInRange(belief.mean(), 2.0, 4.0,          "sequential updates converge");
  assertLT(belief.std(), 5.0,                     "uncertainty decreases");

  endSuite();
}

// =========================================================================
// Suite 8 - Dirichlet
// =========================================================================

proc testDirichlet() {
  beginSuite("Dirichlet - constructor and operations");

  // sampleDirichlet: vector sums to 1
  var dr  = dirichlet([2.0, 3.0, 5.0]);
  for 1..20 {
    const vec = sampleDirichlet(dr, rng);
    assertNear(+ reduce vec, 1.0, 1e-10, "sampleDirichlet sums to 1");
    for v in vec do assertTrue(v >= 0.0, "sampleDirichlet values ≥ 0");
  }

  // updateDirichlet: posterior = prior + counts
  var prior  = dirichletUniform(3, 1.0);
  const counts = [8.0, 2.0, 1.0];
  var post   = updateDirichlet(prior, counts);
  // alpha_post = [9, 3, 2], sum = 14
  assertNear(post.mean(),
             (0.0*9+1.0*3+2.0*2)/14.0, 1e-9,
             "updateDirichlet posterior mean");

  // Concentrated prior => low variance
  var conc = dirichlet([100.0, 1.0, 1.0]);
  assertLT(conc.variance(), prior.variance(), "concentrated prior has less variance");

  endSuite();
}

// =========================================================================
// Suite 9 - Type safety: private fields
// =========================================================================

proc testTypeSafety() {
  beginSuite("Type safety - distKind and invariants");

  // Each constructor produces the correct kind
  assertTrue(gaussian(0.0,1.0).distKind()  == DistKind.GaussianDist,    "kind Gaussian");
  assertTrue(beta(1.0,1.0).distKind()      == DistKind.BetaDist,        "kind Beta");
  assertTrue(gamma(1.0,1.0).distKind()     == DistKind.GammaDist,       "kind Gamma");
  assertTrue(exponential(1.0).distKind()   == DistKind.ExponentialDist, "kind Exponential");
  assertTrue(bernoulli(0.5).distKind()     == DistKind.BernoulliDist,   "kind Bernoulli");
  assertTrue(poisson(1.0).distKind()       == DistKind.PoissonDist,     "kind Poisson");
  assertTrue(dirichlet([1.0,1.0]).distKind()== DistKind.DirichletDist,  "kind Dirichlet");

  // Predicates are mutually exclusive
  var g = gaussian(0.0, 1.0);
  assertTrue( g.isGaussian(),   "isGaussian for Gaussian");
  assertFalse(g.isBeta(),       "!isBeta for Gaussian");
  assertFalse(g.isPoisson(),    "!isPoisson for Gaussian");
  assertFalse(g.isDerived(),    "!isDerived for Gaussian");

  // Operators preserve Gaussian type
  assertTrue((g + g).isGaussian(), "G+G preserves Gaussian");
  assertTrue((g * 2.0).isGaussian(),"G*c preserves Gaussian");
  assertTrue((g + 1.0).isGaussian(),"G+c preserves Gaussian");
  assertTrue((-g).isGaussian(),     "-G preserves Gaussian");

  // Non-Gaussian sum => DerivedDist
  assertTrue((g + bernoulli(0.5)).isDerived(), "G+B => Derived");

  endSuite();
}

// =========================================================================
// Suite 10 - Distribution interface
// =========================================================================

proc testDistributionInterface() {
  beginSuite("Distribution interface");

  // A proc that works on any Distribution
  proc entropy(d: prob, n: int, ref rng: randomStream(real)): real {
    var h = 0.0;
    var dc = d;
    for 1..n { const x = dc.sample(rng); h -= d.logProb(x); }
    return h / n:real;
  }

  // Gaussian entropy: H = 0.5*log(2*pi*e*sigma^2)
  const hG     = entropy(gaussian(0.0, 1.0), N, rng);
  const hGTrue = 0.5 * log(2.0 * pi * exp(1.0));
  assertInRange(hG, hGTrue - 0.05, hGTrue + 0.05,
                "Gaussian entropy empirical ≈ 0.5*log(2*pi*e)");

  // Exponential entropy: H = 1 - log(lambda)
  const hE     = entropy(exponential(1.0), N, rng);
  const hETrue = 1.0;
  assertInRange(hE, hETrue - 0.05, hETrue + 0.05,
                "Exp(1) entropy empirical ≈ 1.0");

  // All distributions satisfy logProb ≤ 0 for high-mass regions
  // (not always true but holds for normalizable distributions at their modes)
  for d in [gaussian(0.0,1.0), beta(2.0,2.0),
            bernoulli(0.5),   poisson(5.0)] {
    assertTrue(isFinite(d.logProb(d.mean())),
               d.name() + " logProb at mean is finite");
  }

  endSuite();
}

// =========================================================================
// Main
// =========================================================================

proc main() {
  writeln("TAP version 13");
  writeln("# ProbType - Proposal Reference Implementation Tests");
  writeln("# Chapel 2.8 | FreeBSD 14+");
  writeln();

  testConstructors();
  testLogProb();
  testExactArithmetic();
  testMomentPropagation();
  testSampling();
  testSampleBatch();
  testCondition();
  testDirichlet();
  testTypeSafety();
  testDistributionInterface();

  writeln("1..", _total);
  printSummary();

  if !allPassed() then exit(1);
}
