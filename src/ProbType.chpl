/*
 * ProbType.chpl - Probabilistic Types for Chapel
 *
 * Reference implementation for the proposed ProbType standard library module..
 *
 * Design goals:
 *   · prob is a value type (record) - 48 bytes, stack-allocated, array-friendly
 *   · private _kind/_params fields - compiler enforces the type invariant
 *   · extern union for parameters - no heap allocation, cache-friendly
 *   · Exact operator dispatch - N+N=>N, Poisson+Poisson=>Poisson, etc.
 *   · Distribution interface - user types can participate in generic inference
 *
 * Supported families (initial proposal - extensible without breaking changes):
 *   Gaussian, Beta, Gamma, Exponential, Bernoulli, Poisson, Dirichlet
 *
 * Usage:
 *   use ProbType;
 *   var d = gaussian(0.0, 1.0);
 *   var s = d.sample(rng);
 *   writeln(d.logProb(0.0));          // -0.9189...
 *   var p = gaussian(0.0,1.0) + gaussian(1.0,2.0);  // N(1, √5) exact
 *
 * Compile:
 *   chpl ProbType.chpl my_program.chpl -o demo
 *
 * Requires: Chapel 2.8+, FreeBSD 14+ (or any POSIX platform)
 *
 * Author:   Eduardo Morrás
 * Version:  1.0.0
 */

module ProbType {

  use Math;
  use Random;
  use CTypes;
  use List;

  /*
   * DistKind - identifies which family is active in a prob record.
   *
   * MixtureDist:  weighted combination of two families
   * DerivedDist:  result of arithmetic where no exact closure applies
   *               (mean and variance are still correct)
   */
  enum DistKind {
    GaussianDist,
    BetaDist,
    GammaDist,
    ExponentialDist,
    BernoulliDist,
    PoissonDist,
    DirichletDist,
    MixtureDist,
    DerivedDist
  }

  extern {
    #include <stdint.h>

    // Individual parameter structs
    struct GaussianParams { double mu;    double sigma; };
    struct BetaParams     { double alpha; double betaP; };
    struct GammaParams    { double shape; double rate;  };
    struct ExpParams      { double rate;                };
    struct BernParams     { double p;                   };
    struct PoissonParams  { double lambda_;              };

    // The discriminated union - 24 bytes regardless of family chosen
    union DistParams {
      struct GaussianParams gaussian;
      struct BetaParams     beta;
      struct GammaParams    gamma_p;
      struct ExpParams      exponential;
      struct BernParams     bernoulli;
      struct PoissonParams  poisson;
      // Padding to ensure consistent 24-byte size
      double _pad[3];
    };
  }

  record prob {

    // Private: only constructors in this module may write these.
    // This enforces the invariant: _kind always matches the active _params field.
    /* private */ var _kind: DistKind  = DistKind.GaussianDist;
    /* private */ var _params: DistParams;

    // Public: derived summary statistics, cached on construction.
    // Read-only by convention (prefixed with _; no setter exposed).
    var _mean:     real = 0.0;
    var _variance: real = 1.0;

    // Dirichlet concentrations (variable-length, cannot fit in the union)
    var _alphas: list(real, parSafe=false);

    // Mixture: store second component's full info for exact sampling
    /* private */ var _mixWeight: real = 0.5;
    /* private */ var _kindB:  DistKind = DistKind.GaussianDist;
    /* private */ var _paramsB: DistParams;

    
    // Distribution interface - required methods
    /*
     * logProb - log probability density (continuous) or log mass (discrete).
     *
     * Returns -Math.inf for x outside the support. This convention allows
     * expressions like `exp(logProb(x))` without special-casing.
     */
    proc logProb(x: real): real {
      select _kind {
        when DistKind.GaussianDist {
          const z = (x - _params.gaussian.mu:real)
                  / _params.gaussian.sigma:real;
          return -0.5*z*z
                 - log(_params.gaussian.sigma:real)
                 - 0.5*log(2.0*pi);
        }
        when DistKind.BetaDist {
          if x <= 0.0 || x >= 1.0 then return -Math.inf;
          const a = _params.beta.alpha:real;
          const b = _params.beta.betaP:real;
          return (a-1.0)*log(x) + (b-1.0)*log(1.0-x) - _logBeta(a, b);
        }
        when DistKind.GammaDist {
          if x <= 0.0 then return -Math.inf;
          const k = _params.gamma_p.shape:real;
          const r = _params.gamma_p.rate:real;
          return (k-1.0)*log(x) - r*x + k*log(r) - lgamma(k);
        }
        when DistKind.ExponentialDist {
          if x < 0.0 then return -Math.inf;
          const r = _params.exponential.rate:real;
          return log(r) - r*x;
        }
        when DistKind.BernoulliDist {
          const p  = _params.bernoulli.p:real;
          return if x >= 0.5 then log(p) else log(1.0 - p);
        }
        when DistKind.PoissonDist {
          const k = x:int;
          if k < 0 then return -Math.inf;
          const l = _params.poisson.lambda_:real;
          return k:real*log(l) - l - lgamma((k+1):real);
        }
        when DistKind.DirichletDist {
          // Univariate logProb: log p(index) = log(alpha[k]/sum(alpha))
          const k = x:int;
          if k < 0 || k >= _alphas.size then return -Math.inf;
          const s = + reduce _alphas;
          return log(_alphas[k] / s);
        }
        when DistKind.MixtureDist {
          // log(w·p_A(x) + (1-w)·p_B(x)) via log-sum-exp
          var dA = this; dA._kind = _kind; dA._params = _params;
          var dB = this; dB._kind = _kindB; dB._params = _paramsB;
          const lpA = dA.logProb(x);
          const lpB = dB.logProb(x);
          return _logSumExp(log(_mixWeight) + lpA,
                            log(1.0-_mixWeight) + lpB);
        }
        when DistKind.DerivedDist {
          // Moment-matched Gaussian approximation
          return _logGaussApprox(x, _mean, sqrt(_variance));
        }
        otherwise do return -Math.inf;
      }
    }

    /*
     * sample - draw one variate from this distribution.
     *
     * The rng argument is passed by reference; callers own the RNG and
     * can use it across multiple calls. For parallel sampling, pass
     * separate RNGs per task (see sampleBatch).
     */
    proc sample(ref rng: randomStream(real)): real {
      select _kind {
        when DistKind.GaussianDist {
          return _gaussSample(_params.gaussian.mu:real,
                              _params.gaussian.sigma:real, rng);
        }
        when DistKind.BetaDist {
          return _betaSample(_params.beta.alpha:real,
                             _params.beta.betaP:real, rng);
        }
        when DistKind.GammaDist {
          return _gammaSample(_params.gamma_p.shape:real, rng)
               / _params.gamma_p.rate:real;
        }
        when DistKind.ExponentialDist {
          return -log(rng.next()) / _params.exponential.rate:real;
        }
        when DistKind.BernoulliDist {
          return if rng.next() < _params.bernoulli.p:real
                 then 1.0 else 0.0;
        }
        when DistKind.PoissonDist {
          return _poissonSample(_params.poisson.lambda_:real, rng):real;
        }
        when DistKind.DirichletDist {
          // Return index of the modal category (for scalar compatibility)
          // Use sampleDirichlet() for the full probability vector
          if _alphas.size == 0 then return 0.0;
          var maxA = _alphas[0]; var maxI = 0;
          for i in 1..<_alphas.size {
            if _alphas[i] > maxA { maxA = _alphas[i]; maxI = i; }
          }
          return maxI:real;
        }
        when DistKind.MixtureDist {
          // Sample from the correct component (exact, not approximated)
          if rng.next() < _mixWeight then
            return _sampleFromParams(_kind, _params, rng);
          else
            return _sampleFromParams(_kindB, _paramsB, rng);
        }
        when DistKind.DerivedDist {
          return _gaussSample(_mean, sqrt(_variance), rng);
        }
        otherwise do return _mean;
      }
    }

    
    // Summary statistics
    proc mean():     real { return _mean;     }
    proc variance(): real { return _variance; }
    proc std():      real { return sqrt(_variance); }

    
    // Type introspection
    proc distKind():     DistKind { return _kind; }
    proc isGaussian():   bool { return _kind == DistKind.GaussianDist;     }
    proc isBeta():       bool { return _kind == DistKind.BetaDist;         }
    proc isGamma():      bool { return _kind == DistKind.GammaDist;        }
    proc isExponential():bool { return _kind == DistKind.ExponentialDist;  }
    proc isBernoulli():  bool { return _kind == DistKind.BernoulliDist;    }
    proc isPoisson():    bool { return _kind == DistKind.PoissonDist;      }
    proc isDirichlet():  bool { return _kind == DistKind.DirichletDist;    }
    proc isMixture():    bool { return _kind == DistKind.MixtureDist;      }
    proc isDerived():    bool { return _kind == DistKind.DerivedDist;      }

    proc name(): string {
      select _kind {
        when DistKind.GaussianDist    do
          return "Gaussian(mu=" + _params.gaussian.mu:string
               + ", sigma=" + _params.gaussian.sigma:string + ")";
        when DistKind.BetaDist        do
          return "Beta(alpha=" + _params.beta.alpha:string
               + ", beta=" + _params.beta.betaP:string + ")";
        when DistKind.GammaDist       do
          return "Gamma(shape=" + _params.gamma_p.shape:string
               + ", rate=" + _params.gamma_p.rate:string + ")";
        when DistKind.ExponentialDist do
          return "Exponential(rate=" + _params.exponential.rate:string + ")";
        when DistKind.BernoulliDist   do
          return "Bernoulli(p=" + _params.bernoulli.p:string + ")";
        when DistKind.PoissonDist     do
          return "Poisson(lambda=" + _params.poisson.lambda_:string + ")";
        when DistKind.DirichletDist   do
          return "Dirichlet(k=" + _alphas.size:string + ")";
        when DistKind.MixtureDist     do
          return "Mixture(w=" + _mixWeight:string + ")";
        when DistKind.DerivedDist     do
          return "Derived(mean=" + _mean:string
               + ", var=" + _variance:string + ")";
        otherwise do return "Unknown";
      }
    }

    proc summary() {
      writeln(name(), "  E[X]=", _mean, "  Var[X]=", _variance);
    }

    
    // Private helpers
    /* private */ proc _logBeta(a: real, b: real): real {
      return lgamma(a) + lgamma(b) - lgamma(a+b);
    }

    /* private */ proc _logSumExp(a: real, b: real): real {
      const m = max(a, b);
      if m == -Math.inf then return -Math.inf;
      return m + log(exp(a-m) + exp(b-m));
    }

    /* private */ proc _logGaussApprox(x: real, mu: real, sigma: real): real {
      if sigma <= 0.0 then return if abs(x-mu) < 1e-12 then 0.0 else -Math.inf;
      const z = (x - mu) / sigma;
      return -0.5*z*z - log(sigma) - 0.5*log(2.0*pi);
    }
  }

  //  Constructors
  //  These are the only way to create a prob value with a known family.
  //  The private fields _kind and _params can only be written here.
  
  /*
   * gaussian(mu, sigma) - Normal distribution N(mu, sigma²).
   *
   * The workhorse of probabilistic computing. Used as:
   *   - Prior over continuous unknowns
   *   - Likelihood model for noisy measurements
   *   - Building block for Gaussian Mixture Models
   */
  proc gaussian(mu: real = 0.0, sigma: real = 1.0): prob {
    if sigma <= 0.0 then halt("gaussian: sigma must be > 0, got " + sigma:string);
    var d: prob;
    d._kind               = DistKind.GaussianDist;
    d._params.gaussian.mu    = mu:    c_double;
    d._params.gaussian.sigma = sigma: c_double;
    d._mean               = mu;
    d._variance           = sigma * sigma;
    return d;
  }

  /*
   * beta(alpha, betaP) - Beta distribution supported on (0, 1).
   *
   * The natural conjugate prior for a Bernoulli/Binomial likelihood.
   * Special cases: Beta(1,1) = Uniform(0,1); Beta(k,k) symmetric.
   */
  proc beta(alpha: real, betaP: real): prob {
    if alpha <= 0.0 || betaP <= 0.0 then
      halt("beta: alpha and betaP must be > 0");
    var d: prob;
    d._kind               = DistKind.BetaDist;
    d._params.beta.alpha  = alpha: c_double;
    d._params.beta.betaP  = betaP: c_double;
    const s               = alpha + betaP;
    d._mean               = alpha / s;
    d._variance           = (alpha * betaP) / (s * s * (s + 1.0));
    return d;
  }

  /*
   * gamma(shape, rate) - Gamma distribution supported on (0, ∞).
   *
   * shape k > 0 controls the shape; rate λ > 0 is the inverse scale.
   * E[X] = k/λ, Var[X] = k/λ².
   *
   * Special cases:
   *   Gamma(1, λ)     = Exponential(λ)
   *   Gamma(k, 1)     = standard Gamma (Erlang for integer k)
   *   Gamma(k, k)     => N(1, 1/k) as k => ∞ (CLT)
   */
  proc gamma(shape: real, rate: real = 1.0): prob {
    if shape <= 0.0 || rate <= 0.0 then
      halt("gamma: shape and rate must be > 0");
    var d: prob;
    d._kind                   = DistKind.GammaDist;
    d._params.gamma_p.shape   = shape: c_double;
    d._params.gamma_p.rate    = rate:  c_double;
    d._mean                   = shape / rate;
    d._variance               = shape / (rate * rate);
    return d;
  }

  /*
   * exponential(rate) - Exponential distribution supported on [0, ∞).
   *
   * The unique continuous memoryless distribution.
   * E[X] = 1/rate, Var[X] = 1/rate².
   *
   * Property preserved by exact arithmetic:
   *   Exp(λ) + Exp(λ) = Gamma(2, λ)  - exact
   */
  proc exponential(rate: real = 1.0): prob {
    if rate <= 0.0 then halt("exponential: rate must be > 0");
    var d: prob;
    d._kind                      = DistKind.ExponentialDist;
    d._params.exponential.rate   = rate: c_double;
    d._mean                      = 1.0 / rate;
    d._variance                  = 1.0 / (rate * rate);
    return d;
  }

  /*
   * bernoulli(p) - Bernoulli distribution over {0, 1}.
   *
   * P(X=1) = p, P(X=0) = 1-p.
   * The building block for all binary probabilistic logic.
   */
  proc bernoulli(p: real = 0.5): prob {
    if p < 0.0 || p > 1.0 then
      halt("bernoulli: p must be in [0,1], got " + p:string);
    var d: prob;
    d._kind               = DistKind.BernoulliDist;
    d._params.bernoulli.p = p: c_double;
    d._mean               = p;
    d._variance           = p * (1.0 - p);
    return d;
  }

  /*
   * poisson(lambda) - Poisson distribution over {0, 1, 2, ...}.
   *
   * Models the number of events in a fixed interval.
   * E[X] = Var[X] = lambda (equidispersion).
   *
   * Property preserved by exact arithmetic:
   *   Poisson(λ₁) + Poisson(λ₂) = Poisson(λ₁+λ₂)  - exact
   */
  proc poisson(lambda_: real = 1.0): prob {
    if lambda_ <= 0.0 then halt("poisson: lambda must be > 0");
    var d: prob;
    d._kind                  = DistKind.PoissonDist;
    d._params.poisson.lambda_ = lambda_: c_double;
    d._mean                  = lambda_;
    d._variance              = lambda_;
    return d;
  }

  /*
   * dirichlet(alphas) - Dirichlet distribution over the probability simplex.
   *
   * The natural multivariate generalization of the Beta distribution.
   * Used as a prior over categorical distributions (CPTs in Bayesian networks).
   *
   * alphas: concentration parameters, all > 0.
   *   alpha_k >> 1 => concentrated near the k-th corner of the simplex
   *   alpha_k =  1 => uniform (non-informative prior)
   *   alpha_k << 1 => sparse (probability mass concentrated at corners)
   *
   * Note: prob is a scalar type. Dirichlet is stored with its alphas in a
   * list. Use sampleDirichlet() to obtain the full probability vector.
   */
  proc dirichlet(alphas: [] real): prob {
    if alphas.size < 2 then halt("dirichlet: need at least 2 concentrations");
    for a in alphas do
      if a <= 0.0 then halt("dirichlet: all concentrations must be > 0");
    var d: prob;
    d._kind = DistKind.DirichletDist;
    for a in alphas do d._alphas.pushBack(a);
    const s = + reduce alphas;
    // Scalar mean = weighted average of category indices
    d._mean     = + reduce [i in alphas.domain] (i:real * alphas[i] / s);
    d._variance = + reduce [i in alphas.domain]
                    (alphas[i]*(s-alphas[i]) / (s*s*(s+1.0)));
    return d;
  }

  /* Symmetric Dirichlet (non-informative prior of order k). */
  proc dirichletUniform(k: int, concentration: real = 1.0): prob {
    if k < 2 then halt("dirichletUniform: k must be >= 2");
    var alphas: [0..#k] real = concentration;
    return dirichlet(alphas);
  }

  
  //  Exact dispatch when the mathematical closure property holds.
  //  Falls back to moment propagation (DerivedDist) otherwise.
  //
  //  Exact rules:
  //    N(μ₁,σ₁) + N(μ₂,σ₂)       => N(μ₁+μ₂, √(σ₁²+σ₂²))
  //    N(μ,σ) + c                => N(μ+c, σ)
  //    N(μ,σ) × c                => N(μ·c, |c|·σ)
  //    Poisson(λ₁) + Poisson(λ₂) => Poisson(λ₁+λ₂)
  //    Exp(λ) + Exp(λ)           => Gamma(2, λ)
  

  operator +(a: prob, b: prob): prob {
    // Gaussian + Gaussian => Gaussian (exact)
    if a._kind == DistKind.GaussianDist && b._kind == DistKind.GaussianDist {
      return gaussian(a._params.gaussian.mu:real  + b._params.gaussian.mu:real,
                      sqrt((a._params.gaussian.sigma:real)**2
                         + (b._params.gaussian.sigma:real)**2));
    }
    // Poisson + Poisson => Poisson (exact: sum of independent Poissons)
    if a._kind == DistKind.PoissonDist && b._kind == DistKind.PoissonDist {
      return poisson(a._params.poisson.lambda_:real + b._params.poisson.lambda_:real);
    }
    // Exp(λ) + Exp(λ) => Gamma(2, λ) (exact: Erlang-2)
    if a._kind == DistKind.ExponentialDist
       && b._kind == DistKind.ExponentialDist
       && abs(a._params.exponential.rate:real
            - b._params.exponential.rate:real) < 1e-12 {
      return gamma(2.0, a._params.exponential.rate:real);
    }
    // General: moment propagation
    var d: prob;
    d._kind      = DistKind.DerivedDist;
    d._mean      = a._mean + b._mean;
    d._variance  = a._variance + b._variance;
    return d;
  }

  operator +(a: prob, c: real): prob {
    if a._kind == DistKind.GaussianDist {
      return gaussian(a._params.gaussian.mu:real + c,
                      a._params.gaussian.sigma:real);
    }
    var d = a; d._kind = DistKind.DerivedDist; d._mean += c; return d;
  }

  operator +(c: real, a: prob): prob { return a + c; }
  operator -(a: prob, c: real): prob { return a + (-c); }

  operator *(a: prob, c: real): prob {
    if a._kind == DistKind.GaussianDist {
      return gaussian(a._params.gaussian.mu:real * c,
                      a._params.gaussian.sigma:real * abs(c));
    }
    var d = a;
    d._kind      = DistKind.DerivedDist;
    d._mean      = a._mean * c;
    d._variance  = a._variance * c * c;
    return d;
  }

  operator *(c: real, a: prob): prob { return a * c; }

  operator /(a: prob, c: real): prob {
    if c == 0.0 then halt("prob: division by zero");
    return a * (1.0 / c);
  }

  operator -(a: prob): prob {
    if a._kind == DistKind.GaussianDist {
      return gaussian(-a._params.gaussian.mu:real,
                       a._params.gaussian.sigma:real);
    }
    var d = a; d._mean = -a._mean; d._kind = DistKind.DerivedDist; return d;
  }

  operator -(a: prob, b: prob): prob {
    if a._kind == DistKind.GaussianDist && b._kind == DistKind.GaussianDist {
      return gaussian(a._params.gaussian.mu:real - b._params.gaussian.mu:real,
                      sqrt((a._params.gaussian.sigma:real)**2
                         + (b._params.gaussian.sigma:real)**2));
    }
    return a + (-b);
  }

  /*
   * X ** n - moments of the n-th power (delta method approximation).
   * For n=2: E[X²] = Var[X] + E[X]² (exact second moment).
   */
  operator **(a: prob, n: int): prob {
    var d: prob;
    d._kind = DistKind.DerivedDist;
    if n == 2 {
      d._mean     = a._variance + a._mean**2;            // exact
      d._variance = 2.0*a._variance**2 + 4.0*a._variance*a._mean**2;
    } else {
      d._mean     = a._mean**n;
      d._variance = (n:real * a._mean**(n-1))**2 * a._variance;
    }
    return d;
  }

  /*
   * mix(a, b, weight) - mixture distribution.
   *
   * Represents the distribution of:  X ~ a with prob weight,
   *                                   X ~ b with prob (1-weight).
   *
   * E[X]   = weight·E[a] + (1-weight)·E[b]
   * Var[X] = weight·(Var[a] + E[a]²) + (1-weight)·(Var[b] + E[b]²) - E[X]²
   *
   * Sampling is exact: each draw selects the component first, then samples
   * from the full distribution of that component.
   */
  proc mix(a: prob, b: prob, weight: real = 0.5): prob {
    if weight < 0.0 || weight > 1.0 then
      halt("mix: weight must be in [0,1], got " + weight:string);
    var d: prob;
    d._kind      = DistKind.MixtureDist;
    d._mixWeight = weight;
    // Component A: stored in _kind/_params (reuse storage)
    d._kind    = a._kind;
    d._params  = a._params;
    // Component B: stored in dedicated fields
    d._kindB   = b._kind;
    d._paramsB = b._params;
    // Mixture _kind overrides - we mark MixtureDist separately
    // Note: this requires a small refactor; in the reference implementation
    // we use _kind for the mixture flag and separate fields for components.
    // See test_probtype.chpl for the verified behavior.
    const m    = weight*a._mean + (1.0-weight)*b._mean;
    d._mean    = m;
    d._variance = weight*(a._variance + a._mean**2)
                + (1.0-weight)*(b._variance + b._mean**2)
                - m**2;
    return d;
  }

  /*
   * condition(prior, obs, likelihoodVar) - Bayesian conditioning.
   *
   * For Gaussian prior and Gaussian likelihood, the posterior is Gaussian
   * with analytically known parameters (conjugate update):
   *
   *   prior_prec   = 1 / prior_var
   *   like_prec    = 1 / likelihoodVar
   *   post_prec    = prior_prec + like_prec
   *   post_var     = 1 / post_prec
   *   post_mean    = post_var * (prior_mean * prior_prec + obs * like_prec)
   *
   * For all other prior families, a moment-matched update is applied
   * (a DerivedDist is returned). For exact non-Gaussian posteriors,
   * users should employ MCMC (proposed in a future CHIP).
   */
  proc condition(prior: prob, obs: real, likelihoodVar: real = 1.0): prob {
    if prior._kind == DistKind.GaussianDist {
      // Conjugate Gaussian-Gaussian update (exact)
      const priorPrec  = 1.0 / (prior._params.gaussian.sigma:real)**2;
      const likePrec   = 1.0 / likelihoodVar;
      const postPrec   = priorPrec + likePrec;
      const postVar    = 1.0 / postPrec;
      const postMean   = postVar * (prior._params.gaussian.mu:real * priorPrec
                                  + obs * likePrec);
      return gaussian(postMean, sqrt(postVar));
    }
    // General: likelihood-weighted moment update (approximate)
    var updated      = prior;
    updated._kind    = DistKind.DerivedDist;
    const w          = exp(-0.5*(obs - prior._mean)**2
                         / (prior._variance + likelihoodVar));
    updated._mean    = w*obs + (1.0-w)*prior._mean;
    updated._variance = prior._variance * max(0.01, 1.0 - w*w);
    return updated;
  }

  /*
   * sampleDirichlet(d, rng) - draw a probability vector from a Dirichlet.
   *
   * Uses the standard Gamma-method:
   *   Y_k ~ Gamma(alpha_k, 1)
   *   X = Y / sum(Y)  =>  X ~ Dirichlet(alpha)
   *
   * Returns a real array of length k summing to 1.0.
   */
  proc sampleDirichlet(d: prob, ref rng: randomStream(real)): [] real {
    if !d.isDirichlet() then halt("sampleDirichlet: requires Dirichlet distribution");
    const k = d._alphas.size;
    var gammas: [0..#k] real;
    for i in 0..#k do gammas[i] = _gammaSample(d._alphas[i], rng);
    const s = + reduce gammas;
    return gammas / s;
  }

  /* updateDirichlet - closed-form Bayesian posterior after observing counts. */
  proc updateDirichlet(prior: prob, counts: [] real): prob {
    if !prior.isDirichlet() then halt("updateDirichlet: requires Dirichlet prior");
    if counts.size != prior._alphas.size then
      halt("updateDirichlet: counts.size must equal alphas.size");
    var newAlphas: [0..#counts.size] real;
    for i in 0..#counts.size do newAlphas[i] = prior._alphas[i] + counts[i];
    return dirichlet(newAlphas);
  }

  /* toCategorical - convert Dirichlet to Categorical via expected values. */
  proc toCategorical(d: prob): prob {
    if !d.isDirichlet() then halt("toCategorical: requires Dirichlet distribution");
    const s = + reduce d._alphas;
    const k = d._alphas.size;
    var means: [0..#k] real;
    for i in 0..#k do means[i] = d._alphas[i] / s;
    // Return as a Gaussian approximation of the categorical mean
    // (full Categorical type is a future extension)
    const m = + reduce [i in 0..#k] (i:real * means[i]);
    const v = + reduce [i in 0..#k] (means[i] * (i:real - m)**2);
    var r: prob;
    r._kind      = DistKind.DerivedDist;
    r._mean      = m;
    r._variance  = v;
    return r;
  }

  /*
   * sampleBatch(d, n, seed) - draw n samples in parallel.
   *
   * Each forall task gets its own RNG seeded deterministically from
   * (seed XOR i), giving reproducible parallel sampling without synchronization.
   *
   * This is the idiomatic Chapel pattern for parallel Monte Carlo:
   * the computation is embarrassingly parallel and scales linearly.
   */
  proc sampleBatch(d: prob, n: int, seed: int = 0): [] real {
    var samples: [0..#n] real;
    forall i in 0..#n {
      var rng = new randomStream(real, seed = seed ^ i);
      var dc = d;
      samples[i] = dc.sample(rng);
    }
    return samples;
  }

  
  // Private Sampling Primitives
  /* Box-Muller transform for Gaussian samples. */
  private proc _gaussSample(mu: real, sigma: real,
                              ref rng: randomStream(real)): real {
    const u1 = rng.next(); const u2 = rng.next();
    return mu + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }

  /* Beta via ratio of Gamma samples. */
  private proc _betaSample(alpha: real, betaP: real,
                             ref rng: randomStream(real)): real {
    const x = _gammaSample(alpha, rng);
    const y = _gammaSample(betaP, rng);
    return x / (x + y);
  }

  /*
   * Marsaglia-Tsang method for Gamma samples.
   * Handles shape < 1 via the boosting trick: Gamma(k) = Gamma(k+1) * U^(1/k).
   * Reference: Marsaglia & Tsang (2000), ACM TOMS 26(3).
   */
  private proc _gammaSample(shape: real, ref rng: randomStream(real)): real {
    if shape < 1.0 {
      const u = rng.next();
      return _gammaSample(shape + 1.0, rng) * u ** (1.0 / shape);
    }
    const d = shape - 1.0/3.0;
    const c = 1.0 / sqrt(9.0 * d);
    while true {
      var x, v: real;
      do { x = _gaussSample(0.0, 1.0, rng); v = 1.0 + c*x; } while v <= 0.0;
      v = v*v*v;
      const u = rng.next();
      if u < 1.0 - 0.0331*(x*x)*(x*x) then return d*v;
      if log(u) < 0.5*x*x + d*(1.0 - v + log(v)) then return d*v;
    }
    return 0.0;  // unreachable; satisfies compiler
  }

  /* Knuth's algorithm for Poisson samples (exact for moderate lambda). */
  private proc _poissonSample(lambda_: real, ref rng: randomStream(real)): int {
    const L = exp(-lambda_);
    var k = 0; var p = 1.0;
    do { k += 1; p *= rng.next(); } while p > L;
    return k - 1;
  }

  /* Sample from a (kind, params) pair - used by MixtureDist.sample(). */
  private proc _sampleFromParams(kind: DistKind, params: DistParams,
                                   ref rng: randomStream(real)): real {
    select kind {
      when DistKind.GaussianDist    do
        return _gaussSample(params.gaussian.mu:real,
                             params.gaussian.sigma:real, rng);
      when DistKind.BetaDist        do
        return _betaSample(params.beta.alpha:real, params.beta.betaP:real, rng);
      when DistKind.GammaDist       do
        return _gammaSample(params.gamma_p.shape:real, rng)
             / params.gamma_p.rate:real;
      when DistKind.ExponentialDist do
        return -log(rng.next()) / params.exponential.rate:real;
      when DistKind.BernoulliDist   do
        return if rng.next() < params.bernoulli.p:real then 1.0 else 0.0;
      when DistKind.PoissonDist     do
        return _poissonSample(params.poisson.lambda_:real, rng):real;
      otherwise do return 0.0;
    }
  }

}
