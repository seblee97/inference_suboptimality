## Metadata
* **Topic 1:** Importance Weighted Autoencoders (IWAE)
* **Topic 2:** Annealed Importance Sampling (AIS)

## Notes
### Importance Weighted Autoencoders
* An **Importance Weighted Autoencoder** (IWAE) is a generative model with the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound derived from importance weighting.
    * The goal of a [VAE](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) is to maximize `log p(x) = log 𝔼 [p(x|z) * p(z) / q(z|x)]` where `z ~ q(z|x)`.
        * This expression for the log-likelihood can be derived as follows:
            ```
            log p(x) = log ∫ p(x,z) dz
                     = log ∫ p(x,z) * q(z|x) / q(z|x) dz
                     = log 𝔼 [p(x,z) / q(z|x)]
                     = log 𝔼 [p(x|z) * p(z) / q(z|x)]
            ```
        * Computing `log p(x)` directly is intractable because of the integration.
    * The ELBO serves as a lower bound on `log p(x)` by [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality).
        * Traditionally, `𝓛(x) = 𝔼 [log p(x,z) / q(z|x))]` where `z ~ q(z|x)`.
        * Maximizing the ELBO is equivalent to maximizing `log p(x)`.
    * An IWAE reduces the gap between `log p(x)` and the ELBO by taking the latent expectation over *multiple samples* instead of just one sample.
        * Define `𝓛ₖ(x) = 𝔼ₖ [log (1/k * Σᵢ p(x,zᵢ) / q(zᵢ|x))]` where `z₁,...,zₖ ~ q(z|x)`.
        * The traditional ELBO is recovered when `k = 1`.
    * Crucially, the estimator `𝓛ₖ(x)` has some desirable statistical properties:
        * `𝓛ₖ(x)` is consistent and eventually converges to `log p(x)`.
            * As a consequence of the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers),
                ```
                𝓛ₖ(x) = 𝔼ₖ [log (1/k * Σᵢ p(x,zᵢ) / q(zᵢ|x))]
                      = 𝔼ₖ [log 𝔼 [p(x,z) / q(z|x)]]           // k → ∞
                      = 𝔼ₖ [log p(x)]                          // Definition
                      = log p(x)                               // Constant
                ```
            * Additionally, `𝓛₁(x) <= 𝓛₂(x) <= ... <= 𝓛ₖ(x) <= ... <= log p(x)`.
            * This implies that `𝓛ₖ(x)` approaches `log p(x)` as `k → ∞`
        * The bias of `𝓛ₖ(x)` vanishes at a rate that is inversely proportional to `k`.
        * The variance of `𝓛ₖ(x)` vanishes at a rate that is inversely proportional to `k`.

### Annealed Importance Sampling
* **Annealed Importance Sampling** (AIS) uses Markov chain transitions for an annealing sequence to define an importance sampler.
    * The Markov chain aspect ensures the method performs well in high dimensions.
    * The importance weights ensure that the estimates converge to the correct values.
* The immediate goal of annealed importance sampling is to generate a sample from a target distribution.
    * Let `p(x) = p₀(x) ∝ f₀(x)` be the target distribution.
    * Let `q(x) = pₙ(x) ∝ fₙ(x)` be the proposal distribution.
    * Let `pᵢ(x)` be an intermediate distributions for each `i` in `{1,...,n - 1}`.
    * Let `Tⱼ(x, x')` be a Markov chain transition probability from `x` to `x'`.
    * Now, sample an independent point `xₙ₋₁ ~ q(x)`.  Then, sample `xₙ₋₂` from `xₙ₋₁` using `Tₙ₋₁`.  Repeat this process until `x₀` is sampled from `x₁` using `T₁`.
* The importance weight `w` of a sample is given by `w = Πᵢ [fᵢ₋₁(xᵢ₋₁)/fᵢ(xᵢ₋₁)]`.
        * Naturally, `𝔼[x] = (Σᵢ xᵢ*wᵢ) / (Σᵢ wᵢ)`.
* The intermediate distributions are usually taken to be `fᵢ(x) = f₀(x)ᵝ⁽ⁱ⁾ * fₙ(x)¹⁻ᵝ⁽ⁱ⁾`.
    * Above, `1 = β₀ > ... > βₙ = 0`.
* The Markov chain transitions may be constructed using [Metropolis-Hastings sampling](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) or [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) updates.


## References
* [IWAE - Introductory Paper](https://arxiv.org/pdf/1509.00519.pdf)
* [IWAE - Bias and Variance](https://openreview.net/pdf?id=HyZoi-WRb)
* [IWAE - Debiasing Evidence Approximations Video](https://www.youtube.com/watch?v=nRgjvACKNAQ)
* [AIS - Introductory Paper](https://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf)
* [AIS - Blog Tutorial](https://wiseodd.github.io/techblog/2017/12/23/annealed-importance-sampling/)