## Metadata
* **Topic 1:** Importance Weighted Autoencoders (IWAE)
* **Topic 2:** Annealed Importance Sampling (AIS)

## Notes
### Importance Weighted Autoencoders
* An **Importance Weighted Autoencoder** (IWAE) is a  generative model with the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound derived from importance
weighting.
    * The intractable expectation of interest is `log 𝔼[p(x|z) * p(z) / q(z|x)]` where `z ~ q(z|x)`.
        * `log p(x) = log ∫ p(x,z) dz`
        * `log p(x) = log ∫ p(x,z) * q(z|x) / q(z|x) dz`
        * `log p(x) = log 𝔼[p(x,z) / q(z|x)]`
        * `log p(x) = log 𝔼[p(x|z) * p(z) / q(z|x)]`
    * Approximate `log p(x)` naïvely using `𝓛ₖ(x) = 𝔼ₖ [log (1/k * Σᵢ p(x, zᵢ) / q(zᵢ|x))]`.
        * Above, the expectation is taken over `z₁,...,zₖ ~ q(z|x)`.
        * The ELBO is equivalent to `𝓛₁`.
    * The estimator `𝓛ₖ(x)` has some desirable statistical properties:
        * `𝓛ₖ(x)` is consistent and so eventually converges to `log p(x)`.
        * The bias vanishes at a rate that is inversely proportional to `k`.
        * The variance vanishes at a rate that is inversely proportional to `k`.
    * By Jensen's inequality, `𝓛ₖ(x) <= log p(x)`.
        * `𝓛ₖ(x) = 𝔼ₖ [log (1/k * Σᵢ p(x,zᵢ) / q(zᵢ|x))]`.
        * `𝓛ₖ(x) <= log 𝔼ₖ [(1/k * Σᵢ p(x,zᵢ) / q(zᵢ|x))]`.
        * `𝓛ₖ(x) <= log (1/k * 𝔼ₖ [Σᵢ p(x,zᵢ) / q(zᵢ|x)])`.
        * `𝓛ₖ(x) <= log (1/k * k * 𝔼 [p(x,z) / q(z|x))`.
        * `𝓛ₖ(x) <= log 𝔼 [p(x,z) / q(z|x)`.
        * `𝓛ₖ(x) <= log p(x)`.

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
* [IWAE - Debiasing Evidence Approximations Video](https://www.youtube.com/watch?v=nRgjvACKNAQ)
* [AIS - Introductory Paper](https://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf)
* [AIS - Blog Tutorial](https://wiseodd.github.io/techblog/2017/12/23/annealed-importance-sampling/)