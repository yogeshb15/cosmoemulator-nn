# CosmoEmulator-NN

**A neural-network emulator for dark-energy distance moduli — trading exact numerical integration of the Friedmann equation for a trained surrogate, benchmarked for accuracy and speed.**

## Motivation

Every likelihood evaluation inside an MCMC (or nested-sampling) run for dark-energy parameter estimation requires numerically re-integrating the Friedmann equation

```
E(z)^2 = Ωm(1+z)^3 + (1-Ωm)(1+z)^{3(1+w0+wa)} exp(-3·wa·z/(1+z))
```

for every trial parameter set. For chains of 10^5–10^6 steps this integration dominates run time. This project trains a feed-forward network to emulate the distance modulus μ(z; Ωm, w0, wa) directly for the **CPL (Chevallier–Polarski–Linder)** dark-energy parametrization — the same idea behind production tools like `CosmoPower` used in Planck/DES/LSST-era analyses, reproduced here at a small, fully reproducible scale.

## Method

1. Sample 1,200 random dark-energy models (Ωm, w0, wa) from broad priors and numerically integrate the exact distance modulus at 40 redshifts spanning z ∈ [0.01, 2.3] (the observed SN Ia range) → 48,000 training points.
2. Train a 3-layer MLP (64-64-32, tanh) to map (Ωm, w0, wa, z) → μ.
3. Validate on 250 **held-out, unseen** dark-energy models (10,000 test points).
4. Benchmark wall-clock time: numerical integration vs. trained emulator over 3,000 likelihood-style evaluations.

## Results

| Metric | Value |
|---|---|
| Held-out RMS residual | **14.3 mmag** (vs. ~100–150 mmag typical SN Ia measurement error) |
| Held-out max residual | 182 mmag (rare edge-of-prior case) |
| Speed-up | **~35×** over 3,000 evaluations |

![Emulator accuracy](results/emulator_accuracy.png)
![Speed benchmark](results/speed_benchmark.png)

## Why it matters

The emulator's error is roughly an order of magnitude below the observational noise floor, meaning it can safely replace the exact integration inside an MCMC likelihood without measurably degrading posterior constraints — while cutting the per-step cost substantially. The larger the chain (or the more expensive the exact integral, e.g. for more complex modified-gravity models), the more this compounds.

## Run it

```bash
pip install -r requirements.txt
python cosmoemulator.py
```

## Stack

Python · NumPy · SciPy · scikit-learn (MLPRegressor) · Matplotlib

## License

All rights reserved — see [LICENSE](LICENSE). This repository is shared publicly to demonstrate the work; it is not open source, and no use (including research or academic use) is permitted without written permission.

---
*Part of a cosmology + ML portfolio by Yogesh Bhardwaj — PhD (Applied Mathematics), Delhi Technological University. [LinkedIn](https://www.linkedin.com/in/yogesh-bhardwaj-23069120a/) · [GitHub](https://github.com/yogeshb15)*
