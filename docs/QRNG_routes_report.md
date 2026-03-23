# QRNG Routes Progress Report

This report summarizes the second-round investigation of the three candidate routes:

1. Route 1: steering-based QRNG following Phys. Rev. A 106, 042414 (2022).
2. Route 2: correct single-device prepare-and-measure MDI QRNG.
3. Route 3: CV hardware with trusted coherent inputs and a single-device MDI analysis.

All prototype code used in this round lives under [src/python/qrng_routes](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes).

## Scope

The goals of this round were:

- make Route 1 closer to the 2022 steering paper by increasing the cutoff and the number of Bob settings,
- push Route 3 beyond the original four-phase trusted-input prototype,
- determine whether Route 2 can exceed `H_min = 2`,
- leave the repo with saved data, figures, and a written recommendation.

Generated assets are in [output/qrng_routes](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes).

## Route 1

### What changed

Route 1 was extended in [steering_2022.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route1/steering_2022.py):

- Bob's trusted homodyne settings are no longer fixed to `m_B = 2`.
- We now support `m_B = 2, 4, 6` with angles evenly spaced between `q` and `p`.
- We added `sweep_route1_eta(...)` for parameter scans.

For the stronger TMSV scan, I used:

- source: TMSV,
- cutoff: `d = 5`,
- squeezing: `-4 dB`,
- Alice bins: `o_A = 6`,
- Bob bins: `o_B = 8`,
- `T_q = 4.0`.

`T_q = 4.0` was chosen after a preliminary scan over `{2, 4, 6}`; in the tested TMSV points it remained the best value.

### Main TMSV results

Saved figure:

- [route1_tmsv_sweep.png](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_sweep.png)

Saved data:

- [route1_tmsv_sweep.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_sweep.json)

Key values:

| eta | tomography | homodyne `m_B=2` | homodyne `m_B=4` | homodyne `m_B=6` |
|---|---:|---:|---:|---:|
| 0.80 | 0.384 | 0.066 | 0.333 | 0.362 |
| 0.85 | 0.526 | 0.128 | 0.450 | 0.495 |
| 0.90 | 0.705 | 0.205 | 0.603 | 0.668 |
| 0.95 | 0.929 | 0.313 | 0.817 | 0.895 |
| 1.00 | 1.177 | 0.493 | 1.177 | 1.177 |

Interpretation:

- This is much closer to the 2022 paper's behavior than the first-round tiny prototype.
- Increasing Bob's trusted settings helps a lot: at `eta = 0.90`, `m_B = 6` raises `H_min` from about `0.205` to about `0.668`.
- In this truncated model, `m_B = 6` is already close to the tomography upper bound across the scan.
- This makes Route 1 a realistic candidate for a lab-facing protocol in the `0.5 - 1 bit/round` regime.

### Split-SMS spot check

Saved data:

- [route1_split_sms_spotcheck.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_split_sms_spotcheck.json)

Spot-check values:

| source | eta | tomography | homodyne `m_B=2` |
|---|---:|---:|---:|
| split-SMS | 0.80 | 0.121 | 0.008 |
| split-SMS | 0.90 | 0.295 | 0.048 |
| split-SMS | 1.00 | 1.155 | 0.395 |

Interpretation:

- In this implementation, split-SMS remains weaker than TMSV under the same rough resource level.
- It is still viable, but TMSV currently looks like the better Route 1 baseline for the lab.

### Route 1 conclusion

If the lab wants the most practical near-term path, Route 1 now looks credible:

- `TMSV + trusted homodyne with m_B = 6` already reaches `H_min ≈ 0.67` at `eta = 0.90`,
- `TMSV + tomography` reaches `H_min ≈ 0.71` at the same point.

This is the route I would send to the experimental team first.

## Route 3

### What changed

Route 3 was generalized in [cv_four_phase.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route3/cv_four_phase.py):

- the trusted alphabet is no longer fixed to four phases,
- `run_route3(...)` now accepts `num_phases`,
- the code keeps the exact support of the chosen coherent-state alphabet,
- the same single-device MDI SDP is reused.

### Phase-count sweep

Saved figure:

- [route3_phase_sweep.png](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_phase_sweep.png)

Saved data:

- [route3_phase_sweep.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_phase_sweep.json)

For `mu = 0.05`, `cutoff = 12`, and quadrant outputs:

| trusted phases per side | certified `H_min` | local rank | joint dim | operator span rank | operator space dim |
|---|---:|---:|---:|---:|---:|
| 4 | 0.547 | 4 | 16 | 16 | 256 |
| 5 | 0.589 | 5 | 25 | 25 | 625 |
| 6 | 0.634 | 6 | 36 | 36 | 1296 |

Interpretation:

- Increasing the trusted input set does help.
- The gain from `4 -> 6` phases is real: `0.547 -> 0.634`.
- But the core bottleneck remains: the input operator span grows only like the number of product inputs, while the full operator space grows quadratically in the joint Hilbert-space dimension.
- Even at 6 phases, we have `36` input operators trying to constrain an operator space of size `1296`.

### Small `mu` sweep at 6 phases

Saved data:

- [route3_mu_sweep_6phase.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_mu_sweep_6phase.json)

Key values:

| `mu` | certified `H_min` |
|---|---:|
| 0.02 | 0.645 |
| 0.05 | 0.634 |
| 0.10 | 0.619 |

Interpretation:

- The current 6-phase prototype prefers small amplitudes.
- The best point I found in this scan is around `mu = 0.02 - 0.05`.

### Route 3 conclusion

Route 3 is better than it was in round 1:

- the original 4-phase point was about `0.55 bits`,
- the 6-phase version reaches about `0.64 bits`.

But it is still noticeably behind the strongest Route 1 TMSV points and far behind Route 2. The reason is structural: the trusted input set is still far from informationally complete over the effective joint operator space.

## Route 2

### Baseline 4-output result

The current baseline Route 2 prototype is in [mdi_single_device.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route2/mdi_single_device.py).

Using:

- 16 informationally complete product inputs,
- a 4-output extremal POVM on the 4-dimensional joint support,

the code gives:

- `p_guess ≈ 0.25023`,
- `H_min ≈ 1.99866`.

So the baseline route already essentially saturates 2 bits.

### Can Route 2 exceed 2 bits?

Short answer: yes, but not with only 4 outcomes.

Reason:

- If the measurement has `o` possible outputs, then always `p_guess >= 1 / o`.
- Therefore `H_min <= log2(o)`.
- With 4 outcomes, the absolute ceiling is exactly 2 bits.

So under the current 4-output Route 2 prototype, `H_min > 2` is impossible.

However, in dimension `d = 4`, extremal POVMs can have up to `d^2 = 16` outcomes. That means the dimension-based ceiling is 4 bits, not 2 bits.

To test whether this is only a formal possibility or a real one, I added a random higher-output POVM search:

- [search_route2_high_entropy(...)](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route2/mdi_single_device.py)

Saved data:

- [route2_high_output_search.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route2_high_output_search.json)

Certified results from a 20-trial random search:

| outputs | certified `H_min` |
|---|---:|
| 8 | 2.539 |
| 12 | 2.917 |
| 16 | 3.217 |

Interpretation:

- Route 2 can definitely exceed 2 bits.
- The obstruction is not the MDI framework itself.
- The obstruction is only the current 4-output design.

### Route 2 conclusion

If the lab is willing to implement a higher-output central measurement, Route 2 is the only route in this project that already shows a clear path past 2 bits. In fact, the numerical search suggests that going well beyond 2 bits is realistic within the same 4-dimensional support.

## Overall recommendation

For the experimental team, I would frame the routes this way:

1. Route 1 is the safest near-term experimental target.
   It now has a stable parameter picture and already reaches the desired `0.5 - 1 bit/round` regime.

2. Route 2 is the strongest high-entropy route.
   It already reaches 2 bits with 4 outputs and exceeds 2 bits once the central measurement is allowed to have more outputs.

3. Route 3 is promising as a CV-hardware compromise.
   It improves with more trusted phases, but it is still fundamentally limited by poor informational completeness of the current trusted-input family.

My practical recommendation is:

- hand Route 1 to the lab as the immediate experimental candidate,
- keep Route 2 as the long-term high-entropy target,
- treat Route 3 as an exploratory CV bridge route, worth studying further but not yet the best candidate.

## Files

Code:

- [common.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/common.py)
- [steering_2022.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route1/steering_2022.py)
- [mdi_single_device.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route2/mdi_single_device.py)
- [cv_four_phase.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/route3/cv_four_phase.py)
- [main.py](/d:/Codes/Quantum/Randomness-certification/src/python/qrng_routes/main.py)

Data and figures:

- [route1_tmsv_sweep.png](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_sweep.png)
- [route1_tmsv_sweep.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_tmsv_sweep.json)
- [route1_split_sms_spotcheck.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route1_split_sms_spotcheck.json)
- [route3_phase_sweep.png](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_phase_sweep.png)
- [route3_phase_sweep.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_phase_sweep.json)
- [route3_mu_sweep_6phase.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route3_mu_sweep_6phase.json)
- [route2_high_output_search.json](/d:/Codes/Quantum/Randomness-certification/output/qrng_routes/route2_high_output_search.json)
