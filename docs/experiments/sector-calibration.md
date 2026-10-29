# Training-only sector-weight calibration

29 October 2026. Follow-up to the [completed larger study](segmented-large.md).
The fixed 50/50 recent/distant preparation could constrain the sector-mixture
model. This inexpensive diagnostic fits one scalar preparation probability per
seed while freezing all selected Hamiltonian parameters. It does not retrain
the quantum core or introduce a classical word decoder.

## Protocol fixed before scoring

For contexts with older words, prepare the recent sector with probability w and
the distant sector with probability 1−w, then apply the existing Hamiltonians
and directly measure the word register. Within each sector the original coherent
amplitude state remains unchanged. Contexts with no older words always prepare
the recent sector. This is a randomized input preparation; no reconstruction of
the output distribution is required for physical sampling.

For each of the three completed mixture seeds, calculate the correct-target
probability from each branch for all 131,072 training examples. Minimize training
NLL over w in [1e-6, 1−1e-6] using 60 bisection steps on its monotone derivative.
The quantum weights were previously selected by validation NLL; the scalar is
fit on training targets only and written to metadata before validation scoring.
The scalar is shared across all contexts and words. No validation grid or test
scoring is used. Existing checkpoints remain unchanged.

`output/segmented-large/calibrate_sectors.py` completed all three seeds in
approximately 105 seconds. Metadata, hashes, training branch probabilities and
validation arrays are in `output/segmented-large/sector-calibration/`. The 50/50
implementation reproduces all original validation arrays within atol/rtol 1e-12,
including short-context handling. All scored distributions pass normalization
checks. Training NLL decreases for every seed.

## Results

Training-fitted recent probabilities are .51603, .51488 and .51363 for seeds
11, 22 and 33. This is close to the existing equal allocation.

| Three-seed mean | Original 50/50 | Training-fitted allocation |
| --- | ---: | ---: |
| Validation NLL | 3.488359 | 3.487537 |
| Sampling success | 12.2600% | 12.2408% |
| Known-target NLL | 4.925706 | 4.922054 |
| Known-target sampling | 2.3647% | 2.3812% |

Fitted minus original NLL is −.000821, descriptive paired article-group 95%
interval [−.001390, −.000222]. Known-target NLL improves by .003652 and
known-target sampling by .01649 percentage points. Aggregate sampling decreases
by .01922 points. The optimization criterion was NLL, not sampling success.

Full [fit metrics](sector-calibration-fits.csv) and
[paired differences](sector-calibration-paired.csv) are versioned. Intervals
average the three seeds before resampling articles and retain the adaptive
validation limitations of the parent study. These tiny gains do not close the
MLP gap (NLL 3.1641, sampling 22.09%). They do not support another long run merely
to tune this global allocation. Jointly learning preparation and dynamics would
be a different experiment; this frozen-core diagnostic does not rule it out.
