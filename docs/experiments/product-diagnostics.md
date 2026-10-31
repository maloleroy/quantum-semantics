# Product checkpoint distribution diagnostics

31 October 2026. Read-only validation inspection after the negative sequential
screen. No training, new samples or test scoring. The expanded product and matched
MLP checkpoints were loaded with `selected_models` under their exact archived
source from commit `b4527ba`; both regenerated all saved validation scores to 1e-12.
The generated tool is `output/product-diagnostics/diagnose.py`, executed as a module
from that archive directory with the project environment. Source locks were retained.

On the 4,096 validation targets, product NLL is 2.9673 and MLP NLL 2.7966. Product
predictive entropy averages 3.4901 nats versus MLP 2.4877; mean maximum probability
is .3203 versus .4258. These are descriptive output statistics, not a calibration
fit or permission to temperature-transform the quantum measurement distribution.

The most common training target accounts for 1,231 validation targets (30.05%).
On the other 2,865 targets, product sampling success is **4.19%** versus MLP
**10.85%**, and argmax is **10.23%** versus **18.46%**. The overall sampling score
therefore conceals a substantial deficit beyond the dominant token. Product
predicts the dominant token by argmax on 28.52% of those other-target examples,
versus MLP 12.60%. Frequency buckets are derived only from all-training counts;
the rarest bucket contains only 11 validation examples and is too small for strong
conclusions. No frequency-specific model selection occurred.

Mean probability assigned to outcomes with zero training-target count is 1.0824%
for product, versus .0000741% for MLP. This can hurt predictions and text validity,
but is too small to explain most of the 7.67-point overall sampling gap. No absent
outcomes were removed or renormalized; the original byte vocabulary and UNK policy
remain intact.

For each Hamiltonian, measured incoming state mass in its learned rank-64 subspace
is 26.63% and 25.51% on average (medians 25.97% and 27.48%). The remaining orthogonal
component is unchanged by that individual spectral block. Interleaving changes the
state before the next block, and learned bases are adaptive: these values do not
prove that doubling rank improves prediction, establish a hard success ceiling,
or identify unused parameters. They motivate a bounded capacity comparison.

The next manual comparison returns to the product family and tests ranks 32/128
against the completed rank-64 reference, with a matched MLP at each new budget.
It distinguishes a rank/capacity response from a generic benefit of giving either
model more parameters. It cannot alone prove quantum advantage or solve generation.

Evidence: [distribution buckets](product-distribution-diagnostics.csv),
[subspace masses](product-subspace-diagnostics.csv).
