# Method notes

## Contribution inputs remain distinct

Signed contribution signatures are leave-one-modality-out loss deltas. Negative
values remain signed and are used for contribution-profile grouping. The
InfoNCE estimates used for FCD are separate quantities and can also be
negative. They are converted scope-wise with Softplus before normalization
because signed lower bounds cannot directly define probability vectors. This
conversion is a differentiable implementation choice; it changes the scale and
is not specified by the Utopia paper.

The FCD estimator uses the positive paired sample, other eligible training
samples in the minibatch, and a detached FIFO feature queue. Sample IDs are
excluded from their own negative sets. AVE also excludes same-source sibling
segments. Validation and test features never update the queue. Because the
supervised batch is small, the queue supplies most contrastive candidates.

## Target granularity

The training objective supports:

- `global`: one target estimated from the designated training-only estimation
  partition.
- `class`: a target per class, with unsupported or unstable classes falling
  back to their parent target.
- `group`: a target per class/profile group, with fallback to the class target.
- `reliability`: choose a stable class/group target at the finest supported
  level; otherwise use the global target. Stability and fallback flags come
  from training-only bootstrap and support checks.

Granularity describes how the precomputed target is selected for each training
scope. It does not change signed signature estimation or the FCD formula.
Validation is used to select checkpoints and hyperparameters. Test examples are
not used to fit, normalize, cluster, calibrate, or select targets.

## Diagnostics and current limitation

Joint-training logs record target and current FCD proportions, raw MI lower
bounds, the fraction of negative bounds, alignment and classification losses,
weighted alignment loss, queue size, and periodic classification/alignment
gradient norms and cosine. Validation logs include overall and training-count-
derived minority Macro-F1 and recall.

On the current natural AVE seed-23 run, the global UCD target was about 82/18
while the estimated FCD started near 51/49 and moved toward the target as
training continued. The alignment gradient remained much smaller than the
classification gradient for the tested weights; larger weights moved the
proportions closer but reduced validation Macro-F1. This points to a mismatch
between target proportions and the current estimator/optimization scale, but
one seed does not identify a causal fix. The saved epoch-level diagnostics and
matched seed experiments should guide further changes.

## Five-seed AVE findings

The completed matched official-split seeds are 17, 23, 42, 73, and 123 for
natural AVE, synthetic IF=10, and synthetic IF=20. The selected lambda is
0.1 for natural AVE and IF=10, and 0.03 for IF=20. For each synthetic
condition, the lambda was selected on seed-23 validation Macro-F1 from
`{0.03, 0.1, 0.3}` and then held fixed across the matched seeds. Test values
did not inform this selection.

The proposed CI+MI method has no consistent overall Macro-F1 advantage over
CI-only across these five seeds. On IF=20 it raises mean minority Macro-F1
and minority recall compared with CI-only, while mean overall test Macro-F1
is effectively tied; IF=10 shows a minority regression. These results support
an imbalance-dependent minority tradeoff for further study, not a general
performance claim. Exact paired differences and sample standard deviations
are in `results/tables/paired_differences.csv` and `results/summary.md`.

## Sparse estimation classes and optimizer ablations

At IF=20, a rare class may be absent from the training-only contribution
estimation partition even though it remains in the retained training set. In
that case its samples receive a generic group assignment, the class target
falls back to the global target, and the fallback is recorded in
`profile_groups.json` and `ucd_targets.json`. Its target is not fitted from
reference-fit samples.

The seed-23 natural granularity ablation found that class, group, and
reliability targets reduce to the same effective targets for this estimation
split: supported groups are singletons per class, and unsupported scopes
inherit their parent target. This finding is specific to this support pattern.

Raw minimum-norm optimization is scale-sensitive. In the natural seed-23
diagnostic, the classification gradient norm was over three orders of
magnitude larger than the MI alignment gradient norm. The raw convex
minimum-norm combination almost entirely suppressed classification gradients
and validation performance collapsed. The normalized-MOO sensitivity run
kept both objective gradients at unit norm before selecting the convex
coefficient and performed substantially better than raw MOO, but remained
below the fixed-weight model on validation. It is an exploratory diagnostic,
not the selected training method.
