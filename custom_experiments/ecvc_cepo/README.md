# ECVC-CEPO experiment

This directory is an isolated copy of `custom_experiments/cepo_imgmask_teacher`.
It does not import or modify files from that experiment.

The original-image and full-black-mask teachers return both sampled-token log
probabilities and full-distribution token entropies. On the first training batch,
the initial policy calibrates `sigma_U_init` from valid response tokens in the
first 128 rollout samples and freezes it for the run. Training then applies:

`D = log P_vis(y_t) - log P_mask(y_t)`

`U_z = (H_mask - H_vis) / sigma_U_init`

`delta = sigmoid(alpha * (U_z - m_pos)) * D - gamma * sigmoid(alpha * (-U_z - m_neg)) * relu(-D)`

Activate the server's `cepo` conda environment before running. Run a one-step
smoke test with `MAX_STEPS=1`, then use the default 50-step run. The launcher
writes timestamped logs under `logs/`.
