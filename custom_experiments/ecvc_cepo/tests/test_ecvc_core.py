import torch

from verl.trainer.core_algos import apply_ecvc_cepo_weighting


def test_ecvc_gates_visual_support_and_conflict():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mask = torch.ones_like(advantages)
    vis_lp = torch.tensor([[-0.1, -1.0], [-1.0, -2.0]])
    mask_lp = torch.tensor([[-1.1, -1.0], [-1.0, -1.0]])
    vis_h = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
    mask_h = torch.tensor([[2.0, 1.0], [1.0, 1.0]])

    weighted, diag = apply_ecvc_cepo_weighting(
        advantages=advantages,
        response_mask=mask,
        vis_teacher_log_probs=vis_lp,
        mask_teacher_log_probs=mask_lp,
        vis_teacher_entropies=vis_h,
        mask_teacher_entropies=mask_h,
        entropy_gap_std=1.0,
        lam=0.5,
        eps_w=0.5,
        alpha=5.0,
        margin_pos_z=0.5,
        margin_neg_z=0.5,
        gamma=0.5,
    )

    assert diag["delta"][0, 0] > 0
    assert diag["delta"][1, 1] < 0
    assert weighted[0, 0] > advantages[0, 0]
    assert weighted[1, 1].abs() > advantages[1, 1].abs()
    assert torch.isclose(diag["delta"][0, 1], torch.tensor(0.0), atol=1e-6)
