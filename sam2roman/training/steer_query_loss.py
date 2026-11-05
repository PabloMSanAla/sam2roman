"""Re-export SteerQuery loss functions into sam2roman.training namespace."""

from sam2roman.losses.steer_query_loss import (
    SteerQueryLoss,
    build_steer_query_optimizer,
    render_star_center_heatmap,
    compute_spike_over_galaxy_weights,
)

__all__ = [
    "SteerQueryLoss",
    "build_steer_query_optimizer",
    "render_star_center_heatmap",
    "compute_spike_over_galaxy_weights",
]
