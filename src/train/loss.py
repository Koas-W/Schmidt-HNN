"""Derivative-matching loss: the training loss is the MSE on (q_dot, p_dot), not a match on H values."""
import torch
import torch.nn.functional as F


def derivative_matching_loss(model, x: torch.Tensor, mu: torch.Tensor,
                             dx: torch.Tensor,
                             create_graph: bool = True) -> torch.Tensor:
    pred = model.time_derivative(x, mu, create_graph=create_graph)
    return F.mse_loss(pred, dx)
