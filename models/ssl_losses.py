import torch
import torch.nn.functional as F


def _off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected a square matrix, got {tuple(x.shape)}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    x,
    y,
    sim_coeff=25.0,
    std_coeff=25.0,
    cov_coeff=1.0,
    variance_margin=1.0,
    eps=1e-4,
):
    x, y = x.float(), y.float()
    if x.shape != y.shape:
        raise ValueError(f"VICReg views differ in shape: {tuple(x.shape)} vs {tuple(y.shape)}")

    invariance = F.mse_loss(x, y)
    if x.shape[0] <= 1:
        total = sim_coeff * invariance
        zero = invariance.new_zeros(())
        return total, {"invariance": invariance, "variance": zero, "covariance": zero}

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + eps)
    std_y = torch.sqrt(y.var(dim=0) + eps)
    variance = 0.5 * (
        F.relu(variance_margin - std_x).mean()
        + F.relu(variance_margin - std_y).mean()
    )

    covariance_x = x.T @ x / (x.shape[0] - 1)
    covariance_y = y.T @ y / (y.shape[0] - 1)
    covariance = (
        _off_diagonal(covariance_x).square().sum()
        + _off_diagonal(covariance_y).square().sum()
    ) / x.shape[1]

    total = sim_coeff * invariance + std_coeff * variance + cov_coeff * covariance
    return total, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }
