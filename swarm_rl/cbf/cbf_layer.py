import math
from typing import Dict

import torch
from qpth.qp import QPFunction
from torch import Tensor, nn


class DistanceAwareCBFLayer(nn.Module):
    """
    Distance-aware attitude CBF layer for omega-thrust control.

    The policy and environment both operate in normalized action space `[-1, 1]^4`.
    This layer therefore solves the QP in normalized omega space and only uses
    physical angular-rate limits to convert the CBF constraint coefficients.
    """

    def __init__(self, alpha_cbf: float = 1.0, k: float = 2.0, sigma: float = 0.1, sdf_resolution: float = 0.1):
        super().__init__()
        self.alpha_cbf = alpha_cbf
        self.k = k
        self.sigma = sigma
        self.sdf_resolution = sdf_resolution
        self.qp = QPFunction(eps=1e-9, verbose=0, maxIter=20)

        # Must stay aligned with gym_art.quadrotor_multi.quadrotor_control.OmegaThrustControl.
        self.register_buffer(
            "omega_scale",
            torch.tensor([5.0 * 2.0 * math.pi, 5.0 * 2.0 * math.pi, 1.0 * 2.0 * math.pi], dtype=torch.float32),
        )

    def compute_alpha(self, sdf_obs: Tensor) -> Tensor:
        """
        Distance weight α(p) = exp(-k p), saturated to (0, 1] for signed-distance inputs.
        """
        signed_distance = sdf_obs[:, 4:5]
        distance = torch.clamp_min(signed_distance, 0.0)
        return torch.exp(-self.k * distance)

    def compute_sdf_gradient(self, sdf_obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Estimate the 2D SDF gradient from the 3x3 local grid.

        Grid indexing follows gym_art.quadrotor_multi.obstacles.utils.get_surround_sdfs:
            [0, 1, 2] -> x - resolution
            [3, 4, 5] -> x
            [6, 7, 8] -> x + resolution
        with y increasing along the second index.
        """
        batch_size = sdf_obs.shape[0]
        resolution = self.sdf_resolution

        center_sdf = sdf_obs[:, 4:5]
        grad_x = (sdf_obs[:, 7] - sdf_obs[:, 1]) / (2.0 * resolution)
        grad_y = (sdf_obs[:, 5] - sdf_obs[:, 3]) / (2.0 * resolution)
        grad_z = torch.zeros(batch_size, device=sdf_obs.device, dtype=sdf_obs.dtype)

        grad = torch.stack([grad_x, grad_y, grad_z], dim=1)
        grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True).clamp_min(1e-6)
        normal = grad / grad_norm

        return normal, center_sdf

    @staticmethod
    def compute_n_dot(n: Tensor, Re3: Tensor, state: Dict[str, Tensor]) -> Tensor:
        # A full Hessian-based derivative is not available from the 3x3 SDF patch.
        return torch.zeros(n.shape[0], 1, device=n.device, dtype=n.dtype)

    @staticmethod
    def _cap_rhs_to_box(A_normalized: Tensor, b: Tensor) -> Tensor:
        """
        Relax the constraint to the actuator-feasible boundary when the requested
        rhs is outside the normalized omega box.
        """
        max_feasible_rhs = torch.sum(torch.abs(A_normalized), dim=1, keepdim=True)
        return torch.minimum(b, max_feasible_rhs - 1e-4)

    @staticmethod
    def _constraint_stats(omega: Tensor, A_normalized: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lhs = torch.sum(A_normalized * omega, dim=1, keepdim=True)
        margin = lhs - b
        violation = torch.clamp_min(-margin, 0.0)
        active_mask = (violation > 1e-6).squeeze(1)
        return lhs, margin, violation, active_mask

    def _constraint_terms(
        self,
        rl_output: Tensor,
        state: Dict[str, Tensor],
        sdf_obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if rl_output.ndim != 2 or rl_output.shape[1] != 4:
            raise ValueError(f"Expected rl_output with shape [batch, 4], got {tuple(rl_output.shape)}")

        device = rl_output.device
        dtype = rl_output.dtype
        batch_size = rl_output.shape[0]

        sdf_obs = sdf_obs.to(device=device, dtype=dtype)
        R = state["R"].to(device=device, dtype=dtype)
        vel = state["vel"].to(device=device, dtype=dtype)
        omega_scale = self.omega_scale.to(device=device, dtype=dtype)

        normal, _ = self.compute_sdf_gradient(sdf_obs)
        Re3 = R[:, :, 2]
        alpha = self.compute_alpha(sdf_obs).to(dtype=dtype)

        n_dot_Re3 = torch.sum(normal * Re3, dim=1, keepdim=True)
        n_dot_v = torch.sum(normal * vel, dim=1, keepdim=True)
        n_dot_Re3_dot = self.compute_n_dot(normal, Re3, state)

        n_cross_Re3 = torch.cross(normal, Re3, dim=1)
        A_physical = -alpha * n_cross_Re3
        b = (
            alpha * self.k * n_dot_v * n_dot_Re3
            - alpha * n_dot_Re3_dot
            - self.alpha_cbf * alpha * n_dot_Re3
            - self.alpha_cbf * self.sigma
        )

        a_thrust = torch.clamp(rl_output[:, 0:1], min=-1.0, max=1.0)
        omega_rl = torch.clamp(rl_output[:, 1:4], min=-1.0, max=1.0)

        A_normalized = A_physical * omega_scale.unsqueeze(0)
        b = self._cap_rhs_to_box(A_normalized, b)
        return a_thrust, omega_rl, A_normalized, b

    def project_with_info(self, rl_output: Tensor, state: Dict[str, Tensor], sdf_obs: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        a_thrust, omega_rl, A_normalized, b = self._constraint_terms(rl_output, state, sdf_obs)
        device = rl_output.device
        dtype = rl_output.dtype

        lhs_nominal, margin_nominal, violation_nominal, active_mask = self._constraint_stats(omega_rl, A_normalized, b)
        omega_safe = omega_rl.clone()
        qp_failed = torch.zeros((), device=device, dtype=torch.bool)

        if active_mask.any():
            active_count = int(active_mask.sum().item())
            eye = torch.eye(3, device=device, dtype=dtype)
            eye_batch = eye.unsqueeze(0).expand(active_count, -1, -1).contiguous()
            ones = torch.ones(active_count, 3, device=device, dtype=dtype)

            Q = (2.0 * eye).unsqueeze(0).expand(active_count, -1, -1).contiguous()
            p = -2.0 * omega_rl[active_mask]

            G = torch.cat(
                [
                    (-A_normalized[active_mask]).unsqueeze(1),
                    -eye_batch,
                    eye_batch,
                ],
                dim=1,
            )
            h_qp = torch.cat(
                [
                    -b[active_mask],
                    ones,
                    ones,
                ],
                dim=1,
            )

            A_eq = torch.empty(active_count, 0, 3, device=device, dtype=dtype)
            b_eq = torch.empty(active_count, 0, device=device, dtype=dtype)

            try:
                omega_safe_active = self.qp(Q, p, G, h_qp, A_eq, b_eq)
                if torch.isfinite(omega_safe_active).all():
                    omega_safe[active_mask] = omega_safe_active
                else:
                    qp_failed = torch.ones((), device=device, dtype=torch.bool)
            except Exception:
                qp_failed = torch.ones((), device=device, dtype=torch.bool)

        omega_safe = torch.clamp(omega_safe, min=-1.0, max=1.0)
        lhs_safe, margin_safe, violation_safe, _ = self._constraint_stats(omega_safe, A_normalized, b)

        info = {
            "A_normalized": A_normalized,
            "b": b,
            "lhs_nominal": lhs_nominal,
            "margin_nominal": margin_nominal,
            "violation_nominal": violation_nominal,
            "lhs_safe": lhs_safe,
            "margin_safe": margin_safe,
            "violation_safe": violation_safe,
            "active_mask": active_mask,
            "qp_failed": qp_failed,
        }
        return torch.cat([a_thrust, omega_safe], dim=1), info

    def forward(self, rl_output: Tensor, state: Dict[str, Tensor], sdf_obs: Tensor) -> Tensor:
        safe_action, _ = self.project_with_info(rl_output, state, sdf_obs)
        return safe_action
