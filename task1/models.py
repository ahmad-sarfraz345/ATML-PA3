from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class LoRALinear(nn.Module):
    """
    Standard Low-Rank Adaptation (LoRA) layer wrapping a frozen linear base weight.
    Effective weight: W = W0 + (alpha / r) * B * A
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA trainable matrices (FP32 master parameters for stability)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        # Initialise A with Kaiming uniform, B with zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Base linear computation in FP16 / backbone dtype
        out = self.base_layer(x)

        # LoRA branch in FP32 or converted to input dtype
        lora_in = self.lora_dropout(x).to(torch.float32)
        lora_out = (lora_in @ self.lora_A.T) @ self.lora_B.T
        out = out + (self.scaling * lora_out).to(orig_dtype)
        return out

    def get_delta_weight(self) -> torch.Tensor:
        """Returns delta W = (alpha / r) * B * A in FP32."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge_into_base(self):
        """Merges adapter into base weight: W0 <- W0 + delta W."""
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        self.base_layer.weight.data.add_(delta)

class OPLoRALinear(nn.Module):
    """
    Orthogonal Projection LoRA (OPLoRA) layer.
    Uses leading singular vectors Uk (dout x k) and Vk (din x k) of starting weight W0
    to project corrections outside dominant input and output subspaces:
      B_tilde = B - Uk (Uk^T B)
      A_tilde = A - (A Vk) Vk^T
      delta W = (alpha / r) * B_tilde * A_tilde
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        k: int = 128,
        dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.k = k
        self.scaling = alpha / r

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable unprojected factors in FP32
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Leading singular bases registered as non-trainable buffers in FP32
        # Computed once from W0 via SVD
        self.register_buffer("U_k", torch.empty(self.out_features, k, dtype=torch.float32))
        self.register_buffer("V_k", torch.empty(self.in_features, k, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def compute_bases(self, device: Optional[torch.device] = None) -> float:
        """
        Computes SVD of base weight W0 in FP32 and caches top-k singular vectors Uk, Vk.
        Returns time taken in seconds.
        """
        t0 = time.perf_counter()
        w0 = self.base_layer.weight.data.to(torch.float32)
        if device is not None:
            w0 = w0.to(device)

        # torch.linalg.svd returns U, S, Vh where Vh = V^T
        U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
        k = min(self.k, U.shape[1], Vh.shape[0])
        self.U_k.data.copy_(U[:, :k].contiguous())
        self.V_k.data.copy_(Vh[:k, :].T.contiguous())  # Vk shape: (in_features, k)
        elapsed = time.perf_counter() - t0
        return elapsed

    def get_projected_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes effective projected factors B_tilde and A_tilde:
          B_tilde = B - Uk (Uk^T B)
          A_tilde = A - (A Vk) Vk^T
        """
        # Uk shape: (dout, k), B shape: (dout, r)
        # Uk^T B shape: (k, r)
        proj_B = self.U_k @ (self.U_k.T @ self.lora_B)
        B_tilde = self.lora_B - proj_B

        # Vk shape: (din, k), A shape: (r, din)
        # A Vk shape: (r, k)
        proj_A = (self.lora_A @ self.V_k) @ self.V_k.T
        A_tilde = self.lora_A - proj_A

        return B_tilde, A_tilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Base linear
        out = self.base_layer(x)

        # Project adapter factors
        B_tilde, A_tilde = self.get_projected_factors()

        lora_in = self.lora_dropout(x).to(torch.float32)
        lora_out = (lora_in @ A_tilde.T) @ B_tilde.T
        out = out + (self.scaling * lora_out).to(orig_dtype)
        return out

    def get_delta_weight(self) -> torch.Tensor:
        """Returns delta W = (alpha / r) * B_tilde * A_tilde in FP32."""
        B_tilde, A_tilde = self.get_projected_factors()
        return self.scaling * (B_tilde @ A_tilde)

    def merge_into_base(self):
        """Merges projected adapter into base weight: W0 <- W0 + delta W."""
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        self.base_layer.weight.data.add_(delta)

def apply_lora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> Dict[str, LoRALinear]:
    """
    Replaces target linear layers across all transformer layers with LoRALinear.
    Freezes all non-adapter parameters.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Freeze base model parameters
    for p in model.parameters():
        p.requires_grad = False

    replaced_modules = {}
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub_layer = getattr(module, target)
                if isinstance(sub_layer, nn.Linear):
                    lora_layer = LoRALinear(sub_layer, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, target, lora_layer)
                    replaced_modules[f"{name}.{target}"] = lora_layer

    return replaced_modules

def apply_oplora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    k: int = 128,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> Tuple[Dict[str, OPLoRALinear], Dict[str, float]]:
    """
    Replaces target linear layers with OPLoRALinear and precomputes SVD bases.
    Returns dictionary of replaced modules and timing/overhead dictionary.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    for p in model.parameters():
        p.requires_grad = False

    replaced_modules = {}
    svd_times = []
    total_basis_bytes = 0

    print(f"Applying OPLoRA (r={r}, alpha={alpha}, k={k}) and precomputing SVD bases...")
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub_layer = getattr(module, target)
                if isinstance(sub_layer, nn.Linear):
                    oplora_layer = OPLoRALinear(sub_layer, r=r, alpha=alpha, k=k, dropout=dropout)
                    # Compute SVD bases in FP32
                    elapsed = oplora_layer.compute_bases(device=device)
                    svd_times.append(elapsed)

                    # Basis memory in bytes: Uk (dout x k) + Vk (din x k) * 4 bytes
                    basis_bytes = (oplora_layer.out_features * k + oplora_layer.in_features * k) * 4
                    total_basis_bytes += basis_bytes

                    setattr(module, target, oplora_layer)
                    replaced_modules[f"{name}.{target}"] = oplora_layer

    overhead_stats = {
        "num_target_modules": len(replaced_modules),
        "total_svd_preprocessing_time_sec": sum(svd_times),
        "mean_svd_time_per_module_sec": sum(svd_times) / max(1, len(svd_times)),
        "total_basis_storage_mb": total_basis_bytes / (1024 * 1024)
    }
    print(f"OPLoRA setup complete: {len(replaced_modules)} layers replaced. "
          f"Total SVD preprocessing: {overhead_stats['total_svd_preprocessing_time_sec']:.2f}s, "
          f"Basis storage: {overhead_stats['total_basis_storage_mb']:.2f} MB.")
    return replaced_modules, overhead_stats

