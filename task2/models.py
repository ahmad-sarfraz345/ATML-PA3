from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

class LoRALinear(nn.Module):
    """
    Standard Low-Rank Adaptation (LoRA) layer wrapping a frozen linear base weight.
    Effective weight: W = W0 + (alpha / r) * B * A
    Zero dependency on external PEFT / torchao.
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
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = self.base_layer(x)
        lora_in = self.lora_dropout(x).to(torch.float32)
        lora_out = (lora_in @ self.lora_A.T) @ self.lora_B.T
        out = out + (self.scaling * lora_out).to(orig_dtype)
        return out

    def get_delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge_into_base(self):
        delta = self.get_delta_weight().to(self.base_layer.weight.dtype)
        self.base_layer.weight.data.add_(delta)

def apply_lora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> Dict[str, LoRALinear]:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for p in model.parameters():
        p.requires_grad = False

    replaced = {}
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub_layer = getattr(module, target)
                if isinstance(sub_layer, nn.Linear):
                    lora_layer = LoRALinear(sub_layer, r=r, alpha=alpha, dropout=dropout).to(
                        device=next(sub_layer.parameters()).device
                    )
                    setattr(module, target, lora_layer)
                    replaced[f"{name}.{target}"] = lora_layer
    return replaced

class VisualConnector(nn.Module):
    """
    2-Layer MLP Connector:
    Linear(768 -> 1024) -> GELU -> Linear(1024 -> 1024)
    Optionally applies 2x2 spatial average pooling (196 -> 49 tokens).
    """
    def __init__(self, in_features: int = 768, hidden_features: int = 1024, out_features: int = 1024, spatial_pooling: bool = False):
        super().__init__()
        self.spatial_pooling = spatial_pooling
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.GELU(),
            nn.Linear(hidden_features, out_features, bias=True)
        )
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.is_calibrated = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 196, 768) in FP16 or FP32
        returns: (B, 196, 1024) or (B, 49, 1024)
        """
        if self.spatial_pooling:
            # Reshape (B, 14, 14, 768) and average pool 2x2 blocks
            b, num_patches, c = x.shape
            grid_size = int(math.isqrt(num_patches))  # 14
            x_grid = x.view(b, grid_size, grid_size, c).permute(0, 3, 1, 2)  # (B, C, 14, 14)
            pooled = nn.functional.avg_pool2d(x_grid, kernel_size=2, stride=2)  # (B, C, 7, 7)
            x = pooled.permute(0, 2, 3, 1).reshape(b, 49, c)  # (B, 49, C)

        out = self.mlp(x)
        return out * self.scale_factor

    def calibrate(self, text_embed_mean_norm: float, sample_vis_features: torch.Tensor):
        """
        Calibrates connector output magnitude against text embeddings using training data:
        gamma = E[||e_text||_2] / E[||e_vis||_2]
        """
        with torch.no_grad():
            raw_out = self.mlp(sample_vis_features)
            vis_norm = torch.norm(raw_out, p=2, dim=-1).mean().item()
            if vis_norm > 1e-6:
                scale = text_embed_mean_norm / vis_norm
                self.scale_factor.data.fill_(scale)
                self.is_calibrated = True
                print(f"[Connector Calibration] text_norm={text_embed_mean_norm:.4f}, vis_norm={vis_norm:.4f}, scale_gamma={scale:.4f}")
            else:
                self.scale_factor.data.fill_(1.0)

class SigLIPVisionEncoder(nn.Module):
    """
    google/siglip-base-patch16-224 (Always frozen).
    Extracts 196 patch tokens in 14x14 spatial order with final LayerNorm applied exactly once.
    """
    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading frozen SigLIP vision encoder from {model_id}...")
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, 224, 224) normalized image tensor
        returns: (B, 196, 768) FP16 tensor
        """
        # SigLIP vision model forward
        vision_model = self.model.vision_model if hasattr(self.model, "vision_model") else self.model
        outputs = vision_model(pixel_values=pixel_values.to(self.device, dtype=torch.float16))
        # outputs.last_hidden_state: (B, 196, 768) with post_layernorm applied
        return outputs.last_hidden_state.detach()

class QwenVLM(nn.Module):
    """
    SigLIP-Qwen Vision-Language Model.
    - Frozen Qwen-0.6B backbone
    - Trainable 2-layer MLP connector throughout Stages 1, 2, 3
    - LoRA on q_proj, k_proj, v_proj, o_proj (r=16, alpha=32, dropout=0.05) enabled in Stages 2 & 3
    """
    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen3-0.6B",
        connector_spatial_pooling: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Qwen backbone in FP16
        print(f"Loading Qwen backbone from {base_model_path}...")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.qwen.gradient_checkpointing_enable()

        # Freeze original Qwen parameters
        for p in self.qwen.parameters():
            p.requires_grad = False

        # Build connector (Linear 768 -> 1024 -> GELU -> Linear 1024 -> 1024)
        self.connector = VisualConnector(
            in_features=768,
            hidden_features=1024,
            out_features=1024,
            spatial_pooling=connector_spatial_pooling
        ).to(self.device, dtype=torch.float32)

        self.lora_enabled = False
        self.lora_layers = {}

    @property
    def lora_model(self):
        return self.qwen

    def enable_lora(self, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        """Enables native LoRA adapters on q_proj, k_proj, v_proj, o_proj for Stages 2 and 3."""
        if not self.lora_enabled:
            print(f"Enabling native LoRA adapters on Qwen (r={r}, alpha={alpha}, dropout={dropout})...")
            self.lora_layers = apply_lora_to_model(
                self.qwen,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.lora_enabled = True
            print(f"LoRA successfully attached ({len(self.lora_layers)} layers).")

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def assemble_multimodal_embeddings(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Assembles inputs_embeds: [prefix_embeds, visual_embeds, suffix_embeds]
        If visual_features is None (no-image condition): [prefix_embeds, suffix_embeds]
        """
        embed_tokens = self.qwen.get_input_embeddings()

        prefix_embeds = embed_tokens(prefix_ids)  # (B, L_pre, 1024)
        suffix_embeds = embed_tokens(suffix_ids)  # (B, L_suf, 1024)

        if visual_features is not None:
            vis_embeds = self.connector(visual_features.to(self.device, dtype=torch.float32)).to(prefix_embeds.dtype)
            return torch.cat([prefix_embeds, vis_embeds, suffix_embeds], dim=1)
        else:
            return torch.cat([prefix_embeds, suffix_embeds], dim=1)

    def forward(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ):
        """Forward pass through Qwen returning logits or hidden states."""
        inputs_embeds = self.assemble_multimodal_embeddings(prefix_ids, suffix_ids, visual_features)
        return self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)

    @torch.no_grad()
    def generate_answer(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16
    ) -> List[str]:
        """
        Greedy decoding (temperature 0, max 16 new tokens, thinking disabled).
        """
        self.eval()
        inputs_embeds = self.assemble_multimodal_embeddings(prefix_ids, suffix_ids, visual_features)
        batch_size = inputs_embeds.shape[0]
        attention_mask = torch.ones(batch_size, inputs_embeds.shape[1], device=self.device, dtype=torch.long)

        # Collect distinct termination tokens
        stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            stop_ids.add(self.tokenizer.pad_token_id)
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            t_id = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(t_id, int) and t_id >= 0:
                stop_ids.add(t_id)

        outputs = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Strict greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=list(stop_ids)
        )

        decoded = []
        for seq in outputs:
            text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            # In case reasoning block was generated, strip it
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            elif "<think>" in text:
                text = text.split("<think>")[-1].strip()
            decoded.append(text)
        return decoded
