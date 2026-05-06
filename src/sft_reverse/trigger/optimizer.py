"""
Soft prompt optimization via activation matching on circuit channels.

Optimizes continuous prompt embeddings to steer circuit channel activations
to match base model (pre-SFT) patterns.

Two optimization methods:
1. Statistical: Match mean circuit activations of SFT(trigger+prompt) to Base(prompt)
2. Statistical+KL: Same as statistical, plus KL divergence on output logits
"""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .activations import ResidualStreamHook


class SoftPromptOptimizer:
    """
    Optimize soft prompt triggers via activation matching on circuit channels.

    The trigger is a sequence of learnable embeddings that, when prepended to
    a prompt, steers the model's circuit channel activations to match
    those of the base model (pre-SFT).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        per_layer_channels: Dict[int, List[int]],
        trigger_length: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.per_layer_channels = per_layer_channels
        self.layer_indices = list(per_layer_channels.keys())
        self.trigger_length = trigger_length

        # Always initialize from random real token embeddings so trigger starts
        # on the natural token embedding manifold (avoids OOD direction bias).
        vocab_size = model.config.vocab_size
        token_ids = torch.randint(0, vocab_size, (trigger_length,), device=model.device)
        with torch.no_grad():
            init_embeds = model.model.embed_tokens(token_ids).float()  # type: ignore[union-attr]
        self.trigger_embeds = nn.Parameter(init_embeds.detach())
        logger.info(
            f"Token init: sampled ids={token_ids.tolist()}, "
            f"norms={init_embeds.norm(dim=-1).tolist()}"
        )

        self.hook = ResidualStreamHook(
            model,
            per_layer_channels=self.per_layer_channels,
            detach=False,
        )

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

    @classmethod
    def from_circuit(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint_path: str,
        trigger_length: int = 5,
        channel_type: str = "write",
    ) -> "SoftPromptOptimizer":
        """
        Create optimizer using circuit information from L0-SFT checkpoint.
        """
        from safety_circuit.circuit import CircuitExtractor

        extractor = CircuitExtractor.from_checkpoint(checkpoint_path)
        n_layer = extractor.config.n_layer

        if channel_type == "write":
            per_layer_channels = extractor.get_residual_write_channels()
        elif channel_type == "all_residual":
            per_layer_channels = extractor.get_all_residual_channels()
        elif channel_type == "ffn_write":
            per_layer_channels = extractor.get_all_active_channels("ffn_write")
        elif channel_type == "attn_write":
            per_layer_channels = extractor.get_all_active_channels("attn_write")
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        active_layers = {
            layer_idx: channels
            for layer_idx in range(n_layer)
            if len(channels := per_layer_channels.get(layer_idx, [])) > 0
        }

        logger.info(f"Circuit: {len(active_layers)} layers with active channels")
        total_channels = sum(len(ch) for ch in active_layers.values())
        logger.info(f"Total channels to match: {total_channels}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            per_layer_channels=active_layers,
            trigger_length=trigger_length,
        )

    @classmethod
    def from_random_channels(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint_path: str,
        trigger_length: int = 5,
        channel_type: str = "write",
    ) -> "SoftPromptOptimizer":
        """C2 ablation: same channel budget as circuit, but randomly sampled indices."""
        from safety_circuit.circuit import CircuitExtractor

        extractor = CircuitExtractor.from_checkpoint(checkpoint_path)
        n_layer = extractor.config.n_layer

        if channel_type == "write":
            circuit_channels = extractor.get_residual_write_channels()
        elif channel_type == "all_residual":
            circuit_channels = extractor.get_all_residual_channels()
        elif channel_type == "ffn_write":
            circuit_channels = extractor.get_all_active_channels("ffn_write")
        elif channel_type == "attn_write":
            circuit_channels = extractor.get_all_active_channels("attn_write")
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        d_model = model.config.hidden_size
        random_channels: Dict[int, List[int]] = {}
        for layer_idx in range(n_layer):
            ch = circuit_channels.get(layer_idx, [])
            if len(ch) > 0:
                k = min(len(ch), d_model)
                random_channels[layer_idx] = random.sample(range(d_model), k)

        total = sum(len(v) for v in random_channels.values())
        logger.info(
            f"Random channels: {len(random_channels)} layers, {total} total "
            f"(same budget as circuit)"
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            per_layer_channels=random_channels,
            trigger_length=trigger_length,
        )

    def _get_embed_tokens(self, model: PreTrainedModel):
        """Get the embedding layer from the model."""
        return model.model.embed_tokens  # type: ignore[union-attr]

    def forward_with_trigger(self, input_ids: torch.Tensor):
        """Forward pass with trigger prepended to input."""
        batch_size = input_ids.shape[0]

        inputs_embeds = self._get_embed_tokens(self.model)(input_ids)  # type: ignore[operator]

        trigger_batch = self.trigger_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        trigger_batch = trigger_batch.to(inputs_embeds.dtype)
        full_embeds = torch.cat([trigger_batch, inputs_embeds], dim=1)

        outputs = self.model(inputs_embeds=full_embeds)

        return outputs

    def collect_activation_statistics(
        self, prompts: List[str], desc: str = "Collecting", prepend_dummy: bool = False
    ) -> Dict[int, torch.Tensor]:
        """Collect mean activations across prompts on circuit channels."""
        all_activations: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in self.layer_indices
        }

        logger.info(f"{desc} activations from {len(prompts)} prompts...")

        for prompt in tqdm(prompts, desc=desc):
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            )["input_ids"].to(self.model.device)  # type: ignore[union-attr]

            with torch.no_grad():
                with self.hook:
                    if prepend_dummy:
                        inputs_embeds = self._get_embed_tokens(self.model)(input_ids)  # type: ignore[operator]
                        dummy_embeds = torch.zeros(
                            1,
                            self.trigger_length,
                            self.model.config.hidden_size,
                            device=self.model.device,
                            dtype=inputs_embeds.dtype,
                        )
                        full_embeds = torch.cat([dummy_embeds, inputs_embeds], dim=1)
                        _ = self.model(inputs_embeds=full_embeds)
                    else:
                        _ = self.model(input_ids)

                    for layer_idx in self.layer_indices:
                        acts = self.hook.activations[layer_idx].mean(dim=0)
                        all_activations[layer_idx].append(acts)

        mean_activations = {}
        for layer_idx in self.layer_indices:
            stacked = torch.stack(all_activations[layer_idx])
            mean_activations[layer_idx] = stacked.mean(dim=0)

        return mean_activations

    def optimize_statistical(
        self,
        harmful_prompts: List[str],
        safe_prompts: List[str],
        n_steps: int = 1000,
        lr: float = 0.01,
        batch_size: int = 1,
        base_model: Optional[PreTrainedModel] = None,
        max_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Optimize trigger using statistical mean matching.

        Two modes:
        1. If base_model is provided: Match SFT(prompt+trigger) to Base(prompt)
           - Target: Base model on prompts (no SFT behavior)
           - This reverts SFT back to base behavior
        2. If base_model is None: Match SFT(harmful+trigger) to SFT(safe)
           - Legacy mode for comparison
        """
        assert max_norm is None or max_norm > 0.0, "max_norm must be > 0 when provided"
        logger.info("=" * 80)
        logger.info("STATISTICAL TRIGGER OPTIMIZATION")
        logger.info("=" * 80)

        if base_model is not None:
            logger.info("Mode: Match SFT(prompt+trigger) to Base(prompt)")

            # Create a temporary hook for base model
            base_hook = ResidualStreamHook(
                base_model,
                per_layer_channels=self.per_layer_channels,
                detach=True,
            )

            all_activations: Dict[int, List[torch.Tensor]] = {
                layer: [] for layer in self.layer_indices
            }

            logger.info(
                f"Collecting base model activations from {len(harmful_prompts)} prompts..."
            )
            for prompt in tqdm(harmful_prompts, desc="Base model"):
                input_ids = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"].to(base_model.device)  # type: ignore[union-attr]

                with torch.no_grad():
                    with base_hook:
                        _ = base_model(input_ids)
                        for layer_idx in self.layer_indices:
                            acts = base_hook.activations[layer_idx].mean(dim=0)
                            all_activations[layer_idx].append(acts)

            target_mean = {}
            for layer_idx in self.layer_indices:
                stacked = torch.stack(all_activations[layer_idx])
                target_mean[layer_idx] = stacked.mean(dim=0)

            target_mean = {k: v.to(self.model.device) for k, v in target_mean.items()}

        else:
            logger.info("Mode: Match SFT(harmful+trigger) to SFT(safe) [Legacy]")
            target_mean = self.collect_activation_statistics(
                safe_prompts, desc="Safe prompts", prepend_dummy=False
            )

        # Baseline stats
        baseline_harmful_mean = self.collect_activation_statistics(
            harmful_prompts, desc="SFT harmful (baseline)", prepend_dummy=False
        )

        logger.info("Target Direction Analysis:")
        for layer_idx in self.layer_indices[:3]:
            diff = target_mean[layer_idx] - baseline_harmful_mean[layer_idx]
            logger.info(
                f"  Layer {layer_idx:2d}: mean |difference| = {diff.abs().mean():.4f}, "
                f"std = {diff.std():.4f}"
            )

        # Optimize trigger
        logger.info(
            f"Optimizing trigger for {n_steps} steps (batch_size={batch_size})..."
        )
        optimizer = torch.optim.Adam([self.trigger_embeds], lr=lr)

        losses = []
        best_loss = float("inf")
        best_trigger = self.trigger_embeds.detach().cpu().clone()
        best_step = 0

        for step in range(n_steps):
            optimizer.zero_grad()

            batch_harmful = random.sample(
                harmful_prompts, min(batch_size, len(harmful_prompts))
            )

            all_acts: Dict[int, List[torch.Tensor]] = {
                layer: [] for layer in self.layer_indices
            }
            for prompt in batch_harmful:
                input_ids = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"].to(self.model.device)  # type: ignore[union-attr]
                with self.hook:
                    _ = self.forward_with_trigger(input_ids)
                    for layer_idx in self.layer_indices:
                        all_acts[layer_idx].append(
                            self.hook.activations[layer_idx].squeeze(0)
                        )

            batch_mean_acts = {
                layer_idx: torch.stack(acts).mean(dim=0)
                for layer_idx, acts in all_acts.items()
            }

            total_loss = torch.tensor(0.0, device=self.model.device)
            for layer_idx in self.layer_indices:
                layer_loss = F.mse_loss(
                    batch_mean_acts[layer_idx], target_mean[layer_idx]
                )
                total_loss = total_loss + layer_loss

            loss = total_loss / len(self.layer_indices)
            loss.backward()

            torch.nn.utils.clip_grad_norm_([self.trigger_embeds], max_norm=1.0)

            optimizer.step()

            # PGD projection: clamp each trigger token to the L2 ball
            if max_norm is not None:
                with torch.no_grad():
                    norms = self.trigger_embeds.norm(dim=-1, keepdim=True)
                    scale = (max_norm / norms).clamp(max=1.0)
                    self.trigger_embeds.data.mul_(scale)

            losses.append(loss.item())

            # Track best trigger (lowest loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_trigger = self.trigger_embeds.detach().cpu().clone()
                best_step = step

            if step % 100 == 0:
                cur_norms = self.trigger_embeds.norm(dim=-1)
                logger.info(
                    f"Step {step:4d} | Loss: {loss.item():.6f} "
                    f"| trigger norms: {cur_norms.tolist()}"
                )

        logger.info(f"Step {n_steps:4d} | Loss: {losses[-1]:.6f}")
        logger.info(f"Best step: {best_step} | Best loss: {best_loss:.6f}")

        return {
            "trigger_embeds": best_trigger,
            "losses": losses,
            "final_loss": best_loss,
            "best_step": best_step,
            "target_mean": {k: v.cpu() for k, v in target_mean.items()},
            "baseline_harmful_mean": {
                k: v.cpu() for k, v in baseline_harmful_mean.items()
            },
        }

    def optimize_statistical_kl(
        self,
        base_model: PreTrainedModel,
        prompts: List[str],
        n_steps: int = 1000,
        lr: float = 0.01,
        batch_size: int = 8,
        kl_weight: float = 0.1,
        kl_tail_k: int = 1,
        l2: float = 0.0,
        max_norm: Optional[float] = None,
        circuit_mse_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Circuit MSE + Logit KL divergence optimization.

        Loss = circuit_MSE(SFT(trig+x), Base(x))
             + kl_weight * KL_tail(Base(x) || SFT(trig+x))
             + l2 * ||trigger||^2

        x is sampled from the SFT training distribution (prompts that activate
        the learned behavior). The trigger acts as a global ON/OFF switch:
        when active, SFT(trigger+x) ≈ Base(x) for all x.
        """
        assert max_norm is None or max_norm > 0.0, "max_norm must be > 0 when provided"
        assert l2 >= 0.0, "l2 must be >= 0"
        assert kl_tail_k > 0, "kl_tail_k must be > 0"
        logger.info("=" * 80)
        logger.info("STATISTICAL + KL TRIGGER OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"KL weight: {kl_weight}")
        logger.info(f"KL tail k: {kl_tail_k}")
        logger.info(f"L2 weight: {l2}")

        # Pre-compute base model circuit activations and tail logits for prompts
        base_hook = ResidualStreamHook(
            base_model,
            per_layer_channels=self.per_layer_channels,
            detach=True,
        )

        all_circuit_acts: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in self.layer_indices
        }
        base_tail_logits_list: List[torch.Tensor] = []
        prompt_ids_list: List[torch.Tensor] = []
        prompt_seq_lens: List[int] = []

        logger.info(f"Pre-computing base model outputs for {len(prompts)} prompts...")
        for prompt in tqdm(prompts, desc="Base model"):
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            )["input_ids"].to(base_model.device)  # type: ignore[union-attr]
            prompt_ids_list.append(input_ids.to(self.model.device))
            seq_len = int(input_ids.shape[1])
            prompt_seq_lens.append(seq_len)

            with torch.no_grad():
                with base_hook:
                    outputs = base_model(input_ids)
                    for layer_idx in self.layer_indices:
                        acts = base_hook.activations[layer_idx].mean(dim=0)
                        all_circuit_acts[layer_idx].append(acts)
                    k = min(kl_tail_k, seq_len)
                    base_tail_logits_list.append(
                        outputs.logits[0, seq_len - k : seq_len, :].detach().cpu()
                    )

        # Mean circuit activations (target for MSE)
        target_mean = {}
        for layer_idx in self.layer_indices:
            stacked = torch.stack(all_circuit_acts[layer_idx])
            target_mean[layer_idx] = stacked.mean(dim=0).to(self.model.device)

        # Baseline circuit stats (SFT without trigger)
        baseline_mean = self.collect_activation_statistics(
            prompts, desc="SFT baseline (no trigger)", prepend_dummy=False
        )
        logger.info("Target Direction Analysis:")
        for layer_idx in self.layer_indices[:3]:
            diff = target_mean[layer_idx] - baseline_mean[layer_idx]
            logger.info(
                f"  Layer {layer_idx:2d}: mean |diff| = {diff.abs().mean():.4f}, "
                f"std = {diff.std(correction=0):.4f}"
            )

        # Optimization
        logger.info(
            f"Optimizing for {n_steps} steps (batch_size={batch_size}, "
            f"kl_weight={kl_weight})..."
        )
        optimizer = torch.optim.Adam([self.trigger_embeds], lr=lr)

        losses_total: List[float] = []
        losses_circuit: List[float] = []
        losses_kl: List[float] = []
        losses_l2: List[float] = []
        best_loss = float("inf")
        best_trigger = self.trigger_embeds.detach().cpu().clone()
        best_step = 0

        for step in range(n_steps):
            optimizer.zero_grad()

            batch_indices = random.sample(
                range(len(prompt_ids_list)),
                min(batch_size, len(prompt_ids_list)),
            )

            # Forward pass: collect circuit activations and logits
            all_acts: Dict[int, List[torch.Tensor]] = {
                layer: [] for layer in self.layer_indices
            }
            kl_losses: List[torch.Tensor] = []

            for idx in batch_indices:
                input_ids = prompt_ids_list[idx]
                seq_len = prompt_seq_lens[idx]
                base_tail_logits = base_tail_logits_list[idx].to(self.model.device)
                k = int(base_tail_logits.shape[0])

                with self.hook:
                    outputs = self.forward_with_trigger(input_ids)
                    for layer_idx in self.layer_indices:
                        all_acts[layer_idx].append(
                            self.hook.activations[layer_idx].squeeze(0)
                        )

                # KL divergence on tail-k aligned logits: KL(Base || SFT+trigger)
                sft_aligned_logits = outputs.logits[
                    0, self.trigger_length : self.trigger_length + seq_len, :
                ]
                assert sft_aligned_logits.shape[0] == seq_len, (
                    "Aligned SFT logits length mismatch"
                )
                sft_tail_logits = sft_aligned_logits[-k:, :]
                sft_log_probs = F.log_softmax(sft_tail_logits.float(), dim=-1)
                base_log_probs = F.log_softmax(base_tail_logits.float(), dim=-1)
                kl_per_token = F.kl_div(
                    input=sft_log_probs,
                    target=base_log_probs,
                    log_target=True,
                    reduction="none",
                )
                kl_losses.append(kl_per_token.sum(dim=-1).mean())

            # Circuit MSE loss (statistical mean matching)
            # Skip when no circuit channels (--no-circuit) or weight is zeroed (C3 ablation)
            if len(self.layer_indices) == 0 or circuit_mse_weight == 0.0:
                circuit_loss = torch.tensor(0.0, device=self.model.device)
            else:
                batch_mean_acts = {
                    layer_idx: torch.stack(acts).mean(dim=0)
                    for layer_idx, acts in all_acts.items()
                }
                circuit_loss = torch.tensor(0.0, device=self.model.device)
                for layer_idx in self.layer_indices:
                    circuit_loss = circuit_loss + F.mse_loss(
                        batch_mean_acts[layer_idx], target_mean[layer_idx]
                    )
                circuit_loss = circuit_loss / len(self.layer_indices)

            # KL loss (mean over batch)
            kl_loss = torch.stack(kl_losses).mean()
            # Mean squared norm per trigger token to keep l2 scale stable
            # across different trigger lengths.
            l2_loss = self.trigger_embeds.pow(2).sum(dim=-1).mean()

            # Total loss
            loss = (
                circuit_mse_weight * circuit_loss + kl_weight * kl_loss + l2 * l2_loss
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_([self.trigger_embeds], max_norm=1.0)
            optimizer.step()

            # PGD projection: clamp each trigger token to the L2 ball
            if max_norm is not None:
                with torch.no_grad():
                    norms = self.trigger_embeds.norm(dim=-1, keepdim=True)
                    scale = (max_norm / norms).clamp(max=1.0)
                    self.trigger_embeds.data.mul_(scale)

            losses_total.append(loss.item())
            losses_circuit.append(circuit_loss.item())
            losses_kl.append(kl_loss.item())
            losses_l2.append(l2_loss.item())

            # Track best trigger (lowest total loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_trigger = self.trigger_embeds.detach().cpu().clone()
                best_step = step

            if step % 100 == 0:
                cur_norms = self.trigger_embeds.norm(dim=-1)
                logger.info(
                    f"Step {step:4d} | Total: {loss.item():.4f} "
                    f"| Circuit: {circuit_loss.item():.6f} "
                    f"| KL: {kl_loss.item():.4f} "
                    f"| L2: {l2_loss.item():.4f} "
                    f"| trigger norms: {[f'{n:.3f}' for n in cur_norms.tolist()]}"
                )

        final_norms = self.trigger_embeds.norm(dim=-1)
        logger.info(
            f"Step {n_steps:4d} | Total: {losses_total[-1]:.4f} "
            f"| Circuit: {losses_circuit[-1]:.6f} "
            f"| KL: {losses_kl[-1]:.4f} "
            f"| L2: {losses_l2[-1]:.4f} "
            f"| trigger norms: {[f'{n:.3f}' for n in final_norms.tolist()]}"
        )
        logger.info(f"Best step: {best_step} | Best loss: {best_loss:.6f}")

        return {
            "trigger_embeds": best_trigger,
            "losses": losses_total,
            "losses_circuit": losses_circuit,
            "losses_kl": losses_kl,
            "losses_l2": losses_l2,
            "final_loss": best_loss,
            "best_step": best_step,
        }
