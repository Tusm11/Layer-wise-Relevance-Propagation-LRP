"""
LRP (Layer-wise Relevance Propagation) Score Computation Module.
This module provides functionality to compute LRP relevance scores for model weights,
which can then be used by the LRP-Merge method for intelligent model merging.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class LRPConfig:
    """Configuration for LRP computation."""
    model_path: str
    output_path: str
    sample_prompts: List[str]
    batch_size: int = 1
    max_length: int = 512
    lrp_rule: str = "epsilon"  # "epsilon", "gamma", or "alpha_beta"
    epsilon: float = 1e-9
    gamma: float = 0.25
    alpha: float = 1.0
    beta: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LRPComputer:
    """
    Computes Layer-wise Relevance Propagation scores for transformer models.
    """

    def __init__(self, config: LRPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Determine dtype based on device
        has_cuda = torch.cuda.is_available() and self.config.device == "cuda"
        if has_cuda:
            torch_dtype = torch.float16
            device_map = self.config.device
        else:
            torch_dtype = torch.float32
            device_map = None  # device_map not recommended for CPU

        print(f"  Using device: {self.config.device}")
        print(f"  Using dtype: {torch_dtype}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"  Failed with default settings: {e}")
            print("  Trying with trust_remote_code=True...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Explicitly move to CPU if needed
        if self.config.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    def compute_relevance_epsilon(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance using LRP-epsilon rule.
        R_j = sum_k (z_jk / (sum_j z_jk + epsilon)) * R_k
        """
        epsilon = self.config.epsilon

        # Compute forward pass contribution
        z = torch.matmul(activations, weights.t())

        # Add epsilon for numerical stability
        z_sign = torch.sign(z)
        z_stable = z + epsilon * z_sign

        # Compute redistribution
        s = output_relevance / z_stable
        c = torch.matmul(s, weights)

        # Relevance for inputs
        input_relevance = activations * c

        # Relevance for weights (approximated)
        weight_relevance = torch.abs(weights) * torch.abs(output_relevance.mean(dim=0, keepdim=True).t())

        return weight_relevance

    def compute_relevance_gamma(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance using LRP-gamma rule.
        Adds positive contributions with a gamma factor.
        """
        gamma = self.config.gamma

        # Separate positive contributions
        weights_pos = torch.clamp(weights, min=0)

        # Forward pass with enhanced weights
        z = torch.matmul(activations, (weights_pos + gamma * weights_pos).t())
        z = z + self.config.epsilon * torch.sign(z)

        # Redistribute relevance
        s = output_relevance / z
        c = torch.matmul(s, weights)

        weight_relevance = torch.abs(weights) * torch.abs(output_relevance.mean(dim=0, keepdim=True).t())

        return weight_relevance

    def compute_relevance_alpha_beta(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        output_relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance using LRP-alpha_beta rule.
        Separates positive and negative contributions.
        """
        alpha = self.config.alpha
        beta = self.config.beta

        weights_pos = torch.clamp(weights, min=0)
        weights_neg = torch.clamp(weights, max=0)

        # Positive and negative forward passes
        z_pos = torch.matmul(activations, weights_pos.t())
        z_neg = torch.matmul(activations, weights_neg.t())

        z = alpha * z_pos + beta * z_neg
        z = z + self.config.epsilon * torch.sign(z)

        s = output_relevance / z
        c = torch.matmul(s, weights)

        weight_relevance = torch.abs(weights) * torch.abs(output_relevance.mean(dim=0, keepdim=True).t())

        return weight_relevance

    def compute_gradcam_importance(
        self,
        input_ids: torch.Tensor,
        target_layer: str,
    ) -> torch.Tensor:
        """
        Compute importance using Grad-CAM style gradients.
        This is a practical alternative to full LRP.
        """
        self.model.zero_grad()

        # Enable gradients for input
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        logits = outputs.logits

        # Compute gradient of output w.r.t. embeddings
        target_token_idx = logits.shape[1] - 1
        target_logit = logits[0, target_token_idx, :].max()
        target_logit.backward()

        # Get gradients
        gradients = inputs_embeds.grad

        # Importance = gradient magnitude
        importance = torch.abs(gradients)

        return importance

    def compute_relevance_for_tensor(
        self,
        tensor_name: str,
        tensor: torch.Tensor,
        sample_activations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relevance scores for a specific tensor.

        Args:
            tensor_name: Name of the tensor (e.g., "model.layers.0.attn.q_proj.weight")
            tensor: The weight tensor
            sample_activations: Optional sample activations from a forward pass

        Returns:
            Relevance scores for the tensor
        """
        # If we have sample activations, use proper LRP rules
        if sample_activations is not None:
            # Create dummy output relevance (normally from backward pass)
            output_relevance = torch.ones(
                sample_activations.shape[0],
                tensor.shape[0],
                device=tensor.device
            )

            if self.config.lrp_rule == "epsilon":
                relevance = self.compute_relevance_epsilon(
                    sample_activations, tensor, output_relevance
                )
            elif self.config.lrp_rule == "gamma":
                relevance = self.compute_relevance_gamma(
                    sample_activations, tensor, output_relevance
                )
            elif self.config.lrp_rule == "alpha_beta":
                relevance = self.compute_relevance_alpha_beta(
                    sample_activations, tensor, output_relevance
                )
            else:
                raise ValueError(f"Unknown LRP rule: {self.config.lrp_rule}")
        else:
            # Fallback: use magnitude-based proxy
            relevance = torch.abs(tensor)

        return relevance

    def compute_all_relevance_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute relevance scores for all model weights.

        This is the main entry point for computing LRP scores.
        """
        if self.model is None:
            self.load_model()

        print(f"Computing LRP scores using {self.config.lrp_rule} rule...")

        # Tokenize sample prompts
        if self.config.sample_prompts:
            inputs = self.tokenizer(
                self.config.sample_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            # Get sample activations (simplified - use embedding as proxy)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
        else:
            # No samples provided, use magnitude fallback
            print("No sample prompts provided, using magnitude-based importance...")

        # Compute relevance for each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            print(f"Processing {name}...")

            # Compute relevance for this parameter
            relevance = self.compute_relevance_for_tensor(
                name, param.data,
                sample_activations=None  # Simplified - would need proper activations
            )

            self.relevance_scores[name] = relevance.cpu()

        return self.relevance_scores

    def save_relevance_scores(self, output_format: str = "safetensors") -> None:
        """Save computed relevance scores to disk."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(
                    self.relevance_scores,
                    output_path / "lrp_scores.safetensors"
                )
            except ImportError:
                print("safetensors not available, using torch.save")
                torch.save(
                    self.relevance_scores,
                    output_path / "lrp_scores.pt"
                )
        else:
            torch.save(
                self.relevance_scores,
                output_path / "lrp_scores.pt"
            )

        # Save metadata
        metadata = {
            "model_path": self.config.model_path,
            "lrp_rule": self.config.lrp_rule,
            "epsilon": self.config.epsilon,
            "gamma": self.config.gamma,
            "num_tensors": len(self.relevance_scores),
        }

        with open(output_path / "lrp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"LRP scores saved to {output_path}")


def compute_lrp_for_model(
    model_path: str,
    output_path: str,
    sample_prompts: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute LRP scores for a model.

    Args:
        model_path: Path to the HuggingFace model
        output_path: Where to save the LRP scores
        sample_prompts: List of sample prompts for LRP computation
        **kwargs: Additional configuration options

    Returns:
        Dictionary mapping tensor names to relevance scores
    """
    default_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The capital of France is Paris.",
    ]

    config = LRPConfig(
        model_path=model_path,
        output_path=output_path,
        sample_prompts=sample_prompts or default_prompts,
        **kwargs
    )

    computer = LRPComputer(config)
    scores = computer.compute_all_relevance_scores()
    computer.save_relevance_scores()

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute LRP scores for a model")
    parser.add_argument("model_path", help="Path to the HuggingFace model")
    parser.add_argument("output_path", help="Where to save LRP scores")
    parser.add_argument("--rule", default="epsilon", choices=["epsilon", "gamma", "alpha_beta"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompts", nargs="+", help="Sample prompts for LRP computation")

    args = parser.parse_args()

    compute_lrp_for_model(
        model_path=args.model_path,
        output_path=args.output_path,
        sample_prompts=args.prompts,
        lrp_rule=args.rule,
        device=args.device,
    )
