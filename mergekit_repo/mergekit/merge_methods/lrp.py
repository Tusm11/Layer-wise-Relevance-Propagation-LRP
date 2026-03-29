import torch
from typing import Any, Dict, List, Optional
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    MergeMethod,
    MergeTensorInput,
    ConfigParameterDef,
)


class LRPMergeTask(Task[torch.Tensor]):
    """
    Performs LRP-based merging with proper error handling and validation.
    """
    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    model_weights: ImmutableMap[ModelReference, float]
    density: float

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _compute_topk_mask(self, importance: torch.Tensor, density: float) -> torch.Tensor:
        """
        Compute binary mask for top-k most important weights.

        Args:
            importance: Importance scores tensor
            density: Fraction of weights to keep (0.0 to 1.0)

        Returns:
            Binary mask (1 = keep, 0 = discard)
        """
        if density <= 0:
            return torch.zeros_like(importance, dtype=torch.bool)
        if density >= 1.0:
            return torch.ones_like(importance, dtype=torch.bool)

        numel = importance.numel()
        k = max(1, int(density * numel))
        k = min(k, numel)

        # Use topk for efficiency
        flat_importance = importance.flatten()
        top_k_values, _ = torch.topk(flat_importance, k)
        threshold = top_k_values[-1]

        return (importance >= threshold).to(importance.dtype)

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        """
        Execute LRP-based merge.

        Args:
            tensors: Dictionary mapping ModelReference to weight tensors.
                    May include LRP scores with keys suffixed by "_lrp".
        """
        # Validate base tensor
        base_tensor = tensors.get(self.base_model) if self.base_model else None
        if base_tensor is None:
            first_tensor = list(tensors.values())[0] if tensors else None
            if first_tensor is None:
                raise ValueError("No tensors provided for merging")
            base_tensor = torch.zeros_like(first_tensor)

        if not isinstance(base_tensor, torch.Tensor):
            raise TypeError(f"base_tensor must be torch.Tensor, got {type(base_tensor)}")

        # Initialize merged deltas
        merged_deltas = torch.zeros_like(base_tensor)

        # Validate weights
        if not self.model_weights:
            raise ValueError("model_weights cannot be empty")

        total_weight = sum(self.model_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of model weights cannot be zero")

        # Process each model
        for ref, fine_tuned_weight in tensors.items():
            # Skip base model and LRP score tensors
            if ref == self.base_model or str(ref).endswith("_lrp"):
                continue

            # Validate tensor
            if not isinstance(fine_tuned_weight, torch.Tensor):
                raise TypeError(f"Weight for {ref} must be torch.Tensor, got {type(fine_tuned_weight)}")

            if fine_tuned_weight.shape != base_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {ref}: expected {base_tensor.shape}, got {fine_tuned_weight.shape}"
                )

            # Compute delta (task vector)
            delta = fine_tuned_weight - base_tensor

            # Get LRP importance scores if available
            lrp_ref = f"{ref}_lrp"
            importance = None
            for tensor_ref in tensors.keys():
                if str(tensor_ref) == lrp_ref or str(tensor_ref).endswith(f"_{ref}_lrp"):
                    importance = tensors.get(tensor_ref)
                    break

            # Fallback to magnitude-based importance
            if importance is None:
                importance = delta.abs()
            elif not isinstance(importance, torch.Tensor):
                raise TypeError(f"LRP score for {ref} must be torch.Tensor, got {type(importance)}")

            # Validate importance shape
            if importance.shape != delta.shape:
                importance = delta.abs()

            # Sparsify based on importance
            mask = self._compute_topk_mask(importance, self.density)
            sparse_delta = delta * mask

            # Weighted averaging
            try:
                weight = self.model_weights[ref] if ref in self.model_weights else 1.0
            except (KeyError, TypeError):
                weight = 1.0

            normalized_weight = weight / total_weight
            merged_deltas += normalized_weight * sparse_delta

        # Final merged tensor
        return base_tensor + merged_deltas

    def uses_accelerator(self) -> bool:
        return True

    def priority(self) -> int:
        return 0


class LRPMerge(MergeMethod):
    """
    LRP-based merge method.

    Merges fine-tuned models by:
    1. Computing task vectors (deltas from base)
    2. Sparsifying based on importance (LRP scores or magnitude)
    3. Weighted averaging of sparse deltas
    """

    def name(self) -> str:
        return "lrp"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", default_value=0.7),
            ConfigParameterDef(name="use_lrp", default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", default_value=1.0)]

    def make_task(
        self,
        *,
        output_weight: Any,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        """Create the LRP merge task with proper validation."""
        # Collect model weights
        model_weights = {}
        for m, p in tensor_parameters.items():
            if m != base_model:
                try:
                    weight = p["weight"]
                except (KeyError, TypeError):
                    weight = 1.0
                model_weights[m] = weight

        if not model_weights:
            raise ValueError("At least one fine-tuned model (other than base) is required")

        # Get density parameter
        try:
            density = parameters["density"]
        except (KeyError, TypeError):
            density = 0.7

        # Validate density
        if not 0 <= density <= 1:
            raise ValueError(f"density must be between 0 and 1, got {density}")

        return LRPMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            model_weights=ImmutableMap(model_weights),
            density=density,
        )
