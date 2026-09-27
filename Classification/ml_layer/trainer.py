"""Bridge module exposing run_training for upstream orchestrator compatibility.

The orchestrator design references:
    from ml_layer.trainer import run_training
    run_training(labeled_df, model_out, config)
"""

from src.train import run_training

__all__ = ["run_training"]
