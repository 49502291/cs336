"""
Pass ``use_wandb=True`` together with ``wandb_project``, ``wandb_entity``,
and ``wandb_run_name`` to the constructor.  The logger will:

* call ``wandb.init()`` and upload ``config`` records as ``wandb.config``.
* call ``wandb.log()`` for every ``train`` / ``val`` record, keying metrics
  by their field names and using ``step`` as the x-axis.
* call ``wandb.finish()`` on ``close()``.

Example usage
-------------
.. code-block:: python

    from cs336_basics.experiment_logger import ExperimentLogger

    # JSONL only
    with ExperimentLogger("logs/run.jsonl") as logger:
        logger.log("train", step=100, wallclock=12.3, loss=2.45, lr=3e-4)

    # JSONL + W&B
    with ExperimentLogger(
        "logs/run.jsonl",
        use_wandb=True,
        wandb_project="cs336",
        wandb_run_name="baseline",
        wandb_config={"d_model": 512, "num_layers": 6},
    ) as logger:
        logger.log("train", step=100, wallclock=12.3, loss=2.45, lr=3e-4)
        logger.log("val",   step=500, wallclock=61.7, val_loss=2.31)
"""
import json
import os
from typing import Any


class ExperimentLogger:
    """
    Writes experiment events to a JSONL file and optionally to W&B.

    Args:
        log_path: Path to the JSONL output file.  Parent directories are
            created automatically.
        append: Open the file in append mode (useful when resuming a run).
        use_wandb: Initialise a W&B run.  Requires ``wandb`` to be installed.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (team / username).  Defaults to your default
            entity.
        wandb_run_name: Display name for the W&B run.
        wandb_config: Dictionary of hyper-parameters to log as W&B config.
        wandb_resume: Pass ``"allow"`` or ``"must"`` to resume an existing W&B
            run; ``None`` (default) starts a new run.
    """

    def __init__(
        self,
        log_path: str,
        append: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "cs336",
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        wandb_resume: str | None = None,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Local JSONL file                                                     #
        # ------------------------------------------------------------------ #
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._f = open(log_path, "a" if append else "w", buffering=1)
        self.log_path = log_path

        # ------------------------------------------------------------------ #
        # Weights & Biases (optional)                                         #
        # ------------------------------------------------------------------ #
        self._wandb = None
        if use_wandb:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=wandb_config or {},
                resume=wandb_resume,
            )
            print(f"W&B run: {wandb.run.url}")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def log(self, event_type: str, step: int, wallclock: float, **kwargs: Any) -> None:
        """Write one event record.

        Args:
            event_type: ``"config"``, ``"train"``, or ``"val"``.
            step: Gradient step (used as W&B x-axis).
            wallclock: Seconds elapsed since training started.
            **kwargs: Arbitrary scalar metrics (loss, lr, val_loss, …).
        """
        record: dict[str, Any] = {
            "type": event_type,
            "step": step,
            "wallclock": round(wallclock, 4),
        }
        record.update(kwargs)

        # JSONL ---------------------------------------------------------------
        self._f.write(json.dumps(record) + "\n")

        # W&B -----------------------------------------------------------------
        if self._wandb is not None and event_type in ("train", "val"):
            # Prefix val metrics so they appear in a separate section in the UI
            prefix = "train/" if event_type == "train" else "val/"
            metrics: dict[str, Any] = {"wallclock": wallclock}
            for k, v in kwargs.items():
                key = f"{prefix}{k}"
                metrics[key] = v
            self._wandb.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Upload hyper-parameters to W&B config (no-op if W&B is disabled)."""
        if self._wandb is not None:
            self._wandb.config.update(config, allow_val_change=True)

    def close(self) -> None:
        """Flush and close the JSONL file; finish the W&B run if active."""
        self._f.close()
        if self._wandb is not None:
            self._wandb.finish()

    # context-manager support
    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
