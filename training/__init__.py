"""Training entrypoints and trainer builders."""

__all__ = ["warmup", "train", "WarmupTrainer", "GRPOTrainingJob"]


def __getattr__(name):
    if name == "WarmupTrainer":
        from undo_the_fake.training.warmup_trainer import WarmupTrainer
        return WarmupTrainer
    if name == "GRPOTrainingJob":
        from undo_the_fake.training.grpo_trainer import GRPOTrainingJob
        return GRPOTrainingJob
    raise AttributeError(name)
