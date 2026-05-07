"""Evaluation and metric calculation entrypoints."""

__all__ = ["calculate_metric", "grpo_gan", "MetricCalculator"]


def __getattr__(name):
    if name == "MetricCalculator":
        from undo_the_fake.evaluation.metric_calculator import MetricCalculator
        return MetricCalculator
    raise AttributeError(name)
