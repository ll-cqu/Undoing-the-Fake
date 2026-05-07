"""Dataset and synthetic data generation modules."""

__all__ = ["DatasetGenerator"]


def __getattr__(name):
    if name == "DatasetGenerator":
        from undo_the_fake.data.dataset_generator import DatasetGenerator
        return DatasetGenerator
    raise AttributeError(name)
