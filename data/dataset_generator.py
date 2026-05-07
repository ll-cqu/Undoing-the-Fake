import json
import os
import random

import numpy as np

from undo_the_fake.data.celeba import parse_celeba_annotations
from undo_the_fake.data.plan_sampler import generate_dataset_plan, print_sampling_config
from undo_the_fake.data.styleclip_generation import launch_workers


class DatasetGenerator:
    def __init__(self, args):
        self.args = args

    def seed_everything(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

    def load_done_keys(self):
        done_keys = set()
        if not os.path.exists(self.args.results_json):
            return done_keys

        with open(self.args.results_json) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done_keys.add(f"{rec['image']}_v{rec.get('variant', 0)}")
                except Exception:
                    pass
        print(f"Already completed {len(done_keys)} rows; resuming from checkpoint")
        return done_keys

    def run(self):
        self.seed_everything()
        print_sampling_config()
        annotations = parse_celeba_annotations(self.args.celeba_anno)
        dataset_plan = generate_dataset_plan(
            annotations,
            output_json=self.args.plan_json,
            variants_per_image=self.args.variants,
            max_images=self.args.max_images,
        )
        launch_workers(self.args, dataset_plan, self.load_done_keys())

