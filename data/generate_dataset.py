import argparse
import warnings

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba_img_dir", default="data/CelebAMask-HQ/CelebA-HQ-img")
    parser.add_argument("--celeba_anno", default="data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt")
    parser.add_argument("--plan_json", default="data/edit_plans.json")
    parser.add_argument("--results_json", default="data/edit_results.jsonl")
    parser.add_argument("--failed_json", default="data/edit_failed.jsonl")
    parser.add_argument("--edited_dir", default="data/styleclip_edited")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    from undo_the_fake.data.dataset_generator import DatasetGenerator

    DatasetGenerator(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()

