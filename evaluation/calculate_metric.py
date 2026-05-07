import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="output_file/samples.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="output_file",
    )
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery_min_pair_f1", type=float, default=0.0)
    parser.add_argument("--meta_jsonl", default="data/warmup/warmup_test.jsonl")
    return parser


def main():
    from undo_the_fake.evaluation.metric_calculator import MetricCalculator

    MetricCalculator(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
