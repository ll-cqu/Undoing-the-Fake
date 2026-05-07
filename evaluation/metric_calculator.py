import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from undo_the_fake.utils.metrics import (
    attr_pair_level_metrics,
    average,
    rebuild_per_attr,
    recovery_metrics_for_rows,
)
from undo_the_fake.utils.parsing import parse_output


class MetricCalculator:
    def __init__(self, args):
        self.args = args
        self.accelerator = None
        self.rank = 0
        self.world = 1
        self.is_main = True
        self.device = "cpu"
        self.records = []
        self.meta_lookup = {}
        self.fake_records = []
        self.real_records = []

    def setup_distributed(self):
        from accelerate import Accelerator

        self.accelerator = Accelerator()
        self.rank = self.accelerator.process_index
        self.world = self.accelerator.num_processes
        self.is_main = self.accelerator.is_main_process
        self.device = str(self.accelerator.device)

    def load_records(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(self.args.input) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if self.is_main:
            print(f"[data] {len(self.records)} records  world={self.world}", flush=True)

    def load_meta_lookup(self):
        if not self.args.meta_jsonl or not os.path.exists(self.args.meta_jsonl):
            return

        with open(self.args.meta_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = rec.get("_meta", {})
                key = meta.get("edited_path", "")
                if key:
                    self.meta_lookup[key] = {
                        "recon_path": meta.get("recon_path", ""),
                        "lpips_edited_recon": meta.get("lpips_edited_recon", None),
                        "arc_recon_edited": meta.get("arc_recon_edited", None),
                        "attrs": meta.get("attrs", []),
                        "actions": meta.get("actions", []),
                        "alpha_levels": meta.get("alpha_levels", []),
                        "alpha_abs": meta.get("alpha_abs", []),
                        "num_edits": meta.get("num_edits", None),
                    }
        if self.is_main:
            print(f"[data] meta_lookup: {len(self.meta_lookup)} entries", flush=True)

    def split_records(self):
        self.fake_records = [
            row for row in self.records
            if row.get("gt_flag") or row.get("sample_type") == "fake"
        ]
        self.real_records = [
            row for row in self.records
            if not row.get("gt_flag") or row.get("sample_type") == "real"
        ]
        if self.is_main:
            print(f"[data] fake={len(self.fake_records)}  real={len(self.real_records)}", flush=True)

    def warmup_recovery_models(self):
        if not self.args.recovery:
            return

        from undo_the_fake.models.gan import get_arcface, get_gan_models

        print(f"[rank{self.rank}] Loading GAN + ArcFace on {self.device}...", flush=True)
        get_gan_models(self.device, with_lpips=True)
        get_arcface(self.device)
        print(f"[rank{self.rank}] Ready", flush=True)

    def process_shard(self):
        shard = self.fake_records[self.rank::self.world]
        local_rows = []
        local_failed = []

        for rec in tqdm(shard, desc=f"[rank{self.rank}]", position=self.rank, leave=True):
            metrics = attr_pair_level_metrics(rec, parse_output)
            image_path = rec.get("image_path", rec.get("image", ""))
            recovery = self.compute_recovery(rec, metrics, image_path)
            local_failed.extend(recovery.pop("failed"))

            row = {
                "image_path": image_path,
                "gt_attrs": sorted(metrics["gt_attrs"]),
                "pred_attrs": sorted(metrics["pred_attrs"]),
                "tp_attrs": sorted(metrics["tp_attrs"]),
                "fp_attrs": sorted(metrics["fp_attrs"]),
                "fn_attrs": sorted(metrics["fn_attrs"]),
                "action_correct_attrs": sorted(metrics["action_correct_attrs"]),
                "level_correct_attrs": sorted(metrics["level_correct_attrs"]),
                "gt_actions": [metrics["gt_act_map"].get(attr, "") for attr in sorted(metrics["gt_attrs"])],
                "pred_actions": [metrics["pred_act_map"].get(attr, "") for attr in sorted(metrics["pred_attrs"])],
                "gt_levels": [metrics["gt_lv_map"].get(attr, "") for attr in sorted(metrics["gt_attrs"])],
                "pred_levels": [metrics["pred_lv_map"].get(attr, "") for attr in sorted(metrics["pred_attrs"])],
                **metrics["row"],
                **recovery,
            }
            local_rows.append(row)

        return local_rows, local_failed

    def compute_recovery(self, rec, metrics, image_path):
        result = {
            "lp_recovery_rate": None,
            "arc_sim_edited": None,
            "arc_sim_rec_level": None,
            "arc_sim_rec_exact": None,
            "failed": [],
        }
        if not self.args.recovery:
            return result

        meta = self.meta_lookup.get(image_path, {})
        recon_path = rec.get("recon_path", "") or meta.get("recon_path", "")
        lp_edited = rec.get("lpips_edited_recon") or meta.get("lpips_edited_recon")
        arc_edited = rec.get("_meta", {}).get("arc_recon_edited") or meta.get("arc_recon_edited")
        exact_manips = self.build_exact_manips(meta, metrics)

        can_recover = (
            metrics["pred_m"]
            and metrics["pair_f1"] >= self.args.recovery_min_pair_f1
            and image_path and os.path.exists(image_path)
            and recon_path and os.path.exists(recon_path)
        )
        if not can_recover:
            result["failed"].append({"image": image_path, "reason": "skip", "pair_f1": round(metrics["pair_f1"], 4)})
            return result

        try:
            from PIL import Image
            from undo_the_fake.models.gan import arcface_sim, run_recovery

            edited_img = Image.open(image_path).convert("RGB")
            recon_img = Image.open(recon_path).convert("RGB")

            recovered_level, lp_rec_level = run_recovery(
                edited_img,
                recon_img,
                metrics["pred_m"],
                self.device,
                use_alpha_abs=False,
                with_lpips=True,
            )

            if lp_rec_level is not None and lp_edited is not None and lp_edited > 0:
                result["lp_recovery_rate"] = float(np.clip((lp_edited - lp_rec_level) / lp_edited, 0.0, 1.0))

            result["arc_sim_edited"] = arc_edited
            if recovered_level is not None:
                result["arc_sim_rec_level"] = float(arcface_sim(recon_img, recovered_level, self.device))

            if exact_manips is not None:
                recovered_exact, _ = run_recovery(
                    edited_img,
                    recon_img,
                    exact_manips,
                    self.device,
                    use_alpha_abs=True,
                    with_lpips=True,
                )
                if recovered_exact is not None:
                    result["arc_sim_rec_exact"] = float(arcface_sim(recon_img, recovered_exact, self.device))

            if recovered_level is None:
                result["failed"].append({"image": image_path, "reason": "recovery_failed"})
        except Exception as exc:
            print(f"[rank{self.rank}] recovery error: {exc}", flush=True)
            result["failed"].append({"image": image_path, "reason": str(exc)})

        return result

    def build_exact_manips(self, meta, metrics):
        alpha_abs_list = meta.get("alpha_abs", [])
        gt_attrs = meta.get("attrs", metrics["gt_attr_list"])
        gt_actions = meta.get("actions", metrics["gt_act_list"])
        gt_levels = meta.get("alpha_levels", metrics["gt_lv_list"])
        if not alpha_abs_list or len(alpha_abs_list) != len(gt_attrs):
            return None
        return [
            {"attr": attr, "action": action, "alpha_level": level, "alpha_abs": alpha_abs}
            for attr, action, level, alpha_abs in zip(gt_attrs, gt_actions, gt_levels, alpha_abs_list)
        ]

    def write_rank_files(self, local_rows, local_failed):
        tmp_rows = os.path.join(self.args.output_dir, f"_tmp_rows_rank{self.rank}.jsonl")
        tmp_failed = os.path.join(self.args.output_dir, f"_tmp_failed_rank{self.rank}.jsonl")
        with open(tmp_rows, "w") as f:
            for row in local_rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        with open(tmp_failed, "w") as f:
            for row in local_failed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def merge_rank_files(self):
        all_rows, all_failed = [], []
        for rank in range(self.world):
            rows_path = os.path.join(self.args.output_dir, f"_tmp_rows_rank{rank}.jsonl")
            failed_path = os.path.join(self.args.output_dir, f"_tmp_failed_rank{rank}.jsonl")
            with open(rows_path) as f:
                all_rows.extend(json.loads(line) for line in f if line.strip())
            with open(failed_path) as f:
                all_failed.extend(json.loads(line) for line in f if line.strip())
            os.remove(rows_path)
            os.remove(failed_path)
        print(f"[merge] {len(all_rows)} rows  {len(all_failed)} failed/skipped", flush=True)
        return all_rows, all_failed

    def detection_summary(self):
        det_tp = sum(1 for row in self.records if row.get("gt_flag") and row.get("pred_flag"))
        det_fp = sum(1 for row in self.records if not row.get("gt_flag") and row.get("pred_flag"))
        det_fn = sum(1 for row in self.records if row.get("gt_flag") and not row.get("pred_flag"))
        det_tn = sum(1 for row in self.records if not row.get("gt_flag") and not row.get("pred_flag"))
        total = len(self.records)
        det_acc = (det_tp + det_tn) / total if total else 0.0
        det_prec = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
        det_rec = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
        det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0
        return {
            "acc": det_acc,
            "precision": det_prec,
            "recall": det_rec,
            "f1": det_f1,
            "tp": det_tp,
            "fp": det_fp,
            "fn": det_fn,
            "tn": det_tn,
        }

    def print_detection(self, detection):
        print("\n" + "=" * 70)
        print("  DETECTION")
        print(f"  acc={detection['acc']:.4f}  prec={detection['precision']:.4f}  rec={detection['recall']:.4f}  f1={detection['f1']:.4f}")
        print(f"  TP={detection['tp']}  FP={detection['fp']}  FN={detection['fn']}  TN={detection['tn']}")

    def print_by_num_edits(self, by_ne, all_rows):
        print("\n" + "=" * 110)
        print("  LOCALIZATION by num_edits")
        print(f"  {'ne':<6} {'n':>6} {'attr_f1':>8} {'attr_rec':>9} {'attr_jacc':>10} "
              f"{'attr_ada':>9} {'pair_f1':>8} {'pair_ada':>9} {'hard_lv':>8} {'soft_lv':>8}")
        print("-" * 110)
        for ne in sorted(by_ne.keys()):
            self.print_num_edit_row(ne, by_ne[ne])
        hard_all = [row for row in all_rows if row.get("hard_level") is not None]
        print("-" * 110)
        print(f"  {'ALL':<6} {len(all_rows):>6} "
              f"{average([row['attr_f1'] for row in all_rows]):>8.4f} "
              f"{average([row['attr_rec'] for row in all_rows]):>9.4f} "
              f"{average([row['attr_jaccard'] for row in all_rows]):>10.4f} "
              f"{average([row['attr_ada'] for row in all_rows]):>9.4f} "
              f"{average([row['pair_f1'] for row in all_rows]):>8.4f} "
              f"{average([row['pair_ada'] for row in all_rows]):>9.4f} "
              f"{average([row['hard_level'] for row in hard_all]):>8.4f} "
              f"{average([row['soft_level'] for row in hard_all]):>8.4f}")
        print("=" * 110)

    def print_num_edit_row(self, ne, rows):
        hard_rows = [row for row in rows if row.get("hard_level") is not None]
        print(f"  {ne:<6} {len(rows):>6} "
              f"{average([row['attr_f1'] for row in rows]):>8.4f} "
              f"{average([row['attr_rec'] for row in rows]):>9.4f} "
              f"{average([row['attr_jaccard'] for row in rows]):>10.4f} "
              f"{average([row['attr_ada'] for row in rows]):>9.4f} "
              f"{average([row['pair_f1'] for row in rows]):>8.4f} "
              f"{average([row['pair_ada'] for row in rows]):>9.4f} "
              f"{average([row['hard_level'] for row in hard_rows]):>8.4f} "
              f"{average([row['soft_level'] for row in hard_rows]):>8.4f}")

    def print_recovery(self, by_ne, all_rows, all_failed):
        if not self.args.recovery:
            return

        global_rec = recovery_metrics_for_rows(all_rows)
        per_ne_rec = {
            ne: recovery_metrics_for_rows(rows)
            for ne, rows in sorted(by_ne.items())
        }

        print("\n" + "=" * 100)
        print("  RECOVERY METRICS")
        print(f"  n_arc_paired={global_rec['n_arc_paired']}  n_lp={global_rec['n_lp']}  n_failed={len(all_failed)}")
        print("  formula: arc_rate = (mean_level - mean_edited) / (mean_exact - mean_edited)")
        print(f"  {'mean_arc_sim_edited':<35} {global_rec['mean_arc_sim_edited']}")
        print(f"  {'mean_arc_sim_rec_exact (GT alpha)':<35} {global_rec['mean_arc_sim_rec_exact']}")
        print(f"  {'mean_arc_sim_rec_level (pred level)':<35} {global_rec['mean_arc_sim_rec_level']}")
        print(f"  {'arc_recovery_rate_level':<35} {global_rec['arc_recovery_rate_level']}")
        print(f"  {'lp_recovery_rate':<35} {global_rec['lp_recovery_rate']}")
        print(f"\n  {'ne':<10} {'n':>6} {'lp_rate':>8} {'arc_edited':>11} {'arc_exact':>10} {'arc_level':>10} {'arc_rate':>9}")
        print(f"  {'-' * 68}")
        for ne, metrics in sorted(per_ne_rec.items()):
            lp = f"{metrics['lp_recovery_rate']:.4f}" if metrics["lp_recovery_rate"] is not None else "   N/A"
            edited = f"{metrics['mean_arc_sim_edited']:.4f}" if metrics["mean_arc_sim_edited"] is not None else "   N/A"
            exact = f"{metrics['mean_arc_sim_rec_exact']:.4f}" if metrics["mean_arc_sim_rec_exact"] is not None else "   N/A"
            level = f"{metrics['mean_arc_sim_rec_level']:.4f}" if metrics["mean_arc_sim_rec_level"] is not None else "   N/A"
            rate = f"{metrics['arc_recovery_rate_level']:.4f}"
            print(f"  {ne:<10} {metrics['n_arc_paired']:>6} {lp:>8} {edited:>11} {exact:>10} {level:>10} {rate:>9}")
        print("=" * 100)

    def build_attr_summary(self, by_attr):
        print("\n" + "=" * 90)
        print("  PER-ATTRIBUTE PERFORMANCE")
        print(f"  {'attr':<12} {'n_gt':>6} {'tp':>5} {'fp':>5} {'fn':>5} "
              f"{'attr_f1':>8} {'attr_rec':>9} {'action_acc':>11} {'level_acc':>10}")
        print("-" * 90)

        attr_summary = []
        for attr in sorted(by_attr.keys()):
            row = self.attr_summary_row(attr, by_attr[attr])
            attr_summary.append(row)
            print(f"  {attr:<12} {row['n_gt']:>6} {row['tp']:>5} {row['fp']:>5} {row['fn']:>5} "
                  f"{row['attr_f1']:>8.4f} {row['attr_rec']:>9.4f} {row['action_acc']:>11.4f} {row['level_acc']:>10.4f}")
        print("=" * 90)
        return attr_summary

    def attr_summary_row(self, attr, counts):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        n_gt = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        action_acc = counts["action_correct"] / counts["action_total"] if counts["action_total"] > 0 else 0.0
        level_acc = counts["level_correct"] / counts["level_total"] if counts["level_total"] > 0 else 0.0
        return {
            "attr": attr,
            "n_gt": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "attr_f1": round(f1, 4),
            "attr_rec": round(recall, 4),
            "action_acc": round(action_acc, 4),
            "level_acc": round(level_acc, 4),
        }

    def build_summary(self, detection, by_ne, all_rows, attr_summary, all_failed):
        summary = {
            "detection": {
                "acc": round(detection["acc"], 4),
                "precision": round(detection["precision"], 4),
                "recall": round(detection["recall"], 4),
                "f1": round(detection["f1"], 4),
                "tp": detection["tp"],
                "fp": detection["fp"],
                "fn": detection["fn"],
                "tn": detection["tn"],
            },
            "by_num_edits": self.by_num_edits_summary(by_ne, all_rows),
            "per_attr": {row["attr"]: row for row in attr_summary},
        }
        if self.args.recovery:
            summary["recovery_global"] = recovery_metrics_for_rows(all_rows)
            summary["n_recovery_failed"] = len(all_failed)
        return summary

    def by_num_edits_summary(self, by_ne, all_rows):
        summary = {
            str(ne): self.metric_row_summary(rows)
            for ne, rows in sorted(by_ne.items())
        }
        summary["all"] = self.metric_row_summary(all_rows)
        return summary

    def metric_row_summary(self, rows):
        return {
            "n": len(rows),
            "attr_f1": average([row["attr_f1"] for row in rows]),
            "attr_rec": average([row["attr_rec"] for row in rows]),
            "attr_jaccard": average([row["attr_jaccard"] for row in rows]),
            "attr_ada": average([row["attr_ada"] for row in rows]),
            "pair_f1": average([row["pair_f1"] for row in rows]),
            "pair_ada": average([row["pair_ada"] for row in rows]),
            "hard_level": average([row["hard_level"] for row in rows if row.get("hard_level") is not None]),
            "soft_level": average([row["soft_level"] for row in rows if row.get("soft_level") is not None]),
            **(recovery_metrics_for_rows(rows) if self.args.recovery else {}),
        }

    def save_outputs(self, all_rows, all_failed, summary):
        out_path = os.path.join(self.args.output_dir, "analysis_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {out_path}")

        metrics_path = os.path.join(self.args.output_dir, "all_metrics.jsonl")
        with open(metrics_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[saved] {metrics_path}  ({len(all_rows)} rows)")

        if self.args.recovery and all_failed:
            failed_path = os.path.join(self.args.output_dir, "recovery_failed.jsonl")
            with open(failed_path, "w") as f:
                for row in all_failed:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[saved] {failed_path}  ({len(all_failed)} entries)")

    def summarize_and_save(self, all_rows, all_failed):
        detection = self.detection_summary()
        by_attr = rebuild_per_attr(all_rows)
        by_ne = defaultdict(list)
        for row in all_rows:
            by_ne[row["ne"]].append(row)

        self.print_detection(detection)
        self.print_by_num_edits(by_ne, all_rows)
        self.print_recovery(by_ne, all_rows, all_failed)
        attr_summary = self.build_attr_summary(by_attr)
        self.save_outputs(
            all_rows,
            all_failed,
            self.build_summary(detection, by_ne, all_rows, attr_summary, all_failed),
        )

    def run(self):
        self.setup_distributed()
        self.load_records()
        self.load_meta_lookup()
        self.split_records()
        self.warmup_recovery_models()
        self.accelerator.wait_for_everyone()

        local_rows, local_failed = self.process_shard()
        self.write_rank_files(local_rows, local_failed)
        self.accelerator.wait_for_everyone()

        if not self.is_main:
            return
        all_rows, all_failed = self.merge_rank_files()
        self.summarize_and_save(all_rows, all_failed)

