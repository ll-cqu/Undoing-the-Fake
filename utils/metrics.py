import json
import re
from collections import defaultdict

import numpy as np

from undo_the_fake.configs.attributes import LEVEL_ORDER


def average(values, default=0.0):
    return round(float(np.mean(values)), 4) if values else default


def detection_acc(pred_flag, gt_flag) -> float:
    if pred_flag is None or gt_flag is None:
        return 0.0
    return 1.0 if pred_flag == gt_flag else 0.0


def json_format_reward(pred_text: str) -> float:
    try:
        match = re.search(r"\{.*\}", pred_text, re.DOTALL)
        if not match:
            return 0.0
        data = json.loads(match.group())
        if "is_manipulated" not in data:
            return 0.5
        if data["is_manipulated"] and "manipulations" not in data:
            return 0.5
        return 1.0
    except Exception:
        return 0.0


def hierarchical_metrics(pred_list: list, gt_list: list) -> dict:
    gt_attrs = {m["attr"] for m in gt_list}
    pred_attrs = {m["attr"] for m in pred_list}
    tp_attrs = gt_attrs & pred_attrs
    attr_precision = len(tp_attrs) / len(pred_attrs) if pred_attrs else 0.0
    attr_recall = len(tp_attrs) / len(gt_attrs) if gt_attrs else 1.0
    denom = attr_precision + attr_recall
    attr_f1 = 2 * attr_precision * attr_recall / denom if denom > 0 else 0.0

    gt_action = {m["attr"]: m["action"] for m in gt_list}
    pred_action = {m["attr"]: m["action"] for m in pred_list}
    action_correct = {a for a in tp_attrs if pred_action.get(a) == gt_action[a]}
    action_acc = len(action_correct) / len(tp_attrs) if tp_attrs else 0.0

    gt_level = {m["attr"]: m["alpha_level"] for m in gt_list}
    pred_level = {m["attr"]: m["alpha_level"] for m in pred_list}
    level_acc = (
        sum(pred_level.get(a) == gt_level[a] for a in action_correct) / len(action_correct)
        if action_correct else 0.0
    )

    return {
        "attr_precision": round(attr_precision, 4),
        "attr_recall": round(attr_recall, 4),
        "attr_f1": round(attr_f1, 4),
        "action_acc": round(action_acc, 4),
        "level_acc": round(level_acc, 4),
    }


def hier_metrics_from_lists(pred_list, gt_attrs, gt_actions, gt_levels):
    gt_set = set(gt_attrs)
    pred_set = {m["attr"] for m in pred_list if m.get("attr")}
    tp = gt_set & pred_set
    prec = len(tp) / len(pred_set) if pred_set else 0.0
    rec = len(tp) / len(gt_set) if gt_set else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    gt_act = dict(zip(gt_attrs, gt_actions))
    pd_act = {m["attr"]: m["action"] for m in pred_list}
    ok_act = {a for a in tp if pd_act.get(a) == gt_act[a]}
    act_acc = len(ok_act) / len(tp) if tp else 0.0

    gt_lv = dict(zip(gt_attrs, gt_levels))
    pd_lv = {m["attr"]: m.get("alpha_level") for m in pred_list}
    lv_acc = sum(pd_lv.get(a) == gt_lv[a] for a in ok_act) / len(ok_act) if ok_act else 0.0
    return f1, act_acc, lv_acc


def attr_pair_level_metrics(rec, parse_output):
    if rec.get("gt_attrs") is not None:
        gt_attr_list = rec.get("gt_attrs", [])
        gt_act_list = rec.get("gt_actions", [])
        gt_lv_list = rec.get("gt_levels", [])
    else:
        _, gt_m = parse_output(rec.get("gt_text", ""))
        gt_attr_list = [m["attr"] for m in gt_m]
        gt_act_list = [m["action"] for m in gt_m]
        gt_lv_list = [m.get("alpha_level", "") for m in gt_m]

    pred_m = rec.get("pred_manips")
    if not pred_m:
        _, pred_m = parse_output(rec.get("pred_text", ""))
    pred_m = [x for x in (pred_m or []) if isinstance(x, dict)]

    gt_attrs = set(gt_attr_list)
    pred_attrs = {m["attr"] for m in pred_m if m.get("attr")}
    gt_act_map = dict(zip(gt_attr_list, gt_act_list))
    pred_act_map = {m["attr"]: m["action"] for m in pred_m if m.get("attr")}
    gt_lv_map = dict(zip(gt_attr_list, gt_lv_list))
    pred_lv_map = {m["attr"]: m.get("alpha_level", "") for m in pred_m if m.get("attr")}

    tp_attrs = gt_attrs & pred_attrs
    fp_attrs = pred_attrs - gt_attrs
    fn_attrs = gt_attrs - pred_attrs
    ne = rec.get("num_edits") or len(gt_attr_list)

    tp_a = len(tp_attrs)
    fp_a = len(fp_attrs)
    fn_a = len(fn_attrs)
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
    rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
    attr_f1 = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0.0
    attr_jaccard = tp_a / (tp_a + fp_a + fn_a) if (tp_a + fp_a + fn_a) > 0 else 1.0
    attr_ada = tp_a / max(len(pred_attrs), len(gt_attrs), 1)

    gt_pairs = {(a, gt_act_map[a]) for a in gt_attrs}
    pred_pairs = {(a, pred_act_map[a]) for a in pred_attrs if a in pred_act_map}
    tp_p = len(gt_pairs & pred_pairs)
    fp_p = len(pred_pairs - gt_pairs)
    fn_p = len(gt_pairs - pred_pairs)
    prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
    rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
    pair_f1 = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0
    pair_ada = tp_p / max(len(pred_pairs), len(gt_pairs), 1)

    hit_pairs = gt_pairs & pred_pairs
    gt_lv_pair = {(a, gt_act_map[a]): gt_lv_map[a] for a in gt_attrs}
    hard_scores, soft_scores = [], []
    for pair in hit_pairs:
        gt_lv = gt_lv_pair[pair]
        pred_lv = pred_lv_map.get(pair[0], "")
        hard_scores.append(1.0 if pred_lv == gt_lv else 0.0)
        if pred_lv == gt_lv:
            soft_scores.append(1.0)
        elif pred_lv in LEVEL_ORDER and gt_lv in LEVEL_ORDER:
            soft_scores.append(max(0.0, 1.0 - 0.5 * abs(LEVEL_ORDER[pred_lv] - LEVEL_ORDER[gt_lv])))
        else:
            soft_scores.append(0.0)

    action_correct_attrs = {a for a in tp_attrs if pred_act_map.get(a) == gt_act_map.get(a)}
    level_correct_attrs = {a for a in action_correct_attrs if pred_lv_map.get(a) == gt_lv_map.get(a)}

    return {
        "gt_attr_list": gt_attr_list,
        "gt_act_list": gt_act_list,
        "gt_lv_list": gt_lv_list,
        "pred_m": pred_m,
        "gt_attrs": gt_attrs,
        "pred_attrs": pred_attrs,
        "gt_act_map": gt_act_map,
        "pred_act_map": pred_act_map,
        "gt_lv_map": gt_lv_map,
        "pred_lv_map": pred_lv_map,
        "tp_attrs": tp_attrs,
        "fp_attrs": fp_attrs,
        "fn_attrs": fn_attrs,
        "gt_pairs": gt_pairs,
        "pred_pairs": pred_pairs,
        "pair_f1": pair_f1,
        "ne": ne,
        "action_correct_attrs": action_correct_attrs,
        "level_correct_attrs": level_correct_attrs,
        "row": {
            "ne": int(ne),
            "attr_f1": attr_f1,
            "attr_rec": rec_a,
            "attr_jaccard": attr_jaccard,
            "attr_ada": attr_ada,
            "pair_f1": pair_f1,
            "pair_ada": pair_ada,
            "hard_level": float(np.mean(hard_scores)) if hard_scores else None,
            "soft_level": float(np.mean(soft_scores)) if soft_scores else None,
        },
    }


def recovery_metrics_for_rows(rows):
    paired = [
        r for r in rows
        if r.get("arc_sim_edited") is not None
        and r.get("arc_sim_rec_level") is not None
        and r.get("arc_sim_rec_exact") is not None
    ]
    lp_rows = [r for r in rows if r.get("lp_recovery_rate") is not None]

    mean_edited = average([r["arc_sim_edited"] for r in paired], default=None)
    mean_rec_exact = average([r["arc_sim_rec_exact"] for r in paired], default=None)
    mean_rec_level = average([r["arc_sim_rec_level"] for r in paired], default=None)

    denom = (
        mean_rec_exact - mean_edited
        if mean_edited is not None and mean_rec_exact is not None else None
    )
    arc_recovery_rate_level = (
        round((mean_rec_level - mean_edited) / denom, 4)
        if denom and denom > 1e-4 else 0.0
    )
    return {
        "n_total": len(rows),
        "n_arc_paired": len(paired),
        "n_lp": len(lp_rows),
        "lp_recovery_rate": average([r["lp_recovery_rate"] for r in lp_rows], default=None),
        "mean_arc_sim_edited": mean_edited,
        "mean_arc_sim_rec_exact": mean_rec_exact,
        "mean_arc_sim_rec_level": mean_rec_level,
        "arc_recovery_rate_level": arc_recovery_rate_level,
        "arc_recovery_rate_exact": 1.0 if denom and denom > 1e-4 else 0.0,
    }


def rebuild_per_attr(rows):
    by_attr = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "action_correct": 0, "action_total": 0,
        "level_correct": 0, "level_total": 0,
    })
    for row in rows:
        for attr in row.get("tp_attrs", []):
            by_attr[attr]["tp"] += 1
        for attr in row.get("fp_attrs", []):
            by_attr[attr]["fp"] += 1
        for attr in row.get("fn_attrs", []):
            by_attr[attr]["fn"] += 1
        for attr in row.get("tp_attrs", []):
            by_attr[attr]["action_total"] += 1
        for attr in row.get("action_correct_attrs", []):
            by_attr[attr]["action_correct"] += 1
        for attr in row.get("action_correct_attrs", []):
            by_attr[attr]["level_total"] += 1
        for attr in row.get("level_correct_attrs", []):
            by_attr[attr]["level_correct"] += 1
    return by_attr
