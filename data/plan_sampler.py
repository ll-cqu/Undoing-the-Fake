import json
import random
from collections import Counter, defaultdict

from undo_the_fake.configs.attributes import (
    ALPHA_CONFIGS,
    ATTR_ACTION_CAP,
    ATTRIBUTE_CONFIG,
    BETA_DEFAULT,
    CELEBA_TO_EDIT,
    CELEBA_TRIGGER_PROB,
    FLIP_PROB,
    INCOMPATIBLE_PAIRS,
)


def is_incompatible(selected: list, candidate: dict) -> bool:
    key_set = {(item["attr"], item["action"]) for item in selected}
    candidate_key = (candidate["attr"], candidate["action"])
    for a_attr, a_act, b_attr, b_act in INCOMPATIBLE_PAIRS:
        if candidate_key == (a_attr, a_act) and (b_attr, b_act) in key_set:
            return True
        if candidate_key == (b_attr, b_act) and (a_attr, a_act) in key_set:
            return True
    return False


def alpha_to_level(value: float) -> str:
    if value < 1.20:
        return "moderate"
    if value < 1.60:
        return "strong"
    return "extreme"


def sample_alpha(attr: str) -> tuple:
    spec = ALPHA_CONFIGS.get(attr, ALPHA_CONFIGS["_default"])
    for _ in range(20):
        value = random.gauss(spec["mean"], spec["std"])
        value = round(value / 0.05) * 0.05
        value = round(max(spec["lo"], min(spec["hi"], value)), 2)
        if value >= spec["lo"]:
            break
    return alpha_to_level(value), value


def get_edit_candidates(img_attrs: dict, capped_keys: set) -> list:
    is_male = img_attrs.get("Male", -1) == 1
    candidates = []
    seen_attrs = set()

    for celeba_attr, mapping in CELEBA_TO_EDIT.items():
        if mapping is None or celeba_attr not in img_attrs:
            continue

        attr, action_when_pos, action_when_neg = mapping
        celeba_val = img_attrs[celeba_attr]
        preferred_action = action_when_pos if celeba_val == 1 else action_when_neg
        if preferred_action is None or attr not in ATTRIBUTE_CONFIG or attr in seen_attrs:
            continue

        cfg = ATTRIBUTE_CONFIG[attr]
        if cfg.get("male_only") and not is_male:
            continue

        available = []
        if cfg["can_add"]:
            available.append("add")
        if cfg["can_remove"]:
            available.append("remove")
        if not available:
            continue
        if preferred_action not in available:
            preferred_action = available[0]

        trigger_prob = CELEBA_TRIGGER_PROB.get(celeba_attr, {}).get(celeba_val, 1.0)
        if random.random() > trigger_prob:
            continue

        if cfg["can_add"] and cfg["can_remove"] and random.random() < FLIP_PROB:
            preferred_action = "remove" if preferred_action == "add" else "add"

        if (attr, preferred_action) in capped_keys:
            alternate = "remove" if preferred_action == "add" else "add"
            if cfg.get(f"can_{alternate}") and (attr, alternate) not in capped_keys:
                preferred_action = alternate
            else:
                continue

        candidates.append({
            "attr": attr,
            "action": preferred_action,
            "celeba_attr": celeba_attr,
            "celeba_value": celeba_val,
        })
        seen_attrs.add(attr)

    return candidates


def sample_edit_plan(candidates: list, num_edits: int) -> list:
    random.shuffle(candidates)
    selected = []
    seen_attrs = set()
    for candidate in candidates:
        if len(selected) >= num_edits:
            break
        if candidate["attr"] in seen_attrs or is_incompatible(selected, candidate):
            continue
        selected.append(candidate)
        seen_attrs.add(candidate["attr"])
    return selected


def generate_dataset_plan(annotations: dict, output_json: str, variants_per_image: int = 3, max_images: int = None) -> list:
    variant_num_edits = [(1, 3), (2, 3), (3, 5)][:variants_per_image]
    plans = []
    skipped = 0
    attr_action_counts = defaultdict(int)
    capped_keys = set()

    items = list(annotations.items())
    if max_images:
        items = items[:max_images]
    random.shuffle(items)

    for img_name, img_attrs in items:
        candidates = get_edit_candidates(img_attrs, capped_keys)
        if not candidates:
            skipped += 1
            continue

        img_plans = []
        used_attr_sets = []
        for variant_idx, (min_edits, max_edits) in enumerate(variant_num_edits):
            pool_all = [item for item in candidates if (item["attr"], item["action"]) not in capped_keys]
            fresh = [item for item in pool_all if not any(item["attr"] in used for used in used_attr_sets)]
            pool = fresh if fresh else pool_all
            if not pool:
                continue

            num_edits = min(random.randint(min_edits, max_edits), len(pool))
            edit_plan = sample_edit_plan(pool, num_edits)
            if not edit_plan:
                continue

            for edit in edit_plan:
                key = (edit["attr"], edit["action"])
                attr_action_counts[key] += 1
                cap = ATTR_ACTION_CAP.get(key)
                if cap and attr_action_counts[key] >= cap:
                    capped_keys.add(key)

            used_attr_sets.append({edit["attr"] for edit in edit_plan})
            img_plans.append({
                "image": img_name,
                "variant": variant_idx,
                "num_edits": len(edit_plan),
                "edits": edit_plan,
            })

        if img_plans:
            plans.extend(img_plans)
        else:
            skipped += 1

    print_plan_summary(plans, skipped, attr_action_counts)
    with open(output_json, "w") as f:
        json.dump(plans, f, indent=2)
    return plans


def print_plan_summary(plans, skipped, attr_action_counts):
    total_images = len(set(plan["image"] for plan in plans))
    edit_dist = Counter(plan["num_edits"] for plan in plans)
    variant_dist = Counter(plan["variant"] for plan in plans)

    print(f"\nPlans: {len(plans)} rows ({total_images} images), skipped: {skipped}")
    print(f"Variant distribution: { {f'v{k}': variant_dist[k] for k in sorted(variant_dist)} }")
    print(f"Edit count distribution: { {k: edit_dist[k] for k in sorted(edit_dist)} }")
    print("\n=== (attr, action) distribution ===")
    for attr, action in sorted(attr_action_counts):
        count = attr_action_counts[(attr, action)]
        cap = ATTR_ACTION_CAP.get((attr, action), "unlimited")
        bar = "#" * (count // 500)
        print(f"  {attr:<12} {action:<7} {count:>6}  (cap={cap})  {bar}")


def print_sampling_config():
    print("\n=== Attribute Config ===")
    for attr, cfg in ATTRIBUTE_CONFIG.items():
        dirs = []
        if cfg["can_add"]:
            dirs.append("add")
        if cfg["can_remove"]:
            dirs.append("remove")
        alpha_cfg = ALPHA_CONFIGS.get(attr, ALPHA_CONFIGS["_default"])
        print(
            f"  {attr:<12} [{', '.join(dirs):<14}]"
            f"  beta={cfg.get('beta_override', BETA_DEFAULT)}"
            f"  alpha=[{alpha_cfg['lo']},{alpha_cfg['hi']}] mean={alpha_cfg['mean']}"
            + ("  [male_only]" if cfg.get("male_only") else "")
        )

    print("\n=== Alpha Groups ===")
    print("  Group A (smile/chubby/eye_big) : [0.80, 1.50]  moderate~40%  strong~55%  extreme~5%")
    print("  Group B (hair/beard/makeup/age): [1.00, 2.00]  moderate~10%  strong~50%  extreme~40%")
    print("  skin_tan                       : [0.80, 2.00]  moderate~25%  strong~45%  extreme~30%")
    print("  Levels: moderate(<1.20)  strong(<1.60)  extreme(>=1.60)")

    print("\n=== Trigger Prob Gates ===")
    for celeba_attr, val_map in CELEBA_TRIGGER_PROB.items():
        for value, prob in val_map.items():
            print(f"  {celeba_attr:<22} val={value:>2}  prob={prob}")

