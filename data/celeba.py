from undo_the_fake.configs.attributes import CELEBA_ATTRS


def parse_celeba_annotations(anno_file: str) -> dict:
    annotations = {}
    with open(anno_file) as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != len(CELEBA_ATTRS) + 1:
            continue
        img_name = parts[0]
        annotations[img_name] = {
            CELEBA_ATTRS[i]: int(parts[i + 1])
            for i in range(len(CELEBA_ATTRS))
        }

    print(f"Parsed annotations: {len(annotations)} images")
    return annotations

