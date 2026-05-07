BETA_DEFAULT = 0.10

ALPHA_LEVEL_TO_MIDPOINT = {
    "moderate": 1.00,
    "strong": 1.40,
    "extreme": 1.80,
}

LEVEL_ORDER = {"moderate": 0, "strong": 1, "extreme": 2}

ATTRIBUTE_CONFIG = {
    "smile": {"neutral": "face without smile", "target": "smiling face", "can_add": True, "can_remove": True},
    "hair_curly": {
        "neutral": "face with straight hair",
        "target": "face with curly hair",
        "can_add": True,
        "can_remove": True,
        "beta_override": 0.08,
    },
    "hair_bangs": {
        "neutral": "face without bangs",
        "target": "face with bangs",
        "can_add": True,
        "can_remove": True,
        "beta_override": 0.08,
    },
    "beard": {
        "neutral": "face without beard",
        "target": "face with beard",
        "can_add": True,
        "can_remove": True,
        "male_only": True,
    },
    "makeup": {"neutral": "face without makeup", "target": "face with full makeup", "can_add": True, "can_remove": True},
    "age": {"neutral": "young face", "target": "old face", "can_add": True, "can_remove": True},
    "skin_tan": {"neutral": "face with fair skin", "target": "face with tanned skin", "can_add": True, "can_remove": False},
    "chubby": {"neutral": "face with normal face shape", "target": "chubby face", "can_add": True, "can_remove": True},
    "eye_big": {
        "neutral": "face with normal sized eyes",
        "target": "face with big eyes",
        "can_add": True,
        "can_remove": True,
        "beta_override": 0.09,
    },
}

ALPHA_CONFIGS = {
    "smile": {"mean": 1.10, "std": 0.25, "lo": 0.80, "hi": 1.50},
    "chubby": {"mean": 1.10, "std": 0.25, "lo": 0.80, "hi": 1.50},
    "eye_big": {"mean": 1.10, "std": 0.25, "lo": 0.80, "hi": 1.50},
    "hair_curly": {"mean": 1.50, "std": 0.30, "lo": 1.00, "hi": 2.00},
    "hair_bangs": {"mean": 1.50, "std": 0.30, "lo": 1.00, "hi": 2.00},
    "beard": {"mean": 1.50, "std": 0.30, "lo": 1.00, "hi": 2.00},
    "makeup": {"mean": 1.50, "std": 0.30, "lo": 1.00, "hi": 2.00},
    "age": {"mean": 1.50, "std": 0.30, "lo": 1.00, "hi": 2.00},
    "skin_tan": {"mean": 1.30, "std": 0.30, "lo": 0.80, "hi": 2.00},
    "_default": {"mean": 1.30, "std": 0.30, "lo": 0.80, "hi": 2.00},
}

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]

CELEBA_TO_EDIT = {
    "Smiling": ("smile", "remove", "add"),
    "Straight_Hair": ("hair_curly", "add", "remove"),
    "Wavy_Hair": ("hair_curly", "remove", "add"),
    "Bangs": ("hair_bangs", "remove", "add"),
    "Goatee": ("beard", "remove", "add"),
    "Mustache": ("beard", "remove", "add"),
    "Sideburns": ("beard", "remove", "add"),
    "5_o_Clock_Shadow": ("beard", "remove", "add"),
    "No_Beard": ("beard", "add", "remove"),
    "Heavy_Makeup": ("makeup", "remove", "add"),
    "Wearing_Lipstick": ("makeup", "remove", "add"),
    "Young": ("age", "add", "remove"),
    "Chubby": ("chubby", "remove", "add"),
    "Double_Chin": ("chubby", "remove", "add"),
    "Narrow_Eyes": ("eye_big", "add", "add"),
    "Pale_Skin": ("skin_tan", "add", None),
    "Big_Lips": None,
    "Big_Nose": None,
    "Pointy_Nose": None,
    "Bald": None,
    "Receding_Hairline": None,
    "Black_Hair": None,
    "Blond_Hair": None,
    "Brown_Hair": None,
    "Gray_Hair": None,
    "Arched_Eyebrows": None,
    "Attractive": None,
    "Bags_Under_Eyes": None,
    "Blurry": None,
    "Bushy_Eyebrows": None,
    "Eyeglasses": None,
    "High_Cheekbones": None,
    "Male": None,
    "Mouth_Slightly_Open": None,
    "Oval_Face": None,
    "Rosy_Cheeks": None,
    "Wearing_Earrings": None,
    "Wearing_Hat": None,
    "Wearing_Necklace": None,
    "Wearing_Necktie": None,
}

CELEBA_TRIGGER_PROB = {
    "Young": {1: 0.50},
    "Smiling": {-1: 0.70},
    "No_Beard": {1: 0.50},
    "Bangs": {-1: 0.40},
}

ATTR_ACTION_CAP = {
    ("smile", "add"): 13000,
    ("smile", "remove"): 13000,
    ("hair_curly", "add"): 13000,
    ("hair_curly", "remove"): 13000,
    ("hair_bangs", "add"): 13000,
    ("hair_bangs", "remove"): 13000,
    ("beard", "add"): 13000,
    ("beard", "remove"): 13000,
    ("makeup", "add"): 13000,
    ("makeup", "remove"): 13000,
    ("age", "add"): 13000,
    ("age", "remove"): 13000,
    ("skin_tan", "add"): 13000,
    ("chubby", "add"): 13000,
    ("chubby", "remove"): 13000,
    ("eye_big", "add"): 13000,
    ("eye_big", "remove"): 13000,
}

INCOMPATIBLE_PAIRS = []
FLIP_PROB = 0.30

CELEBA_TRIGGER_PROB = {
    "Young": {1: 0.50},
    "Smiling": {-1: 0.70},
    "No_Beard": {1: 0.50},
    "Bangs": {-1: 0.40},
}

ATTR_ACTION_CAP = {
    ("smile", "add"): 13000,
    ("smile", "remove"): 13000,
    ("hair_curly", "add"): 13000,
    ("hair_curly", "remove"): 13000,
    ("hair_bangs", "add"): 13000,
    ("hair_bangs", "remove"): 13000,
    ("beard", "add"): 13000,
    ("beard", "remove"): 13000,
    ("makeup", "add"): 13000,
    ("makeup", "remove"): 13000,
    ("age", "add"): 13000,
    ("age", "remove"): 13000,
    ("skin_tan", "add"): 13000,
    ("chubby", "add"): 13000,
    ("chubby", "remove"): 13000,
    ("eye_big", "add"): 13000,
    ("eye_big", "remove"): 13000,
}
