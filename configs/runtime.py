import os


WORKSPACE = os.environ.get("UTF_WORKSPACE", ".")
FACE_EDITING_ROOT = os.environ.get("UTF_FACE_EDITING_ROOT", "face_editing")

BASE_MODEL = os.environ.get("UTF_BASE_MODEL", "hf_models/Qwen3-VL-8B-Instruct")
FACE_DETECT_BASE_MODEL = os.environ.get("UTF_FACE_DETECT_BASE_MODEL", BASE_MODEL)

WARMUP_CHECKPOINT = os.environ.get("UTF_WARMUP_CHECKPOINT", "output_file/warmup_checkpoint")
STYLECLIP_ROOT = os.environ.get("UTF_STYLECLIP_ROOT", "StyleCLIP")
E4E_ROOT = os.environ.get("UTF_E4E_ROOT", f"{WORKSPACE}/face_editing/encoder4editing")
STYLEGAN_PKL = os.environ.get("UTF_STYLEGAN_PKL", f"{STYLECLIP_ROOT}/global_torch/model/ffhq.pkl")
FS3_PATH = os.environ.get("UTF_FS3_PATH", f"{STYLECLIP_ROOT}/global_torch/npy/ffhq/fs3.npy")
E4E_CKPT = os.environ.get("UTF_E4E_CKPT", f"{WORKSPACE}/face_editing/pretrained/e4e_ffhq_encode.pt")

LOCAL_STYLECLIP_ROOT = "face_editing/StyleCLIP"
LOCAL_E4E_ROOT = "face_editing/encoder4editing"
LOCAL_STYLEGAN_PKL = "face_editing/StyleCLIP/global_torch/model/ffhq.pkl"
LOCAL_FS3_PATH = "face_editing/StyleCLIP/global_torch/npy/ffhq/fs3.npy"
LOCAL_E4E_CKPT = "face_editing/pretrained/e4e_ffhq_encode.pt"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = (
    "You are a face manipulation detection expert. "
    "Given an image, determine whether it has been manipulated. "
    "If manipulated, identify each altered facial attribute, "
    "the direction of change (add/remove), and the manipulation strength "
    "(moderate/strong/extreme). "
    "Respond ONLY with a valid JSON object."
)

USER_PROMPT = (
    "Perform a deepfake detection analysis on this face image. "
    "Report whether manipulation is detected and describe each altered attribute "
    "with its action (add/remove) and alpha level (moderate/strong/extreme).\n"
    'Output format — JSON only: {"is_manipulated": bool, "manipulations": '
    '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
)

USER_PROMPTS = [
    (
        "Analyze this face image and determine if it has been manipulated. "
        "If so, identify each altered attribute, the direction (add/remove), "
        "and strength (moderate/strong/extreme).\n"
        'Reply with JSON only: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    (
        "Is this face image authentic or has it been edited? "
        "List any manipulated facial attributes with their edit direction and intensity.\n"
        'Output JSON: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    (
        "Detect facial attribute manipulations in this image. "
        "Return a JSON object with is_manipulated (bool) and a list of manipulations, "
        "each with attr, action (add/remove), and alpha_level (moderate/strong/extreme)."
    ),
    (
        "This face image may have been tampered with. "
        "Identify all edited facial attributes so that the edits can be reversed. "
        "For each, provide the attribute name, the edit action (add/remove), "
        "and the manipulation strength (moderate/strong/extreme).\n"
        'JSON output only: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    USER_PROMPT,
]
