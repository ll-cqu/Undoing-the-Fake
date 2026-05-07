import json
import re


def parse_output(text):
    try:
        if not isinstance(text, str):
            return None, []
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, []
        data = json.loads(match.group())
        manipulations = []
        for item in data.get("manipulations", []):
            if not isinstance(item, dict):
                continue
            attr = item.get("attr")
            action = item.get("action")
            level = item.get("alpha_level", "")
            if attr and action:
                manipulations.append({"attr": attr, "action": action, "alpha_level": level})
        return data.get("is_manipulated", None), manipulations
    except Exception:
        return None, []


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    return ""


def extract_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(x.get("text", "") for x in content if x.get("type") == "text")
        return " ".join(parts)
    return str(completion)


def normalize_messages(messages):
    for msg in messages:
        if isinstance(msg["content"], str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
    return messages

