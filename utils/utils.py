# utils/utils.py
import re

def messages_prepare(messages: list) -> str:
    """处理消息列表，将多轮对话合并为最终 prompt，并添加角色标签"""
    processed = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if item.get("type") == "text"]
            text = "\n".join(texts)
        else:
            text = str(content)
        processed.append({"role": role, "text": text})
    if not processed:
        return ""
    merged = [processed[0]]
    for msg in processed[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["text"] += "\n\n" + msg["text"]
        else:
            merged.append(msg)
    parts = []
    for idx, block in enumerate(merged):
        role = block["role"]
        text = block["text"]
        if role == "assistant":
            parts.append(f"<｜Assistant｜>{text}<｜end▁of▁sentence｜>")
        elif role in ("user", "system"):
            if idx > 0:
                parts.append(f"<｜User｜>{text}")
            else:
                parts.append(text)
        else:
            parts.append(text)
    final_prompt = "".join(parts)
    final_prompt = re.sub(r"!", "", final_prompt)
    return final_prompt