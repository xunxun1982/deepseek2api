# utils/config.py
import json
import logging

logger = logging.getLogger("config")

CONFIG_PATH = "config.json"

def load_config():
    """从 config.json 加载配置，出错则返回空 dict"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[load_config] 无法读取配置文件: {e}")
        return {}

def save_config(cfg):
    """将配置写回 config.json"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info("[save_config] 配置已写回 config.json")
    except Exception as e:
        logger.error(f"[save_config] 写入 config.json 失败: {e}")

CONFIG = load_config()