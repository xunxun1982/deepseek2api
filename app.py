#!/usr/bin/env python3
import json
import base64
import time
import logging
from curl_cffi import requests
import random
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, g
import os
import struct
import ctypes
from wasmtime import Store, Module, Linker
import re
import transformers

# -------------------------- 初始化 tokenizer --------------------------
chat_tokenizer_dir = "./"  # 请确保目录下有正确的 tokenizer 配置文件或模型文件
tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)

# ----------------------------------------------------------------------
# =========================== 日志配置 ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
app = Flask(__name__)

# -------------------- 全局添加 CORS 支持 --------------------
@app.before_request
def handle_options_request():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ----------------------------------------------------------------------
# 全局集合：记录当前正在对话中的账号（以 email 或 phone 标识），保证同一账号同时只进行一个对话
active_accounts = set()

# ----------------------------------------------------------------------
# (1) 配置文件的读写函数
# ----------------------------------------------------------------------
CONFIG_PATH = "config.json"

def load_config():
    """从 config.json 加载配置，出错则返回空 dict"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.warning(f"[load_config] 无法读取配置文件: {e}")
        return {}

def save_config(cfg):
    """将配置写回 config.json"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        app.logger.info("[save_config] 配置已写回 config.json")
    except Exception as e:
        app.logger.error(f"[save_config] 写入 config.json 失败: {e}")

CONFIG = load_config()

# ----------------------------------------------------------------------
# (2) DeepSeek 相关常量
# ----------------------------------------------------------------------
DEEPSEEK_HOST = "chat.deepseek.com"

DEEPSEEK_LOGIN_URL = f"https://{DEEPSEEK_HOST}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/completion"

BASE_HEADERS = {
    'Host': "chat.deepseek.com",
    'User-Agent': "DeepSeek/1.0.7 Android/34",
    'Accept': "application/json",
    'Accept-Encoding': "gzip",
    'Content-Type': "application/json",
    'x-client-platform': "android",
    'x-client-version': "1.0.7",
    'x-client-locale': "zh_CN",
    'x-rangers-id': "7883327620434123524",
    'accept-charset': "UTF-8",
}

# WASM 模块文件路径（请确保文件存在）
WASM_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"

# ----------------------------------------------------------------------
# 辅助函数：获取账号唯一标识（优先 email，否则 mobile）
# ----------------------------------------------------------------------
def get_account_identifier(account):
    """返回账号的唯一标识，优先使用 email，否则使用 mobile"""
    return account.get("email", "").strip() or account.get("mobile", "").strip()

# ----------------------------------------------------------------------
# (3) 登录函数：支持使用 email 或 mobile 登录
# ----------------------------------------------------------------------
def login_deepseek_via_account(account):
    """使用 account 中的 email 或 mobile 登录 DeepSeek，
    成功后将返回的 token 写入 account 并保存至配置文件，返回新 token。"""
    email = account.get("email", "").strip()
    mobile = account.get("mobile", "").strip()
    password = account.get("password", "").strip()
    if not password or (not email and not mobile):
        raise ValueError("账号缺少必要的登录信息（必须提供 email 或 mobile 以及 password）")
    
    if email:
        app.logger.info(f"[login_deepseek_via_account] 正在使用 email 登录账号：{email}")
        payload = {
            "email": email,
            "mobile": "",
            "password": password,
            "area_code": "",
            "device_id": "deepseek_to_api",
            "os": "android"
        }
    else:
        app.logger.info(f"[login_deepseek_via_account] 正在使用 mobile 登录账号：{mobile}")
        payload = {
            "mobile": mobile,
            "area_code": None,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android"
        }
    
    # 增加 timeout 参数，防止请求阻塞过久
    resp = requests.post(DEEPSEEK_LOGIN_URL, headers=BASE_HEADERS, json=payload, timeout=30)
    app.logger.debug(f"[login_deepseek_via_account] 状态码: {resp.status_code}")
    app.logger.debug(f"[login_deepseek_via_account] 响应体: {resp.text}")
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise ValueError(f"登录失败, code={data.get('code')}, msg={data.get('msg')}")
    
    new_token = data["data"]["biz_data"]["user"]["token"]
    account["token"] = new_token
    save_config(CONFIG)
    identifier = email if email else mobile
    app.logger.info(f"[login_deepseek_via_account] 成功登录账号 {identifier}，token: {new_token}")
    return new_token

# ----------------------------------------------------------------------
# (4) 从 accounts 中随机选择一个未忙且未尝试过的账号
# ----------------------------------------------------------------------
def choose_new_account(exclude_ids):
    accounts = CONFIG.get("accounts", [])
    available = [
        acc for acc in accounts
        if get_account_identifier(acc) not in exclude_ids and get_account_identifier(acc) not in active_accounts
    ]
    if available:
        chosen = random.choice(available)
        app.logger.info(f"[choose_new_account] 新选择账号: {get_account_identifier(chosen)}")
        return chosen
    app.logger.warning("[choose_new_account] 没有可用的账号")
    return None

# ----------------------------------------------------------------------
# (5) 判断调用模式：配置模式 vs 用户自带 token
# ----------------------------------------------------------------------
def determine_mode_and_token():
    """根据请求头 Authorization 判断使用哪种模式：
    - 如果 Bearer token 出现在 CONFIG["keys"] 中，则为配置模式，从 CONFIG["accounts"] 中随机选择一个账号（排除已尝试账号），
      检查该账号是否已有 token，否则调用登录接口获取；
    - 否则，直接使用请求中的 Bearer 值作为 DeepSeek token。
    结果存入 g.deepseek_token；配置模式下同时存入 g.account 与 g.tried_accounts。
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return Response(json.dumps({"error": "Unauthorized: missing Bearer token."}),
                        status=401, mimetype="application/json")
    caller_key = auth_header.replace("Bearer ", "", 1).strip()
    config_keys = CONFIG.get("keys", [])
    if caller_key in config_keys:
        g.use_config_token = True
        g.tried_accounts = []  # 初始化已尝试账号
        selected_account = choose_new_account(g.tried_accounts)
        if not selected_account:
            return Response(json.dumps({"error": "No accounts configured."}),
                            status=500, mimetype="application/json")
        if not selected_account.get("token", "").strip():
            try:
                login_deepseek_via_account(selected_account)
            except Exception as e:
                app.logger.error(f"[determine_mode_and_token] 账号 {get_account_identifier(selected_account)} 登录失败：{e}")
                return Response(json.dumps({"error": "Account login failed."}),
                                status=500, mimetype="application/json")
        else:
            app.logger.info(f"[determine_mode_and_token] 账号 {get_account_identifier(selected_account)} 已有 token，无需重新登录")
        g.deepseek_token = selected_account.get("token")
        g.account = selected_account
        app.logger.info(f"[determine_mode_and_token] 配置模式：使用账号 {get_account_identifier(selected_account)} 的 token")
    else:
        g.use_config_token = False
        g.deepseek_token = caller_key
        app.logger.info("[determine_mode_and_token] 使用用户自带 DeepSeek token")
    return None

def get_auth_headers():
    """返回 DeepSeek 请求所需的公共请求头"""
    return { **BASE_HEADERS, "authorization": f"Bearer {g.deepseek_token}" }

# ----------------------------------------------------------------------
# (6) 封装对话接口调用的重试机制
# ----------------------------------------------------------------------
def call_completion_endpoint(payload, headers, stream, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            deepseek_resp = requests.post(DEEPSEEK_COMPLETION_URL, headers=headers, json=payload, stream=True)
        except Exception as e:
            app.logger.warning(f"[call_completion_endpoint] 请求异常: {e}")
            time.sleep(1)
            attempts += 1
            continue
        if deepseek_resp.status_code == 200:
            return deepseek_resp
        else:
            app.logger.warning(f"[call_completion_endpoint] 调用对话接口失败, 状态码: {deepseek_resp.status_code}")
            deepseek_resp.close()
            time.sleep(1)
            attempts += 1
    return None

# ----------------------------------------------------------------------
# (7) 创建会话 & 获取 PoW（重试时，配置模式下错误会切换账号；用户自带 token 模式下仅重试）
# ----------------------------------------------------------------------
def create_session(max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers()
        try:
            resp = requests.post(DEEPSEEK_CREATE_SESSION_URL, headers=headers, json={"agent": "chat"}, timeout=30)
        except Exception as e:
            app.logger.error(f"[create_session] 请求异常: {e}")
            attempts += 1
            continue
        try:
            data = resp.json()
        except Exception as e:
            app.logger.error(f"[create_session] JSON解析异常: {e}")
            data = {}
        if resp.status_code == 200 and data.get("code") == 0:
            session_id = data["data"]["biz_data"]["id"]
            app.logger.info(f"[create_session] 新会话 chat_session_id={session_id}")
            resp.close()
            return session_id
        else:
            code = data.get("code")
            app.logger.warning(f"[create_session] 创建会话失败, code={code}, msg={data.get('msg')}")
            resp.close()
            if g.use_config_token:
                current_id = get_account_identifier(g.account)
                if not hasattr(g, 'tried_accounts'):
                    g.tried_accounts = []
                if current_id not in g.tried_accounts:
                    g.tried_accounts.append(current_id)
                new_account = choose_new_account(g.tried_accounts)
                if new_account is None:
                    break
                try:
                    login_deepseek_via_account(new_account)
                except Exception as e:
                    app.logger.error(f"[create_session] 账号 {get_account_identifier(new_account)} 登录失败：{e}")
                    attempts += 1
                    continue
                g.account = new_account
                g.deepseek_token = new_account.get("token")
            else:
                attempts += 1
                continue
        attempts += 1
    return None

# ----------------------------------------------------------------------
# (7.1) 使用 WASM 模块计算 PoW 答案的辅助函数
# ----------------------------------------------------------------------
def compute_pow_answer(algorithm: str,
                       challenge_str: str,
                       salt: str,
                       difficulty: int,
                       expire_at: int,
                       signature: str,
                       target_path: str,
                       wasm_path: str) -> int:
    """
    使用 WASM 模块计算 DeepSeekHash 答案（answer）。
    根据 JS 逻辑：
      - 拼接前缀： "{salt}_{expire_at}_"
      - 将 challenge 与前缀写入 wasm 内存后调用 wasm_solve 进行求解，
      - 从 wasm 内存中读取状态与求解结果，
      - 若状态非 0，则返回整数形式的答案，否则返回 None。
    """
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"不支持的算法：{algorithm}")

    prefix = f"{salt}_{expire_at}_"

    # --- 加载 wasm 模块 ---
    store = Store()
    linker = Linker(store.engine)
    try:
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()
    except Exception as e:
        raise RuntimeError(f"加载 wasm 文件失败: {wasm_path}, 错误: {e}")
    module = Module(store.engine, wasm_bytes)
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    try:
        memory = exports["memory"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as e:
        raise RuntimeError(f"缺少 wasm 导出函数: {e}")

    def write_memory(offset: int, data: bytes):
        size = len(data)
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, size)

    def read_memory(offset: int, size: int) -> bytes:
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)

    def encode_string(text: str):
        data = text.encode("utf-8")
        length = len(data)
        ptr_val = alloc(store, length, 1)
        ptr = int(ptr_val.value) if hasattr(ptr_val, "value") else int(ptr_val)
        write_memory(ptr, data)
        return ptr, length

    # 1. 申请 16 字节栈空间
    retptr = add_to_stack(store, -16)
    # 2. 编码 challenge 与 prefix 到 wasm 内存中
    ptr_challenge, len_challenge = encode_string(challenge_str)
    ptr_prefix, len_prefix = encode_string(prefix)
    # 3. 调用 wasm_solve（注意：difficulty 以 float 形式传入）
    wasm_solve(store, retptr, ptr_challenge, len_challenge, ptr_prefix, len_prefix, float(difficulty))
    # 4. 从 retptr 处读取 4 字节状态和 8 字节求解结果
    status_bytes = read_memory(retptr, 4)
    if len(status_bytes) != 4:
        add_to_stack(store, 16)
        raise RuntimeError("读取状态字节失败")
    status = struct.unpack("<i", status_bytes)[0]
    value_bytes = read_memory(retptr + 8, 8)
    if len(value_bytes) != 8:
        add_to_stack(store, 16)
        raise RuntimeError("读取结果字节失败")
    value = struct.unpack("<d", value_bytes)[0]
    # 5. 恢复栈指针
    add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)

# ----------------------------------------------------------------------
# (7.2) 获取 PoW 响应，融合计算 answer 逻辑
# ----------------------------------------------------------------------
def get_pow_response(max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers()
        try:
            resp = requests.post(DEEPSEEK_CREATE_POW_URL, headers=headers, json={"target_path": "/api/v0/chat/completion"}, timeout=30)
        except Exception as e:
            app.logger.error(f"[get_pow_response] 请求异常: {e}")
            attempts += 1
            continue
        try:
            data = resp.json()
        except Exception as e:
            app.logger.error(f"[get_pow_response] JSON解析异常: {e}")
            data = {}
        if resp.status_code == 200 and data.get("code") == 0:
            challenge = data["data"]["biz_data"]["challenge"]
            difficulty = challenge.get("difficulty", 144000)
            expire_at = challenge.get("expire_at", 1680000000)
            try:
                answer = compute_pow_answer(
                    challenge["algorithm"],
                    challenge["challenge"],
                    challenge["salt"],
                    difficulty,
                    expire_at,
                    challenge["signature"],
                    challenge["target_path"],
                    WASM_PATH
                )
            except Exception as e:
                app.logger.error(f"[get_pow_response] PoW 答案计算异常: {e}")
                answer = None
            if answer is None:
                app.logger.warning("[get_pow_response] PoW 答案计算失败，重试中...")
                resp.close()
                attempts += 1
                continue

            pow_dict = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": answer,  # 整数形式答案
                "signature": challenge["signature"],
                "target_path": challenge["target_path"]
            }
            pow_str = json.dumps(pow_dict, separators=(',', ':'), ensure_ascii=False)
            encoded = base64.b64encode(pow_str.encode("utf-8")).decode("utf-8").rstrip("=")
            resp.close()
            return encoded
        else:
            code = data.get("code")
            app.logger.warning(f"[get_pow_response] 获取 PoW 失败, code={code}, msg={data.get('msg')}")
            resp.close()
            if g.use_config_token:
                current_id = get_account_identifier(g.account)
                if not hasattr(g, 'tried_accounts'):
                    g.tried_accounts = []
                if current_id not in g.tried_accounts:
                    g.tried_accounts.append(current_id)
                new_account = choose_new_account(g.tried_accounts)
                if new_account is None:
                    break
                try:
                    login_deepseek_via_account(new_account)
                except Exception as e:
                    app.logger.error(f"[get_pow_response] 账号 {get_account_identifier(new_account)} 登录失败：{e}")
                    attempts += 1
                    continue
                g.account = new_account
                g.deepseek_token = new_account.get("token")
            else:
                attempts += 1
                continue
            attempts += 1
    return None

# ----------------------------------------------------------------------
# (8) 路由：/v1/models（模拟 OpenAI 模型列表）
# ----------------------------------------------------------------------
@app.route("/v1/models", methods=["GET"])
def list_models():
    app.logger.info("[list_models] 用户请求 /v1/models")
    models_list = [
        {
            "id": "DeepSeek-R1",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        },
        {
            "id": "deepseek-reasoner",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        },
        {
            "id": "DeepSeek-V3",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        },
        {
            "id": "deepseek-chat",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        }
    ]
    data = {"object": "list", "data": models_list}
    return jsonify(data), 200

# ----------------------------------------------------------------------
# (新增) 消息预处理函数，将多轮对话合并成最终 prompt
# ----------------------------------------------------------------------
def messages_prepare(messages: list) -> str:
    """处理消息列表，合并连续相同角色的消息，并添加角色标签：
    - 对于 assistant 消息，加上 <｜Assistant｜> 前缀及 <｜end▁of▁sentence｜> 结束标签；
    - 对于 user/system 消息（除第一条外）加上 <｜User｜> 前缀；
    - 如果消息 content 为数组，则提取其中 type 为 "text" 的部分；
    - 最后移除 markdown 图片格式的内容。
    """
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
    # 合并连续同一角色的消息
    merged = [processed[0]]
    for msg in processed[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["text"] += "\n\n" + msg["text"]
        else:
            merged.append(msg)
    # 添加标签
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
    # 移除 markdown 图片格式：
    final_prompt = re.sub(r"!", "", final_prompt)
    return final_prompt

# ----------------------------------------------------------------------
# (10) 路由：/v1/chat/completions
# ----------------------------------------------------------------------
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    mode_resp = determine_mode_and_token()
    if mode_resp:
        return mode_resp

    # 如果使用配置模式，检查账号是否正忙；如果忙则尝试切换账号
    if g.use_config_token:
        account_id = get_account_identifier(g.account)
        if account_id in active_accounts:
            g.tried_accounts.append(account_id)
            new_account = choose_new_account(g.tried_accounts)
            if new_account is None:
                return jsonify({"error": "All accounts are busy."}), 503
            try:
                login_deepseek_via_account(new_account)
            except Exception as e:
                return jsonify({"error": "Account login failed."}), 500
            g.account = new_account
            g.deepseek_token = new_account.get("token")
            account_id = get_account_identifier(new_account)
        active_accounts.add(account_id)
    try:
        req_data = request.json or {}
        app.logger.info(f"[chat_completions] 收到请求: {req_data}")
        model = req_data.get("model")
        messages = req_data.get("messages", [])
        if not model or not messages:
            return jsonify({"error": "Request must include 'model' and 'messages'."}), 400

        # 判断是否启用“思考”功能（这里根据模型名称判断）
        model_lower = model.lower()
        if model_lower in ["deepseek-v3", "deepseek-chat"]:
            thinking_enabled = False
        elif model_lower in ["deepseek-r1", "deepseek-reasoner"]:
            thinking_enabled = True
        else:
            return jsonify({"error": f"Model '{model}' is not available."}), 503

        # 使用 messages_prepare 函数构造最终 prompt
        final_prompt = messages_prepare(messages)
        app.logger.debug(f"[chat_completions] 最终 Prompt: {final_prompt}")

        session_id = create_session()
        if not session_id:
            return jsonify({"error": "invalid token."}), 401

        pow_resp = get_pow_response()
        if not pow_resp:
            return jsonify({"error": "Failed to get PoW (invalid token or unknown error)."}), 401
        app.logger.info(f"获取 PoW 成功: {pow_resp}")

        headers = {
            **get_auth_headers(),
            "x-ds-pow-response": pow_resp
        }
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": False
        }
        app.logger.debug(f"[chat_completions] -> {DEEPSEEK_COMPLETION_URL}, payload={payload}")

        deepseek_resp = call_completion_endpoint(payload, headers, stream=bool(req_data.get("stream", False)), max_attempts=3)
        if not deepseek_resp:
            return jsonify({"error": "Failed to get completion."}), 500

        created_time = int(time.time())
        completion_id = f"{session_id}"

        # 流式响应：SSE 格式返回事件流
        if bool(req_data.get("stream", False)):
            if deepseek_resp.status_code != 200:
                deepseek_resp.close()
                return Response(deepseek_resp.content,
                                status=deepseek_resp.status_code,
                                mimetype="application/json")

            def sse_stream():
                try:
                    final_text = ""
                    final_thinking = ""
                    first_chunk_sent = False
                    for raw_line in deepseek_resp.iter_lines(chunk_size=512):
                        try:
                            line = raw_line.decode("utf-8")
                        except Exception as e:
                            app.logger.warning(f"[sse_stream] 解码失败: {e}")
                            continue
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                prompt_tokens = len(tokenizer.encode(final_prompt))
                                completion_tokens = len(tokenizer.encode(final_text))
                                usage = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens
                                }
                                finish_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": [
                                        {"delta": {}, "index": 0, "finish_reason": "stop"}
                                    ],
                                    "usage": usage
                                }
                                yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                chunk = json.loads(data_str)
                                app.logger.debug(f"[sse_stream] 解析到 chunk: {chunk}")
                            except Exception as e:
                                app.logger.warning(f"[sse_stream] 无法解析: {data_str}, 错误: {e}")
                                continue
                            new_choices = []
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                ctype = delta.get("type")
                                ctext = delta.get("content", "")
                                if ctype == "thinking":
                                    if thinking_enabled:
                                        final_thinking += ctext
                                elif ctype == "text":
                                    final_text += ctext
                                delta_obj = {}
                                if not first_chunk_sent:
                                    delta_obj["role"] = "assistant"
                                    first_chunk_sent = True
                                if ctype == "thinking":
                                    if thinking_enabled:
                                        delta_obj["reasoning_content"] = ctext
                                elif ctype == "text":
                                    delta_obj["content"] = ctext
                                if delta_obj:
                                    new_choices.append({"delta": delta_obj, "index": choice.get("index", 0)})
                            if new_choices:
                                out_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": new_choices
                                }
                                yield f"data: {json.dumps(out_chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    app.logger.error(f"[sse_stream] 异常: {e}")
                finally:
                    deepseek_resp.close()
                    if g.use_config_token:
                        active_accounts.discard(get_account_identifier(g.account))
            return Response(stream_with_context(sse_stream()), content_type="text/event-stream")
        else:
            # 非流式响应处理
            think_list = []
            text_list = []
            try:
                for raw_line in deepseek_resp.iter_lines(chunk_size=512):
                    try:
                        line = raw_line.decode("utf-8")
                    except Exception as e:
                        app.logger.warning(f"[chat_completions] 解码失败: {e}")
                        continue
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            app.logger.debug(f"[chat_completions] 非流式 chunk: {chunk}")
                        except Exception as e:
                            app.logger.warning(f"[chat_completions] 无法解析: {data_str}, 错误: {e}")
                            continue
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta", {})
                            ctype = delta.get("type")
                            if ctype == "thinking" and thinking_enabled:
                                think_list.append(delta.get("content", ""))
                            elif ctype == "text":
                                text_list.append(delta.get("content", ""))
            finally:
                deepseek_resp.close()
            final_reasoning = "".join(think_list)
            final_content = "".join(text_list)
            prompt_tokens = len(tokenizer.encode(final_prompt))
            completion_tokens = len(tokenizer.encode(final_content))
            total_tokens = prompt_tokens + completion_tokens
            result = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content,
                            "reasoning_content": final_reasoning
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            return jsonify(result), 200
    finally:
        if g.use_config_token:
            active_accounts.discard(get_account_identifier(g.account))

# ----------------------------------------------------------------------
# (11) 路由：/
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("welcome.html")

# ----------------------------------------------------------------------
# 启动 Flask 应用（直接使用 Flask 内置服务器）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)