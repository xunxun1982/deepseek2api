# utils/deepseek.py
import json
import base64
import time
import logging
import random
import re
import struct
import ctypes
import threading
import asyncio
import queue

from fastapi import HTTPException, Request
from wasmtime import Store, Module, Linker

from utils.config import CONFIG, save_config
from utils.client import Client

logger = logging.getLogger("deepseek")

# 创建全局异步客户端实例
client = Client()

# DeepSeek 相关常量
DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_LOGIN_URL = f"https://{DEEPSEEK_HOST}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/completion"
BASE_HEADERS = {
    'Host': "chat.deepseek.com",
    'User-Agent': "DeepSeek/1.0.9 Android/34",
    'Accept': "application/json",
    'Accept-Encoding': "gzip",
    'Content-Type': "application/json",
    'x-client-platform': "android",
    'x-client-version': "1.0.9",
    'x-client-locale': "zh_CN",
    'x-rangers-id': "7883327620434123524",
    'accept-charset': "UTF-8",
}

WASM_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"

# 全局集合，记录当前正忙的账号
active_accounts = set()

def get_account_identifier(account: dict) -> str:
    return account.get("email", "").strip() or account.get("mobile", "").strip()

async def login_deepseek_via_account(account: dict) -> str:
    email = account.get("email", "").strip()
    mobile = account.get("mobile", "").strip()
    password = account.get("password", "").strip()
    if not password or (not email and not mobile):
        raise HTTPException(status_code=400, detail="账号缺少必要的登录信息（必须提供 email 或 mobile 以及 password）")
    if email:
        logger.info(f"[login_deepseek_via_account] 正在使用 email 登录账号：{email}")
        payload = {
            "email": email,
            "mobile": "",
            "password": password,
            "area_code": "",
            "device_id": "deepseek_to_api",
            "os": "android"
        }
    else:
        logger.info(f"[login_deepseek_via_account] 正在使用 mobile 登录账号：{mobile}")
        payload = {
            "mobile": mobile,
            "area_code": None,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android"
        }
    try:
        resp = await client.post(DEEPSEEK_LOGIN_URL, headers=BASE_HEADERS, json=payload)
    except Exception as e:
        logger.error(f"[login_deepseek_via_account] 登录请求异常: {e}")
        raise HTTPException(status_code=500, detail="Account login failed: 请求异常")
    if resp.status_code != 200:
        logger.error(f"[login_deepseek_via_account] HTTP 错误: {resp.status_code}")
        raise HTTPException(status_code=500, detail="Account login failed: HTTP error")
    try:
        data = await resp.json()
    except Exception as e:
        logger.error(f"[login_deepseek_via_account] JSON解析失败: {e}")
        raise HTTPException(status_code=500, detail="Account login failed: invalid JSON response")
    if (data.get("data") is None or 
        data["data"].get("biz_data") is None or 
        data["data"]["biz_data"].get("user") is None):
        logger.error(f"[login_deepseek_via_account] 登录响应格式错误: {data}")
        raise HTTPException(status_code=500, detail="Account login failed: invalid response format")
    new_token = data["data"]["biz_data"]["user"].get("token")
    if not new_token:
        logger.error(f"[login_deepseek_via_account] 登录响应中缺少 token: {data}")
        raise HTTPException(status_code=500, detail="Account login failed: missing token")
    account["token"] = new_token
    save_config(CONFIG)
    identifier = email if email else mobile
    logger.info(f"[login_deepseek_via_account] 成功登录账号 {identifier}，token: {new_token}")
    return new_token

def choose_new_account(exclude_ids: list) -> dict:
    accounts = CONFIG.get("accounts", [])
    available = [
        acc for acc in accounts
        if get_account_identifier(acc) not in exclude_ids and get_account_identifier(acc) not in active_accounts
    ]
    if available:
        chosen = random.choice(available)
        logger.info(f"[choose_new_account] 新选择账号: {get_account_identifier(chosen)}")
        return chosen
    logger.warning("[choose_new_account] 没有可用的账号")
    return None

async def determine_mode_and_token(request: Request) -> None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: missing Bearer token.")
    caller_key = auth_header.replace("Bearer ", "", 1).strip()
    config_keys = CONFIG.get("keys", [])
    if caller_key in config_keys:
        request.state.use_config_token = True
        request.state.tried_accounts = []
        selected_account = choose_new_account(request.state.tried_accounts)
        if not selected_account:
            raise HTTPException(status_code=500, detail="No accounts configured.")
        if not selected_account.get("token", "").strip():
            try:
                await login_deepseek_via_account(selected_account)
            except Exception as e:
                logger.error(f"[determine_mode_and_token] 账号 {get_account_identifier(selected_account)} 登录失败：{e}")
                raise HTTPException(status_code=500, detail="Account login failed.")
        else:
            logger.info(f"[determine_mode_and_token] 账号 {get_account_identifier(selected_account)} 已有 token，无需重新登录")
        request.state.deepseek_token = selected_account.get("token")
        request.state.account = selected_account
        logger.info(f"[determine_mode_and_token] 配置模式：使用账号 {get_account_identifier(selected_account)} 的 token")
    else:
        request.state.use_config_token = False
        request.state.deepseek_token = caller_key
        logger.info("[determine_mode_and_token] 使用用户自带 DeepSeek token")

def get_auth_headers(request: Request) -> dict:
    return { **BASE_HEADERS, "authorization": f"Bearer {request.state.deepseek_token}" }

async def call_completion_endpoint(payload: dict, headers: dict, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            deepseek_resp = await client.post(DEEPSEEK_COMPLETION_URL, headers=headers, json=payload)
        except Exception as e:
            logger.warning(f"[call_completion_endpoint] 请求异常: {e}")
            await asyncio.sleep(1)
            attempts += 1
            continue
        if deepseek_resp.status_code == 200:
            return deepseek_resp
        else:
            logger.warning(f"[call_completion_endpoint] 调用对话接口失败, 状态码: {deepseek_resp.status_code}")
            await asyncio.sleep(1)
            attempts += 1
    return None

async def create_session(request: Request, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        try:
            resp = await client.post(DEEPSEEK_CREATE_SESSION_URL, headers=headers, json={"agent": "chat"})
        except Exception as e:
            logger.error(f"[create_session] 请求异常: {e}")
            await asyncio.sleep(1)
            attempts += 1
            continue
        try:
            data = await resp.json()
        except Exception as e:
            logger.error(f"[create_session] JSON解析异常: {e}")
            data = {}
        if resp.status_code == 200 and data.get("code") == 0:
            session_id = data["data"]["biz_data"]["id"]
            logger.info(f"[create_session] 新会话 chat_session_id={session_id}")
            return session_id
        else:
            code = data.get("code")
            logger.warning(f"[create_session] 创建会话失败, code={code}, msg={data.get('msg')}")
            if getattr(request.state, "use_config_token", False):
                current_id = get_account_identifier(request.state.account)
                if not hasattr(request.state, 'tried_accounts'):
                    request.state.tried_accounts = []
                if current_id not in request.state.tried_accounts:
                    request.state.tried_accounts.append(current_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if new_account is None:
                    break
                try:
                    await login_deepseek_via_account(new_account)
                except Exception as e:
                    logger.error(f"[create_session] 账号 {get_account_identifier(new_account)} 登录失败：{e}")
                    attempts += 1
                    continue
                request.state.account = new_account
                request.state.deepseek_token = new_account.get("token")
            else:
                attempts += 1
                continue
        attempts += 1
    return None

def compute_pow_answer(algorithm: str,
                       challenge_str: str,
                       salt: str,
                       difficulty: int,
                       expire_at: int,
                       signature: str,
                       target_path: str,
                       wasm_path: str) -> int:
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"不支持的算法：{algorithm}")
    prefix = f"{salt}_{expire_at}_"
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

    retptr = add_to_stack(store, -16)
    ptr_challenge, len_challenge = encode_string(challenge_str)
    ptr_prefix, len_prefix = encode_string(prefix)
    wasm_solve(store, retptr, ptr_challenge, len_challenge, ptr_prefix, len_prefix, float(difficulty))
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
    add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)

async def get_pow_response(request: Request, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        try:
            resp = await client.post(DEEPSEEK_CREATE_POW_URL, headers=headers, json={"target_path": "/api/v0/chat/completion"}, timeout=30)
        except Exception as e:
            logger.error(f"[get_pow_response] 请求异常: {e}")
            await asyncio.sleep(1)
            attempts += 1
            continue
        try:
            data = await resp.json()
        except Exception as e:
            logger.error(f"[get_pow_response] JSON解析异常: {e}")
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
                logger.error(f"[get_pow_response] PoW 答案计算异常: {e}")
                answer = None
            if answer is None:
                logger.warning("[get_pow_response] PoW 答案计算失败，重试中...")
                await asyncio.sleep(1)
                attempts += 1
                continue
            pow_dict = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": answer,
                "signature": challenge["signature"],
                "target_path": challenge["target_path"]
            }
            pow_str = json.dumps(pow_dict, separators=(',', ':'), ensure_ascii=False)
            encoded = base64.b64encode(pow_str.encode("utf-8")).decode("utf-8").rstrip("=")
            return encoded
        else:
            code = data.get("code")
            logger.warning(f"[get_pow_response] 获取 PoW 失败, code={code}, msg={data.get('msg')}")
            if getattr(request.state, "use_config_token", False):
                current_id = get_account_identifier(request.state.account)
                if not hasattr(request.state, 'tried_accounts'):
                    request.state.tried_accounts = []
                if current_id not in request.state.tried_accounts:
                    request.state.tried_accounts.append(current_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if new_account is None:
                    break
                try:
                    await login_deepseek_via_account(new_account)
                except Exception as e:
                    logger.error(f"[get_pow_response] 账号 {get_account_identifier(new_account)} 登录失败：{e}")
                    attempts += 1
                    continue
                request.state.account = new_account
                request.state.deepseek_token = new_account.get("token")
            else:
                attempts += 1
                continue
            attempts += 1
    return None