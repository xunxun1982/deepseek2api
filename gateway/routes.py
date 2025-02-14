# gateway/routes.py
import time
import json
import queue
import threading
import logging
import asyncio

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import transformers

from utils.deepseek import (
    determine_mode_and_token,
    get_auth_headers,
    create_session,
    call_completion_endpoint,
    get_pow_response,
    active_accounts,
    get_account_identifier,
    choose_new_account,
    login_deepseek_via_account
)
from utils.utils import messages_prepare

logger = logging.getLogger("routes")
router = APIRouter()

chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
KEEP_ALIVE_TIMEOUT = 5

templates = Jinja2Templates(directory="templates")

@router.get("/v1/models")
def list_models():
    logger.info("[list_models] 用户请求 /v1/models")
    models_list = [
        {
            "id": "deepseek-chat",
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
            "id": "deepseek-chat-search",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        },
        {
            "id": "deepseek-reasoner-search",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": []
        }
    ]
    data = {"object": "list", "data": models_list}
    return JSONResponse(content=data, status_code=200)

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        try:
            await determine_mode_and_token(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        except Exception as exc:
            logger.error(f"[chat_completions] determine_mode_and_token 异常: {exc}")
            return JSONResponse(status_code=500, content={"error": "Account login failed."})

        if getattr(request.state, "use_config_token", False):
            account_id = get_account_identifier(getattr(request.state, "account", {}))
            if account_id in active_accounts:
                request.state.tried_accounts.append(account_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if new_account is None:
                    raise HTTPException(status_code=503, detail="All accounts are busy.")
                try:
                    await login_deepseek_via_account(new_account)
                except Exception as e:
                    raise HTTPException(status_code=500, detail="Account login failed.")
                request.state.account = new_account
                request.state.deepseek_token = new_account.get("token")
                account_id = get_account_identifier(new_account)
            active_accounts.add(account_id)

        req_data = await request.json()
        logger.info(f"[chat_completions] 收到请求: {req_data}")
        model = req_data.get("model")
        messages = req_data.get("messages", [])
        if not model or not messages:
            raise HTTPException(status_code=400, detail="Request must include 'model' and 'messages'.")
        model_lower = model.lower()
        if model_lower in ["deepseek-v3", "deepseek-chat"]:
            thinking_enabled = False
            search_enabled = False
        elif model_lower in ["deepseek-r1", "deepseek-reasoner"]:
            thinking_enabled = True
            search_enabled = False
        elif model_lower in ["deepseek-v3-search", "deepseek-chat-search"]:
            thinking_enabled = False
            search_enabled = True
        elif model_lower in ["deepseek-r1-search", "deepseek-reasoner-search"]:
            thinking_enabled = True
            search_enabled = True
        else:
            raise HTTPException(status_code=503, detail=f"Model '{model}' is not available.")
        final_prompt = messages_prepare(messages)
        session_id = await create_session(request)
        if not session_id:
            raise HTTPException(status_code=401, detail="invalid token.")
        pow_resp = await get_pow_response(request)
        if not pow_resp:
            raise HTTPException(status_code=401, detail="Failed to get PoW (invalid token or unknown error).")
        logger.info(f"获取 PoW 成功: {pow_resp}")
        headers = { **get_auth_headers(request), "x-ds-pow-response": pow_resp }
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled
        }
        logger.info(f"[chat_completions] -> {payload}")
        deepseek_resp = await call_completion_endpoint(payload, headers, max_attempts=3)
        if not deepseek_resp:
            raise HTTPException(status_code=500, detail="Failed to get completion.")
        created_time = int(time.time())
        completion_id = f"{session_id}"
        if bool(req_data.get("stream", False)):
            if deepseek_resp.status_code != 200:
                return JSONResponse(content=await deepseek_resp.text(), status_code=deepseek_resp.status_code)
            def sse_stream():
                try:
                    final_text = ""
                    final_thinking = ""
                    first_chunk_sent = False
                    result_queue = queue.Queue()
                    last_send_time = time.time()
                    citation_map = {}
                    def process_data():
                        try:
                            for raw_line in deepseek_resp.iter_lines(chunk_size=512):
                                try:
                                    line = raw_line.decode("utf-8")
                                except Exception as e:
                                    logger.warning(f"[sse_stream] 解码失败: {e}")
                                    continue
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        result_queue.put(None)
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        if chunk.get("choices", [{}])[0].get("delta", {}).get("type") == "search_index":
                                            search_indexes = chunk["choices"][0]["delta"].get("search_indexes", [])
                                            for idx in search_indexes:
                                                citation_map[str(idx.get("cite_index"))] = idx.get("url", "")
                                            continue
                                        result_queue.put(chunk)
                                    except Exception as e:
                                        logger.warning(f"[sse_stream] 无法解析: {data_str}, 错误: {e}")
                        finally:
                            pass
                    process_thread = threading.Thread(target=process_data)
                    process_thread.start()
                    while True:
                        current_time = time.time()
                        if current_time - last_send_time >= KEEP_ALIVE_TIMEOUT:
                            logger.info("[sse_stream] 发送保活信号")
                            yield ": keep-alive\n\n"
                            last_send_time = current_time
                            continue
                        try:
                            chunk = result_queue.get(timeout=0.1)
                            if chunk is None:
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
                                last_send_time = current_time
                                break
                            new_choices = []
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                ctype = delta.get("type")
                                ctext = delta.get("content", "")
                                if search_enabled and ctext.startswith("[citation:"):
                                    ctext = ""
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
                                last_send_time = current_time
                        except queue.Empty:
                            continue
                except Exception as e:
                    logger.error(f"[sse_stream] 异常: {e}")
                finally:
                    if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
                        active_accounts.discard(get_account_identifier(request.state.account))
            return StreamingResponse(sse_stream(), media_type="text/event-stream")
        else:
            think_list = []
            text_list = []
            result = None
            citation_map = {}
            data_queue = queue.Queue()
            def collect_data():
                nonlocal result
                try:
                    for raw_line in deepseek_resp.iter_lines(chunk_size=512):
                        try:
                            line = raw_line.decode("utf-8")
                        except Exception as e:
                            logger.warning(f"[chat_completions] 解码失败: {e}")
                            continue
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                data_queue.put(None)
                                break
                            try:
                                chunk = json.loads(data_str)
                                if chunk.get("choices", [{}])[0].get("delta", {}).get("type") == "search_index":
                                    search_indexes = chunk["choices"][0]["delta"].get("search_indexes", [])
                                    for idx in search_indexes:
                                        citation_map[str(idx.get("cite_index"))] = idx.get("url", "")
                                    continue
                                for choice in chunk.get("choices", []):
                                    delta = choice.get("delta", {})
                                    ctype = delta.get("type")
                                    ctext = delta.get("content", "")
                                    if search_enabled and ctext.startswith("[citation:"):
                                        ctext = ""
                                    if ctype == "thinking" and thinking_enabled:
                                        think_list.append(ctext)
                                    elif ctype == "text":
                                        text_list.append(ctext)
                            except Exception as e:
                                logger.warning(f"[chat_completions] 无法解析: {data_str}, 错误: {e}")
                                continue
                finally:
                    final_reasoning = "".join(think_list)
                    final_content = "".join(text_list)
                    prompt_tokens = len(tokenizer.encode(final_prompt))
                    completion_tokens = len(tokenizer.encode(final_content))
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
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }
                    data_queue.put("DONE")
            collect_thread = threading.Thread(target=collect_data)
            collect_thread.start()
            def generate():
                last_send_time = time.time()
                while True:
                    current_time = time.time()
                    if current_time - last_send_time >= KEEP_ALIVE_TIMEOUT:
                        logger.info("[chat_completions] 发送保活信号(空行)")
                        yield '\n'
                        last_send_time = current_time
                    if not collect_thread.is_alive() and result is not None:
                        yield json.dumps(result)
                        break
                    asyncio.sleep(0.1)
            return StreamingResponse(generate(), media_type="application/json")
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception as exc:
        logger.error(f"[chat_completions] 未知异常: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    finally:
        if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
            active_accounts.discard(get_account_identifier(request.state.account))

@router.get("/")
def index(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})
