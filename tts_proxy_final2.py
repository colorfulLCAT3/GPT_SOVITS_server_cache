#!/usr/bin/env python3
# tts_proxy.py
import os
import re
import time
import json
import random
import hashlib
import logging
import threading
from flask import Flask, request, jsonify, send_file
import requests

app = Flask(__name__)

# ---------- 配置 ----------
REAL_SERVER_URL = "http://192.168.30.2:9880"  # 真实 TTS / 模型切换服务地址，按需修改
PROXY_PORT = 6880
# -------------------------

logging.basicConfig(level=logging.DEBUG)

# 读取 emoji 映射（把 emoji 替换成中文读法）
try:
    with open("emoji_all.json", "r", encoding="utf-8") as f:
        emoji_map = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load emoji_all.json: {e}")
    emoji_map = {}

# 模型映射表（示例）
MODEL_MAP = {
    "snh": {
        "gpt": "GPT_weights_v2ProPlus/snhpromax-e15.ckpt",
        "sovits": "SoVITS_weights_v2ProPlus/snhpromax_e8_s88.pth",
    },
    "dingzhen": {
        "gpt": "GPT_weights_v2/丁真GPT.ckpt",
        "sovits": "SoVITS_weights_v2/丁真SoVITS.pth",
    },
    "kobe": {
        "gpt": "GPT_weights_v2/科比GPT.ckpt",
        "sovits": "SoVITS_weights_v2/科比SoVITS.pth",
    },
    "td": {
        "gpt": "GPT_weights_v2ProPlus/TD-e16.ckpt",
        "sovits": "SoVITS_weights_v2ProPlus/TD_e8_s80.pth",
    },
}

# ===== seed 预设 =====
SEED_MAP = {
    0: -1,            # 随机（会在解析时生成一个随机整数）
    1: 3593077178,
    2: 1234567890,      # 预留，找到好的自己加
}

# 全局状态（由 switching_lock 保护切换过程）
current_gpt = None
current_sovits = None
current_model_name = None
switching_lock = threading.Lock()


# ---------- 辅助函数 ----------

def replace_emojis(text: str) -> str:
    """把文本中的 emoji 替换成 emoji_map 中的中文读法（若没有映射则保留原字符）。"""
    if not text:
        return text
    # 使用每字符替换，保持原有逻辑
    return "".join(emoji_map.get(ch, ch) for ch in text)


def parse_model_from_ref(ref_audio_path: str):
    """
    从 ref_audio_path 提取 (model_name, seed_value_or_None, emotion_id)
    filename 格式应为: 模型名_种子序号_语气序号.wav (三段皆为字母/数字，种子序号和语气序号为数字)
    例如: alice_01_02.wav -> ("alice", SEED_MAP[1] or random, 2)
    如果解析失败返回 (None, None, None)
    """
    if not ref_audio_path:
        return None, None, None

    filename = os.path.basename(ref_audio_path)
    # 匹配 e.g. alice_01_02.wav 或 Alice_1_2.wav
    match = re.match(r"^([A-Za-z0-9]+)_(\d+)_(\d+)\.wav$", filename)
    if not match:
        logging.debug(f"ref filename does not match expected pattern: {filename}")
        return None, None, None

    model_name = match.group(1).lower()
    seed_idx = int(match.group(2))
    emotion_id = int(match.group(3))

    # 解析 seed（如果是 SEED_MAP 中的 None 则生成真实随机值）
    if seed_idx in SEED_MAP:
        seed_val = SEED_MAP[seed_idx]
        if seed_val is None:
            seed_val = random.randint(0, 2**32 - 1)
    else:
        # 未定义索引也生成随机
        seed_val = random.randint(0, 2**32 - 1)

    return model_name, seed_val, emotion_id


def resp_is_success(resp: requests.Response) -> bool:
    """
    鲁棒判断模型切换接口是否返回成功：
    - HTTP status code 必须为 200
    - 支持 JSON 返回如 {"message":"success"} 或 {"result":"success"}
    - 也支持纯文本 "success"
    """
    if resp is None:
        return False
    if resp.status_code != 200:
        return False

    # 先尝试解析 JSON
    try:
        j = resp.json()
        # 如果是字符串直接判断
        if isinstance(j, str) and j.strip().lower() == "success":
            return True
        # 如果是 dict，检查任意值里是否为 success（忽略大小写）
        if isinstance(j, dict):
            for v in j.values():
                if isinstance(v, str) and v.strip().lower() == "success":
                    return True
    except ValueError:
        # 非 JSON，退到文本判断
        pass

    # 文本回退匹配
    body = (resp.text or "").strip().lower()
    if "success" in body:
        return True

    return False


def get_cache_path(model_name: str, emotion_id: int, text: str) -> str:
    """
    根据模型名、语气 id 及文本生成缓存路径（注意：**不包含 seed**，不同种子共享同一缓存文件）
    缓存目录: audio_cache_{ModelNameCapitalized}_{emotion_id}
    缓存文件: md5(text).wav
    """
    if not model_name:
        model_name = "default"
    if emotion_id is None:
        emotion_id = 0
    # 目录名保留首字母大写，和你之前的约定一致
    cache_dir = f"audio_cache_{model_name.capitalize()}_{emotion_id}"
    os.makedirs(cache_dir, exist_ok=True)

    # 文本提前应已做 emoji 替换
    key = text or ""
    hash_hex = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{hash_hex}.wav")


def switch_models_if_needed(model_name: str) -> bool:
    """
    若 model_name 对应的模型与当前不同，则执行切换（非阻塞锁）。
    切换顺序：先切 GPT（成功后等待 1s），再切 SoVITS。两步都成功后返回 True。
    若已有切换在进行，直接返回 False（caller 返回 503）。
    若 model_name 在 MODEL_MAP 中未定义，则视为无需切换并返回 True。
    """
    global current_gpt, current_sovits, current_model_name

    if not model_name:
        return True

    if model_name not in MODEL_MAP:
        logging.warning(f"No model mapping found for '{model_name}', skipping model switch.")
        return True

    # 如果已经是目标模型则跳过
    if current_model_name == model_name:
        logging.debug(f"Already using model '{model_name}', skip switching")
        return True

    # 非阻塞锁，若无法获取立即返回 False
    got = switching_lock.acquire(blocking=False)
    if not got:
        logging.warning("Another model switching is in progress; rejecting this request.")
        return False

    try:
        model_info = MODEL_MAP[model_name]

        # 切换 GPT
        target_gpt = model_info.get("gpt")
        if target_gpt and current_gpt != target_gpt:
            try:
                logging.info(f"Requesting GPT switch to: {target_gpt}")
                resp = requests.get(
                    f"{REAL_SERVER_URL}/set_gpt_weights",
                    params={"weights_path": target_gpt},
                    timeout=60
                )
            except Exception as e:
                logging.exception(f"Exception while requesting GPT switch: {e}")
                return False

            if resp_is_success(resp):
                current_gpt = target_gpt
                logging.info(f"GPT switch ok -> {target_gpt}. Waiting 1s for init...")
                time.sleep(1)  # 你的要求：切换 GPT 后等待 1s
            else:
                logging.error(f"Failed to switch GPT model. Response: {getattr(resp, 'text', None)}")
                return False

        # 切换 SoVITS
        target_sovits = model_info.get("sovits")
        if target_sovits and current_sovits != target_sovits:
            try:
                logging.info(f"Requesting SoVITS switch to: {target_sovits}")
                resp = requests.get(
                    f"{REAL_SERVER_URL}/set_sovits_weights",
                    params={"weights_path": target_sovits},
                    timeout=60
                )
            except Exception as e:
                logging.exception(f"Exception while requesting SoVITS switch: {e}")
                return False

            if resp_is_success(resp):
                current_sovits = target_sovits
                logging.info(f"SoVITS switch ok -> {target_sovits}")
            else:
                logging.error(f"Failed to switch SoVITS model. Response: {getattr(resp, 'text', None)}")
                return False

        # 两个切换都成功，更新当前模型名
        current_model_name = model_name
        logging.info(f"Model switching completed. Current model: {current_model_name}")
        return True

    finally:
        # 释放锁（仅在成功 acquire 后会到这）
        switching_lock.release()


# ---------- 主路由 ----------

@app.route('/tts', methods=['GET'])
def tts():
    """
    代理 /tts 请求到真实服务器：
    - 解析 ref_audio_path 获取 model_name / seed / emotion_id
    - 替换文本中的 emoji
    - 根据 model_name 判断是否需要切换模型（按顺序：GPT -> 1s -> SoVITS）
    - 切换成功后发送真实 /tts 请求（仅在 seed 不为 None 时传 seed）
    - 缓存按 model+emotion+text（seed 不参与 md5）
    """
    # 收集常用参数（保留原有参数，便于兼容）
    text = request.args.get('text')
    text_lang = request.args.get('text_lang')
    ref_audio_path = request.args.get('ref_audio_path')
    prompt_lang = request.args.get('prompt_lang')
    prompt_text = request.args.get('prompt_text')
    media_type = request.args.get('media_type', 'wav')
    streaming_mode = request.args.get('streaming_mode', 'false').lower() == 'true'
    parallel_infer = request.args.get('parallel_infer', 'true').lower() == 'true'
    return_fragment = request.args.get('return_fragment', 'true').lower() == 'true'
    text_split_method = request.args.get('text_split_method', 'cut1')
    try:
        batch_size = int(request.args.get('batch_size', 1))
    except Exception:
        batch_size = 1

    logging.debug(f"Incoming /tts request: text={text!r}, ref_audio_path={ref_audio_path!r}")

    if not text:
        logging.error("Missing required parameter: text")
        return jsonify({"error": "Missing required parameter: text"}), 400

    # 先替换 emoji（确保缓存 key 与替换后文本一致）
    text_processed = replace_emojis(text)

    # 解析 ref_audio_path -> model_name, seed_val, emotion_id
    model_name, seed_val, emotion_id = parse_model_from_ref(ref_audio_path)
    if model_name:
        logging.debug(f"Parsed ref: model={model_name}, seed={seed_val}, emotion={emotion_id}")
        ok = switch_models_if_needed(model_name)
        if not ok:
            # 切换正在进行或失败，返回 503
            return jsonify({"error": "Model switch failed or another switch in progress"}), 503
    else:
        # 未指定或解析失败时，使用默认
        model_name = "default"
        emotion_id = 0
        logging.debug("No valid ref parsed -> using default model/emotion")

    # 生成缓存路径（**不包含 seed**）
    cache_path = get_cache_path(model_name, emotion_id, text_processed)
    logging.debug(f"Cache path: {cache_path}")

    # 命中缓存则直接返回
    if os.path.exists(cache_path):
        logging.info(f"Cache hit: {cache_path}")
        return send_file(cache_path, mimetype='audio/wav')

    # 构造转发给真实 TTS 的参数（只有 seed_val 非 None 时才传 seed）
    params = {
        "text": text_processed,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "return_fragment": return_fragment,
        "text_split_method": "cut1",
        "batch_size": batch_size,
    }
    if seed_val is not None:
        params["seed"] = seed_val

    logging.info(f"Forwarding to real TTS server with params (seed included only if set): { {k:v for k,v in params.items() if k!='prompt_text' and k!='text'} }")
    try:
        resp = requests.get(f"{REAL_SERVER_URL}/tts", params=params, stream=True, timeout=300)
    except Exception as e:
        logging.exception(f"Failed to contact real TTS server: {e}")
        return jsonify({"error": "Failed to contact TTS server"}), 502

    if resp.status_code != 200:
        logging.error(f"Real TTS server returned status {resp.status_code}: {resp.text}")
        return jsonify({"error": "TTS generation failed", "detail": resp.text}), resp.status_code

    # 将生成的音频写入缓存（同一文本在同一模型+语气目录下）
    try:
        with open(cache_path, "wb") as fw:
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    fw.write(chunk)
    except Exception as e:
        logging.exception(f"Failed to write cache file: {e}")
        return jsonify({"error": "Failed to save generated audio"}), 500

    logging.info(f"Saved generated audio to cache: {cache_path}")
    return send_file(cache_path, mimetype='audio/wav')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PROXY_PORT)
