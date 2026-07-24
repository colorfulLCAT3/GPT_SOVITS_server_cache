from flask import Flask, request, send_file, jsonify
import requests
import os
import hashlib
import logging
import json
import re
import random
import time
import threading

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 真实服务器的URL（请按需修改）
REAL_SERVER_URL = "http://192.168.30.2:9880"

# 加载 emoji 映射表
with open("emoji_all.json", "r", encoding="utf-8") as f:
    emoji_map = json.load(f)

# ===== 模型映射表（只写模型名，不带编号） =====
MODEL_MAP = {
    "snh": {
        "gpt": "GPT_weights_v2ProPlus/snhpromax-e15.ckpt",
        "sovits": "SoVITS_weights_v2ProPlus/snhpromax_e8_s88.pth",
    },
    "dingzhen": {
        "gpt": "GPT_weights_v2/丁真GPT.ckpt",
        "sovits": "SoVITS_weights_v2/丁真SoVITS.pth",
    },
}

# ===== seed 预设 =====
SEED_MAP = {
    0: -1,            # 随机（会在解析时生成一个随机整数）
    1: 3593077178,
    2: 1234567890,      # 预留，可自行扩展
}

# 全局状态（线程安全通过 switching_lock 控制）
current_gpt = None
current_sovits = None
current_model_name = None
switching_lock = threading.Lock()


def replace_emojis(text: str) -> str:
    """把文本中的 emoji 替换成中文读音"""
    new_text = []
    for ch in text:
        if ch in emoji_map:
            logging.debug(f"Replacing emoji {ch} -> {emoji_map[ch]}")
            new_text.append(emoji_map[ch])
        else:
            new_text.append(ch)
    return "".join(new_text)


def get_cache_path(model_name: str, text: str, seed) -> str:
    """根据模型名、文本和 seed 生成缓存文件路径（按模型分目录）"""
    if not model_name:
        model_name = "default"
    cache_dir = f"audio_cache_{model_name.capitalize()}"
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{text}_{seed}"
    hash_object = hashlib.md5(key.encode())
    cache_filename = f"{hash_object.hexdigest()}.wav"
    return os.path.join(cache_dir, cache_filename)


def parse_model_from_ref(ref_audio_path: str):
    """
    从 ref_audio_path 提取模型名和 seed
    例子: reference/alice_01.wav -> ("alice", 1 -> seed value)
    返回 (model_name, seed_value_or_None)
    """
    if not ref_audio_path:
        return None, None

    filename = os.path.basename(ref_audio_path)
    # 支持像 alice_01.wav 或 Alice_1.wav 等形式
    match = re.match(r"([a-zA-Z0-9]+)_(\d+)\.wav", filename)
    if not match:
        logging.warning(f"Filename {filename} not match pattern, skip model selection")
        return None, None

    model_name = match.group(1).lower()
    seed_idx = int(match.group(2))

    # 处理 seed
    if seed_idx in SEED_MAP:
        seed_val = SEED_MAP[seed_idx]
        if seed_val is None:  # 代表随机
            seed_val = random.randint(0, 2**32 - 1)
    else:
        # 未定义的索引也用随机种子
        seed_val = random.randint(0, 2**32 - 1)

    return model_name, seed_val


def resp_is_success(resp: requests.Response) -> bool:
    """
    更鲁棒地判断 set_* 接口是否成功：
    - status_code 必须为 200
    - 支持返回 plain "success"、"Success" 等文本
    - 或者 JSON like {"message": "success"} / {"result": "success"} 等
    """
    if resp.status_code != 200:
        return False

    # 先尝试解析 JSON
    try:
        j = resp.json()
        # 如果是字符串 "success"
        if isinstance(j, str) and j.strip().lower() == "success":
            return True
        # 如果是 dict，检查任意值里是否有 success
        if isinstance(j, dict):
            for v in j.values():
                if isinstance(v, str) and v.strip().lower() == "success":
                    return True
    except ValueError:
        # 不是 JSON，继续用文本匹配
        pass

    # 文本匹配回退
    text = resp.text or ""
    if "success" in text.strip().lower():
        return True

    return False


def switch_models_if_needed(model_name: str) -> bool:
    """
    根据模型名切换模型，必须先切换 GPT（成功后 sleep 10s），再切换 SoVITS。
    使用非阻塞锁：若另一个切换正在进行，则立即失败并返回 False（由 caller 处理）。
    """
    global current_gpt, current_sovits, current_model_name

    if not model_name:
        return True  # 没指定模型则不切换

    if model_name not in MODEL_MAP:
        logging.warning(f"No model mapping found for '{model_name}', skip switching")
        return True

    # 若已是目标模型，直接返回成功
    if current_model_name == model_name:
        logging.debug(f"Already using model '{model_name}', skip switching")
        return True

    # 尝试获得切换锁（非阻塞）
    got = switching_lock.acquire(blocking=False)
    if not got:
        logging.warning("Model switching already in progress, rejecting this request")
        return False

    try:
        model_info = MODEL_MAP[model_name]

        # 切换 GPT
        target_gpt = model_info.get("gpt")
        if target_gpt and current_gpt != target_gpt:
            try:
                logging.info(f"Requesting GPT switch to {target_gpt}")
                resp = requests.get(f"{REAL_SERVER_URL}/set_gpt_weights", params={"weights_path": target_gpt}, timeout=60)
            except Exception as e:
                logging.exception(f"Exception while requesting GPT switch: {e}")
                return False

            if resp_is_success(resp):
                logging.info(f"GPT switched to {target_gpt}, waiting 10s for init")
                current_gpt = target_gpt
                time.sleep(1)
            else:
                logging.error(f"Failed to switch GPT model: {resp.text}")
                return False

        # 切换 SoVITS
        target_sovits = model_info.get("sovits")
        if target_sovits and current_sovits != target_sovits:
            try:
                logging.info(f"Requesting SoVITS switch to {target_sovits}")
                resp = requests.get(f"{REAL_SERVER_URL}/set_sovits_weights", params={"weights_path": target_sovits}, timeout=60)
            except Exception as e:
                logging.exception(f"Exception while requesting SoVITS switch: {e}")
                return False

            if resp_is_success(resp):
                logging.info(f"SoVITS switched to {target_sovits}")
                current_sovits = target_sovits
            else:
                logging.error(f"Failed to switch SoVITS model: {resp.text}")
                return False

        # 所有切换成功，更新当前模型名
        current_model_name = model_name
        logging.info(f"Model switching completed. Current model: {current_model_name}")
        return True
    finally:
        switching_lock.release()


@app.route('/tts', methods=['GET'])
def tts():
    # 获取请求参数
    text = request.args.get('text')
    text_lang = request.args.get('text_lang')
    ref_audio_path = request.args.get('ref_audio_path')
    prompt_lang = request.args.get('prompt_lang')
    prompt_text = request.args.get('prompt_text')
    media_type = request.args.get('media_type', 'wav')
    streaming_mode = request.args.get('streaming_mode', 'false').lower() == 'true'
    parallel_infer = request.args.get('parallel_infer', 'true').lower() == 'true'
    return_fragment = request.args.get('return_fragment', 'true').lower() == 'true'
    text_split_method = request.args.get('text_split_method', 'cut5')
    batch_size = int(request.args.get('batch_size', 1))

    logging.debug(f"Received request: text={text}, ref_audio_path={ref_audio_path}")

    if not text:
        return jsonify({"error": "Missing required parameter: text"}), 400

    # 替换 emoji
    text = replace_emojis(text)

    # 解析 ref_audio_path 获取模型名和 seed
    model_name, seed = parse_model_from_ref(ref_audio_path)
    if model_name:
        success = switch_models_if_needed(model_name)
        if not success:
            return jsonify({"error": "Model switch failed or another switch in progress"}), 503
    else:
        logging.info("No model switch triggered")
        model_name = "default"

    # 生成缓存路径（按模型划分）
    cache_path = get_cache_path(model_name, text, seed)

    # 检查缓存
    if os.path.exists(cache_path):
        logging.info(f"Cache hit: {cache_path}")
        return send_file(cache_path, mimetype='audio/wav')

    # 构造请求参数（仅当 seed 非 None 时才传）
    params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "return_fragment": return_fragment,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
    }
    if seed is not None:
        params["seed"] = seed

    logging.debug(f"Forwarding request to real server with params: {params}")

    # 请求真实 TTS 服务器
    try:
        response = requests.get(f"{REAL_SERVER_URL}/tts", params=params, stream=True, timeout=300)
    except Exception as e:
        logging.exception(f"Exception while requesting TTS server: {e}")
        return jsonify({"error": "Failed to contact TTS server"}), 502

    if response.status_code != 200:
        logging.error(f"TTS request failed with status code {response.status_code}")
        return jsonify({"error": f"TTS request failed with status code {response.status_code}"}), response.status_code

    # 保存音频文件到缓存
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    logging.info(f"Cache miss for text: {text}. Saved new audio file to cache: {cache_path}")

    return send_file(cache_path, mimetype='audio/wav')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6880)
