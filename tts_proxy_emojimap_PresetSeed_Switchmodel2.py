from flask import Flask, request, send_file, jsonify
import requests
import os
import hashlib
import logging
import json
import re
import random

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 真实服务器的URL
REAL_SERVER_URL = "http://192.168.30.2:9880"
CACHE_DIR = "audio_cache_snh"

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 加载 emoji 映射表
with open("emoji_all.json", "r", encoding="utf-8") as f:
    emoji_map = json.load(f)

# ===== 模型映射表（只写模型名，不带编号） =====
MODEL_MAP = {
    "preset1": {
        "gpt": "xxx.ckpt",
        "sovits": "xxx.pth",
    },
    "preset2": {
        "gpt": "GPT_weights_v2/丁真GPT.ckpt",
        "sovits": "SoVITS_weights_v2/丁真SoVITS.pth",
    },
}

# ===== seed 预设 =====
SEED_MAP = {
    0: -1,            # 随机
    1: 3593077178,
    2: 1234567890,      # 预留，先找到不错的种子再加进去
}

# 记录当前已加载的模型，避免重复切换
current_gpt = None
current_sovits = None

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

def get_cache_path(text: str, seed: int) -> str:
    """根据文本和 seed 生成缓存文件路径"""
    key = f"{text}_{seed}"
    hash_object = hashlib.md5(key.encode())
    cache_filename = f"{hash_object.hexdigest()}.wav"
    return os.path.join(CACHE_DIR, cache_filename)

def parse_model_from_ref(ref_audio_path: str):
    """
    从 ref_audio_path 提取模型名和 seed
    例子: reference/alice_01.wav → ("alice", 1)
    """
    if not ref_audio_path:
        return None, None

    filename = os.path.basename(ref_audio_path)
    match = re.match(r"([a-zA-Z0-9]+)_(\d+)\.wav", filename)
    if not match:
        logging.warning(f"Filename {filename} not match pattern, skip model selection")
        return None, None

    model_name = match.group(1).lower()
    seed_idx = int(match.group(2))

    # 处理 seed
    if seed_idx in SEED_MAP:
        seed = SEED_MAP[seed_idx]
        if seed is None:  # 随机
            seed = random.randint(0, 2**32 - 1)
    else:
        seed = random.randint(0, 2**32 - 1)  # 未定义则随机

    return model_name, seed

def switch_models_if_needed(model_name: str):
    """根据模型名切换模型"""
    global current_gpt, current_sovits

    if not model_name or model_name not in MODEL_MAP:
        logging.warning(f"No model mapping found for {model_name}, skip switching")
        return

    model_info = MODEL_MAP[model_name]

    # 切换 GPT 模型
    if current_gpt != model_info["gpt"]:
        resp = requests.get(f"{REAL_SERVER_URL}/set_gpt_weights", params={"weights_path": model_info["gpt"]})
        if resp.status_code == 200:
            logging.info(f"Switched GPT model to {model_info['gpt']}")
            current_gpt = model_info["gpt"]
        else:
            logging.error(f"Failed to switch GPT model: {resp.text}")

    # 切换 SoVITS 模型
    if current_sovits != model_info["sovits"]:
        resp = requests.get(f"{REAL_SERVER_URL}/set_sovits_weights", params={"weights_path": model_info["sovits"]})
        if resp.status_code == 200:
            logging.info(f"Switched SoVITS model to {model_info['sovits']}")
            current_sovits = model_info["sovits"]
        else:
            logging.error(f"Failed to switch SoVITS model: {resp.text}")

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
        switch_models_if_needed(model_name)
    else:
        logging.info("No model switch triggered")

    # 生成缓存路径
    cache_path = get_cache_path(text, seed)

    # 检查缓存
    if os.path.exists(cache_path):
        logging.info(f"Cache hit: {cache_path}")
        return send_file(cache_path, mimetype='audio/wav')

    # 构造请求参数
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
        "seed": seed,
    }

    # 请求真实 TTS 服务器
    response = requests.get(f"{REAL_SERVER_URL}/tts", params=params, stream=True)
    if response.status_code != 200:
        return jsonify({"error": f"TTS request failed with {response.status_code}"}), response.status_code

    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    logging.info(f"Cache miss. Saved new audio file {cache_path}")
    return send_file(cache_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6880)
