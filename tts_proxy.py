from flask import Flask, request, send_file, jsonify
import requests
import os
import hashlib
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 真实服务器的URL
REAL_SERVER_URL = "http://192.168.30.2:9880"
CACHE_DIR = "audio_cache_snh"

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(text: str) -> str:
    """根据文本生成缓存文件路径"""
    hash_object = hashlib.md5(text.encode())
    cache_filename = f"{hash_object.hexdigest()}.wav"
    return os.path.join(CACHE_DIR, cache_filename)

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

    logging.debug(f"Received request data: text={text}, text_lang={text_lang}, ref_audio_path={ref_audio_path}, prompt_lang={prompt_lang}, prompt_text={prompt_text}, media_type={media_type}, streaming_mode={streaming_mode}, parallel_infer={parallel_infer}, return_fragment={return_fragment}, text_split_method={text_split_method}, batch_size={batch_size}")

    if not text:
        logging.error("Error: 'text' is required")
        return jsonify({"error": "Missing required parameter: text"}), 400

    cache_path = get_cache_path(text)
    
    # 检查缓存是否存在
    if os.path.exists(cache_path):
        logging.info(f"Cache hit for text: {text}")
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
    }

    logging.debug(f"Forwarding request to real server with params: {params}")
    
    # 调用真实服务器
    response = requests.get(f"{REAL_SERVER_URL}/tts", params=params, stream=True)
    
    if response.status_code != 200:
        logging.error(f"TTS request failed with status code {response.status_code}")
        return jsonify({"error": f"TTS request failed with status code {response.status_code}"}), response.status_code
    
    # 保存音频文件到缓存
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    logging.info(f"Cache miss for text: {text}. Saved new audio file to cache.")
    
    # 返回新生成的音频文件
    return send_file(cache_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6880)
