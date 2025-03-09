import os
import json
import torch
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from threading import Thread
import time
from datetime import datetime
from collections import deque
from vllm import LLM, SamplingParams

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vllm-webui-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 限制上传大小为32MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量
loaded_model = None
model_loading = False
model_info = {
    'name': None,
    'status': 'not_loaded',
    'error': None
}

# 日志存储，使用双端队列存储最近的日志
system_logs = deque(maxlen=100)

# 添加日志函数
def add_log(message, level="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    system_logs.append(log_entry)
    print(f"[{timestamp}] [{level.upper()}] {message}")

# 初始日志
add_log("系统启动")

# 模型加载函数
def load_model_task(model_path):
    global loaded_model, model_loading, model_info
    model_info['status'] = 'loading'
    model_info['error'] = None
    
    add_log(f"开始加载模型: {model_path}", "info")
    
    try:
        # 这里使用vllm加载模型
        loaded_model = LLM(model=model_path)
        # 验证模型是否正常加载
        test_prompt = "测试模型加载状态"
        test_params = SamplingParams(max_tokens=5)
        test_output = loaded_model.generate([test_prompt], sampling_params=test_params)
        add_log(f"模型加载验证成功，测试输出: {test_output[0].outputs[0].text[:20]}...", "info")
        
        model_info['status'] = 'loaded'
        model_info['name'] = os.path.basename(model_path)
        add_log(f"模型 {os.path.basename(model_path)} 加载成功", "success")
    except Exception as e:
        model_info['status'] = 'error'
        model_info['error'] = str(e)
        add_log(f"模型加载失败: {str(e)}", "error")
    
    model_loading = False

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)

# 路由：加载模型
@app.route('/load_model', methods=['POST'])
def load_model():
    global model_loading
    
    if model_loading:
        add_log("模型正在加载中，拒绝新的加载请求", "warning")
        return jsonify({'status': 'error', 'message': '模型正在加载中，请稍后再试'})
    
    if 'model_file' in request.files:
        file = request.files['model_file']
        if file.filename == '':
            add_log("用户尝试上传空文件", "warning")
            return jsonify({'status': 'error', 'message': '未选择文件'})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        add_log(f"模型文件已上传: {filename}", "info")
        
        model_loading = True
        thread = Thread(target=load_model_task, args=(file_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': '模型开始加载'})
    
    elif 'model_path' in request.form:
        model_path = request.form['model_path']
        if not os.path.exists(model_path):
            add_log(f"模型路径不存在: {model_path}", "error")
            return jsonify({'status': 'error', 'message': f'模型路径不存在: {model_path}'})
        
        model_loading = True
        thread = Thread(target=load_model_task, args=(model_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': '模型开始加载'})
    
    add_log("加载模型请求缺少必要参数", "error")
    return jsonify({'status': 'error', 'message': '请提供模型文件或路径'})

# 路由：获取模型状态
@app.route('/model_status')
def model_status():
    return jsonify(model_info)

# 路由：生成文本
@app.route('/generate', methods=['POST'])
def generate():
    if loaded_model is None:
        add_log("尝试在模型未加载时生成文本", "warning")
        return jsonify({'status': 'error', 'message': '模型未加载'})
    
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.0)
    presence_penalty = data.get('presence_penalty', 0.0)
    
    add_log(f"开始生成文本，最大长度: {max_tokens}, 温度: {temperature}", "info")
    
    try:
        # 使用vllm生成文本
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty
        )
        start_time = time.time()
        outputs = loaded_model.generate([prompt], sampling_params=sampling_params)
        generation_time = time.time() - start_time
        
        # 直接使用原始输出，不进行额外处理
        generated_text = outputs[0].outputs[0].text
        add_log(f"文本生成完成，耗时: {generation_time:.2f}秒", "success")
        
        return jsonify({
            'status': 'success',
            'generated_text': generated_text,
            'generation_time': f"{generation_time:.2f}秒"
        })
    except Exception as e:
        add_log(f"生成文本失败: {str(e)}", "error")
        return jsonify({'status': 'error', 'message': str(e)})

# API路由：生成文本
@app.route('/api/v1/generate', methods=['POST'])
def api_generate():
    if loaded_model is None:
        add_log("API请求：模型未加载", "warning")
        return jsonify({'error': '模型未加载'}), 400
    
    data = request.json
    if not data or 'prompt' not in data:
        add_log("API请求：缺少必要参数", "warning")
        return jsonify({'error': '缺少必要参数'}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.0)
    presence_penalty = data.get('presence_penalty', 0.0)
    
    add_log(f"API请求：开始生成文本，最大长度: {max_tokens}", "info")
    
    try:
        # 使用vllm生成文本
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty
        )
        start_time = time.time()
        outputs = loaded_model.generate([prompt], sampling_params=sampling_params)
        generation_time = time.time() - start_time
        
        # 直接使用原始输出，不进行额外处理
        generated_text = outputs[0].outputs[0].text
        add_log(f"API请求：文本生成完成，耗时: {generation_time:.2f}秒", "success")
        
        return jsonify({
            'generated_text': generated_text,
            'generation_time': f"{generation_time:.2f}秒"
        })
    except Exception as e:
        add_log(f"API请求：生成文本失败: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500

# 路由：获取日志
@app.route('/get_logs')
def get_logs():
    return jsonify(list(system_logs))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)