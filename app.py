import os
import json
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from threading import Thread
import time

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

# 模型加载函数
def load_model_task(model_path):
    global loaded_model, model_loading, model_info
    model_info['status'] = 'loading'
    model_info['error'] = None
    
    try:
        # 这里使用vllm加载模型
        from vllm import LLM
        loaded_model = LLM(model=model_path)
        model_info['status'] = 'loaded'
        model_info['name'] = os.path.basename(model_path)
    except Exception as e:
        model_info['status'] = 'error'
        model_info['error'] = str(e)
        print(f"Error loading model: {e}")
    
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
        return jsonify({'status': 'error', 'message': '模型正在加载中，请稍后再试'})
    
    if 'model_file' in request.files:
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择文件'})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        model_loading = True
        thread = Thread(target=load_model_task, args=(file_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': '模型开始加载'})
    
    elif 'model_path' in request.form:
        model_path = request.form['model_path']
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': f'模型路径不存在: {model_path}'})
        
        model_loading = True
        thread = Thread(target=load_model_task, args=(model_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'message': '模型开始加载'})
    
    return jsonify({'status': 'error', 'message': '请提供模型文件或路径'})

# 路由：获取模型状态
@app.route('/model_status')
def model_status():
    return jsonify(model_info)

# 路由：生成文本
@app.route('/generate', methods=['POST'])
def generate():
    if loaded_model is None:
        return jsonify({'status': 'error', 'message': '模型未加载'})
    
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    
    try:
        # 使用vllm生成文本
        outputs = loaded_model.generate([prompt], sampling_params={
            'temperature': temperature,
            'max_tokens': max_tokens
        })
        
        generated_text = outputs[0].outputs[0].text
        return jsonify({
            'status': 'success', 
            'generated_text': generated_text
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API路由：生成文本
@app.route('/api/v1/generate', methods=['POST'])
def api_generate():
    if loaded_model is None:
        return jsonify({'error': '模型未加载'}), 400
    
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    
    try:
        # 使用vllm生成文本
        outputs = loaded_model.generate([prompt], sampling_params={
            'temperature': temperature,
            'max_tokens': max_tokens
        })
        
        generated_text = outputs[0].outputs[0].text
        return jsonify({
            'generated_text': generated_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)