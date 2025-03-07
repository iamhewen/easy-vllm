# vLLM 本地服务端

这是一个基于vLLM构建的本地大语言模型服务端，提供简洁的Web界面用于模型管理和API调用。

## 功能特点

- 简洁的Web管理界面
- 支持通过本地路径加载模型
- 支持上传模型文件
- 实时显示模型加载状态
- 提供文本生成功能
- 提供RESTful API接口

## 安装要求

- Python 3.8+
- vLLM
- Flask
- PyTorch

## 安装步骤

1. 克隆或下载本项目

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行服务

```bash
python app.py
```

服务将在 http://localhost:5001 启动

## 使用方法

### Web界面

1. 打开浏览器访问 http://localhost:5001
2. 在「模型管理」部分，选择通过本地路径加载模型或上传模型文件
3. 模型加载完成后，在「文本生成」部分输入提示词并设置参数
4. 点击「生成文本」按钮获取结果

### API接口

#### 文本生成

```
POST /api/v1/generate
```

请求体:

```json
{
  "prompt": "你好，请介绍一下自己",
  "max_tokens": 100,
  "temperature": 0.7
}
```

响应:

```json
{
  "generated_text": "我是一个AI助手，由vLLM驱动..."
}
```

## 注意事项

- 模型加载可能需要较长时间，取决于模型大小和硬件配置
- 请确保有足够的GPU内存用于加载大型模型
- 上传模型文件大小限制为32MB，如需加载更大的模型，请使用本地路径方式

## 许可证

MIT