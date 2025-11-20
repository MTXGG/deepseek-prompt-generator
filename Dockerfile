# 使用官方 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码 (假设您已重命名为 app.py)
# 如果您没有重命名，这里请改为 COPY random_prompt.py .
COPY app.py .

# Gunicorn 默认在 80 端口运行，Hugging Face Spaces 会将流量转发到这个端口
EXPOSE 80

# 定义环境变量（注意：这里只是一个占位符）
ENV PORT=80
ENV HOST=0.0.0.0
# Hugging Face Spaces 部署时，您需要在界面上设置 DEEPSEEK_API_KEY
# 确保在 HF Spaces 的 Settings -> Repository Secrets 里设置此密钥

# 启动 Gunicorn WSGI 服务器，运行 app.py 文件中的 server 实例
# 如果您的主文件是 random_prompt.py，则应为: gunicorn random_prompt:server
CMD exec gunicorn --bind $HOST:$PORT app:server
