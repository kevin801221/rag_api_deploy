
# 使用官方 Python 映像檔作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴項文件
COPY requirements.txt ./

# 安裝依賴項
# 我們添加 --no-cache-dir 來減小映像檔大小
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# 複製所有應用程式程式碼到工作目錄
COPY . .

# 開放 API 服務的端口
EXPOSE 8000

# 執行 FastAPI 應用程式的指令
# 使用 uvicorn 來運行，並監聽所有網絡接口
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

