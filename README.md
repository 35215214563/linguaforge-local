# SRT Tool FastAPI

本專案把原本純前端 SRT 工具改成「靜態前端 + FastAPI 本地後端」。前端只負責上傳音訊、顯示 SRT、下載與複製；語音辨識由後端的 `faster-whisper` 處理，不使用 OpenAI API、不需要 API key，也不在瀏覽器端執行 Whisper / ONNX / WebGPU / WASM。

## 專案結構

```text
srt-tool/
├─ frontend/
│  ├─ srt.html
│  ├─ css/
│  │  └─ style.css
│  └─ js/
│     └─ app.js
├─ backend/
│  ├─ main.py
│  └─ transcriber.py
├─ audio/
├─ output/
├─ requirements.txt
└─ README.md
```

## 1. 建立 venv

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. 安裝 dependencies

```bash
pip install -r requirements.txt
```

## 3. 啟動後端

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

後端預設設定：

```text
model_name = "large-v3"
device = "cpu"
compute_type = "int8"
```

第一次執行會下載 faster-whisper 的 `large-v3` 模型。CPU 模式可能會比較慢。

後端目前會限制單次上傳最大約 500 MB，只接受常見音訊容器格式，把轉錄 timeout 設為 30 分鐘，並限制同時間只跑 1 個 large-v3 轉錄工作，避免 CPU 被多個請求同時打滿。

## 4. 啟動前端

另開一個 terminal：

```bash
cd frontend
python3 -m http.server 8000
```

## 5. 打開頁面

[http://localhost:8000/srt.html](http://localhost:8000/srt.html)

## 6. API Docs

[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

## Docker 啟動

也可以直接用 Docker Compose 跑前端和後端：

```bash
docker compose up --build
```

打開：

[http://localhost:8000/srt.html](http://localhost:8000/srt.html)

API docs：

[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

第一次啟動後端會下載 `large-v3` 模型，Docker volume `hf_cache` 會保留 Hugging Face 模型快取，之後通常不用重下載。輸出的 `.srt` 會寫到本機的 `output/`。Docker 服務以非 root 使用者執行。

停止服務：

```bash
docker compose down
```

## API

`GET /health`

回傳：

```json
{"status": "ok"}
```

`GET /model-status`

檢查目前本地 `large-v3` cache 是否和 Hugging Face 遠端 `main` revision 一致。這個 endpoint 只做版本比較，不會下載或更新模型。

回傳範例：

```json
{
  "status": "latest",
  "model_name": "large-v3",
  "repo_id": "Systran/faster-whisper-large-v3",
  "revision": "main",
  "local_revision": "...",
  "remote_revision": "...",
  "is_latest": true,
  "message": "本地模型已是遠端 main 最新版本。"
}
```

`POST /transcribe`

`multipart/form-data` 欄位：

- `file`: 音訊檔案
- `language`: `auto`, `mixed`, `ko`, `ja`, `vi`, `zh`, `en`，預設 `auto`
- `save_output`: `true` 或 `false`，預設 `true`
- `mixed_ranges`: 可選。指定少數混合語言區間，例如 `0-8` 或 `01:12-01:20`，每行、逗號或分號分隔。這些區間會用 mixed 逐段偵測，其餘時間使用 `language` 指定的目標語言。建議搭配 `ko`, `ja`, `vi`, `zh`, `en` 這類明確目標語言使用；如果整條音檔都需要多語偵測，才使用 `language=mixed`。

成功時回傳 `text/plain; charset=utf-8` 的 SRT 字幕文字。若 `save_output=true`，後端會把同名 `.srt` 寫入 `output/`。

`POST /transcribe-job`

手機和平板前端預設使用這個 endpoint。它會先上傳音訊並建立背景轉錄工作，立即回傳 JSON：

```json
{"job_id": "...", "status": "running"}
```

前端接著輪詢：

- `GET /jobs/{job_id}` 查狀態
- `GET /jobs/{job_id}/srt` 在完成後取得 `text/plain; charset=utf-8` 的 SRT

這個流程可以避免 iPhone / iPad 的瀏覽器因為長時間等待單一 fetch 連線而顯示 `Load failed`。

`mixed` 會先用 VAD 找出語音段落，再把每段音訊切出來獨立呼叫 faster-whisper，讓每段重新偵測語言與轉錄。這比較適合中英日韓越混合音檔，但會比單語音檔更慢。

`DELETE /output`

清空後端 `output/` 資料夾裡的所有內容，保留 `output/` 資料夾本身。前端的「清空 output/」按鈕會先跳出確認視窗。
