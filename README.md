# LinguaForge Local

LinguaForge Local 是一個本地執行的多語言標準 `.srt` 字幕生成工具，可用於 YouTube、影片剪輯軟件、字幕編輯器和本地媒體播放器等 SRT 工作流。前端只負責上傳音訊、顯示 SRT、下載與複製；語音辨識由 FastAPI 後端的 `faster-whisper` 處理，不使用 OpenAI API、不需要 API key，也不在瀏覽器端執行 Whisper / ONNX / WebGPU / WASM。

## 專案結構

```text
linguaforge-local/
├─ frontend/
│  ├─ Dockerfile
│  ├─ srt.html
│  ├─ css/
│  │  └─ style.css
│  └─ js/
│     └─ app.js
├─ backend/
│  ├─ Dockerfile
│  ├─ entrypoint.sh
│  ├─ main.py
│  └─ transcriber.py
├─ audio/
│  └─ .gitkeep
├─ output/
│  └─ .gitkeep
├─ .dockerignore
├─ .gitignore
├─ docker-compose.yml
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

Docker Compose 已替後端設定 healthcheck，前端 container 會等 `/health` 可以回應後才啟動。第一次下載模型時，這個等待可能會比較久。

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
- `mixed_ranges`: 可選。指定少數混合 / 例外語言區間，其餘時間使用 `language` 指定的主要語言。
- `mixed_ranges_custom`: `true` 或 `false`，預設 `false`。
- `mixed_ranges_default_language`: 可選。只在 `mixed_ranges_custom=true` 時使用，支援 `auto`, `mixed`, `ko`, `ja`, `vi`, `zh`, `en`。
- `professional_optimization`: `true` 或 `false`，預設 `false`。啟用保守的 VAD 短段合併、前後 context padding、過長多句字幕拆分、短字幕合併、重疊時間修正與教材格式修正。

`mixed_ranges_custom=false` 時，每行只輸入時間段，該區間會用 `mixed` 逐段偵測：

```text
0-8
8-01:12
01:12-00:01:20
00:00:00,000 --> 00:00:03,000
```

`mixed_ranges_custom=true` 時，每行必須輸入 `時間段 語言碼`：

```text
0-3 ja
8-01:12 mixed
01:12-00:01:20 en
00:00:00,000 --> 00:00:03,000 ja
```

如果 `mixed_ranges_custom=true` 且 `mixed_ranges_default_language=ja`，每行只需要輸入時間段，後端會自動把這些例外區間當成日文處理：

```text
0-3
8-01:12
01:12-00:01:20
00:00:00,000 --> 00:00:03,000
```

時間格式支援 `ss-ss`、`ss-mm:ss`、`mm:ss-mm:ss`、`mm:ss-hh:mm:ss`、`hh:mm:ss-hh:mm:ss`、`hh:mm:ss,mmm-hh:mm:ss,mmm`、`hh:mm:ss:mmm-hh:mm:ss:mmm`，也支援標準 SRT 箭頭格式，例如 `00:00:00,000 --> 00:00:03,000`。使用冒號格式時，分鐘與秒數需小於 60；第 90 秒請寫成 `90`，不要寫成 `01:90`。

語言碼支援 `auto`, `mixed`, `ko`, `ja`, `vi`, `zh`, `en`。這適合音檔主體是韓文，但開頭幾秒是日文標題，或某些片段已知是英文 / 中文 / 越文的情況。

`professional_optimization=true` 時，未改變模型本身，但會啟用保守的字幕優化流程：

- mixed / VAD 模式會合併過短且間隔很近的語音段，並在轉錄時給前後約 0.4 秒 context，再把輸出時間裁回原語音區間。
- 主要語言區段也會用較保守的 VAD 分段，避免長靜音前後的兩句話被塞進同一條字幕。
- 如果模型仍把過長且含多個句子的字幕放在同一段，會按句號、問號、驚嘆號等句界保守拆分。
- SRT 後處理會合併太短且相鄰的字幕、修正重疊時間，並盡量避免一個短詞被拆成兩條字幕。
- 會套用少量教材格式修正，例如 `94%ページ` → `94ページ`、`第 10 課` → `第10課`。

未啟用時會保持原本轉錄流程與輸出。

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

清空後端 `output/` 資料夾裡的輸出內容，保留 `output/` 資料夾本身和 `.gitkeep`。前端的「清空 output/」按鈕會先跳出確認視窗。
