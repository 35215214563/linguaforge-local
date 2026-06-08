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
│  ├─ ai_cleaner.py
│  ├─ ai_clients/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ ollama_client.py
│  ├─ main.py
│  ├─ srt_cleaner.py
│  ├─ srt_parser.py
│  ├─ transcriber.py
│  └─ subtitle_corrections/
│     ├─ common_zh.json
│     ├─ common_ja.json
│     ├─ common_ko.json
│     ├─ language_learning_terms.json
│     └─ protected_phrases.json
├─ tests/
│  ├─ test_ai_cleaner.py
│  ├─ test_ai_client_config.py
│  ├─ test_api_ai_clean_srt.py
│  ├─ test_api_clean_srt.py
│  ├─ test_backend_main_helpers.py
│  ├─ test_srt_parser.py
│  ├─ test_srt_cleaner.py
│  ├─ test_subtitle_wrapping.py
│  └─ test_transcriber_helpers.py
├─ audio/
│  └─ .gitkeep
├─ output/
│  └─ .gitkeep
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ .dockerignore
├─ .gitignore
├─ docker-compose.yml
├─ requirements-dev.txt
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

## 測試

一般本機驗證：

```bash
python -m unittest discover -s tests -v
python -m compileall backend
docker compose config
```

如需跑 interface tests 和 pytest：

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions 會執行 backend compile、unittest、pytest，以及 `docker compose config`。

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
- `professional_optimization`: `true` 或 `false`，預設 `false`。啟用 word timestamps 字級重切分、保守的 VAD 短段合併、前後 context padding、閱讀速度 / 時長控制、短字幕合併、重疊時間修正與教材格式修正。

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

`professional_optimization=true` 時，仍使用 faster-whisper `large-v3`，但會啟用較完整的字幕優化流程：

- 轉錄時會要求 `word_timestamps=True`，取得字級時間，再依句界、字數、閱讀速度與最大時長重新合成較完整的字幕段，避免大量 1-2 秒短字幕。
- mixed / VAD 模式會合併過短且間隔很近的語音段，並在轉錄時給前後約 0.4 秒 context，再把輸出時間裁回原語音區間。
- 主要語言區段也會用較保守的 VAD 分段，避免長靜音前後的兩句話被塞進同一條字幕。
- 如果模型仍把過長且含多個句子的字幕放在同一段，會按句號、問號、驚嘆號等句界保守拆分。
- SRT 後處理會合併太短且相鄰的字幕、修正重疊時間，並盡量避免一個短詞被拆成兩條字幕。
- 會套用少量高置信度教材與同音字修正，例如 `94%ページ` → `94ページ`、`第 10 課` → `第10課`。中文錯字表只會在主要語言明確選 `zh` 時套用，例如 `滾瓜爛薯` → `滾瓜爛熟`、`沒看過的生殖` → `沒看過的生詞`，避免 `auto` / `mixed` 誤傷日文、韓文或英文內容。

未啟用時會保持原本轉錄流程與輸出。啟用後通常字幕品質較好，但可能稍微增加轉錄時間。

成功時回傳 `text/plain; charset=utf-8` 的 SRT 字幕文字。若 `save_output=true`，後端會把同名 `.srt` 寫入 `output/`。

`POST /srt/clean`

把 Raw SRT 轉成 rule-based Clean SRT。這個 endpoint 不重新轉錄音訊、不接 OpenAI / Ollama / LLM，只做文字層清理，並保留原本時間軸。第一版預設只啟用 safe replacements 和 term replacements；contextual replacements 需要明確打開。這個 endpoint 也套用和轉錄 API 相同的簡單 IP rate limit。

輸入：

```json
{
  "srt_text": "1\n00:00:00,000 --> 00:00:02,000\nPatternDrill\n",
  "language": "zh",
  "script": "simplified",
  "enable_contextual_corrections": false,
  "custom_terms": [
    "CIA",
    "MI6",
    "FSI",
    "Pattern Drill",
    "ChatGPT",
    "Gemini",
    "procrastinate"
  ]
}
```

輸出：

```json
{
  "clean_srt": "1\n00:00:00,000 --> 00:00:02,000\nPattern Drill\n",
  "changes": [
    {
      "index": 1,
      "before": "PatternDrill",
      "after": "Pattern Drill",
      "type": "term_replacement"
    }
  ]
}
```

Clean SRT 會檢查 SRT 編號、時間格式、`start < end`、空字幕 block 與 block 數量。如果清理後驗證失敗，API 會回傳原始 Raw SRT 作為 `clean_srt`，並在 `changes` 裡標記 `validation_fallback`，避免輸出壞掉的 SRT。

目前外部詞表放在 `backend/subtitle_corrections/`。`language=zh` 會套用 `common_zh.json`；`language=ja` / `ko` 會讀取對應預留詞表；`auto` / `mixed` 預設只套用通用術語 spacing，不套中文錯字表。

- `safe_replacements`: 明顯 ASR 錯字，預設啟用。
- `term_replacements`: 英文術語 spacing 或固定寫法，預設啟用。
- `contextual_replacements`: 有誤傷風險的語意替換，預設關閉。
- `terms`: 會自動產生去空白版本的 term replacement，例如 `PatternDrill` → `Pattern Drill`。
- `custom_terms`: API 呼叫時額外傳入的專案詞表，也會套用同樣的 term spacing 規則。

`common_ja.json` 和 `common_ko.json` 目前是預留詞表位置，方便之後加入日文 / 韓文高置信度修正；第一版主要可用詞表集中在 `common_zh.json` 和 `language_learning_terms.json`。

`POST /srt/ai-clean`

把 Raw SRT 轉成 AI Clean Text SRT。流程是：

```text
Raw SRT
↓
Rule-based Clean
↓
AI Clean Text
↓
Strict Validation
↓
AI Clean Text SRT
```

AI 只是一個字幕文字校正引擎，不是字幕編輯器。後端只會把每個 subtitle block 的 `index` 和文字送給 AI，不送完整 SRT 時間軸，也要求 AI 回傳 JSON array，不允許回傳 SRT。驗證層會拒絕 invalid JSON、錯誤 block 數量、錯誤 index、timestamp、markdown code fence、疑似完整 SRT、過長或過短的文字。

輸入：

```json
{
  "srt_text": "1\n00:00:00,000 --> 00:00:02,000\n問答對練\n",
  "language": "zh",
  "script": "",
  "enable_contextual_corrections": false,
  "custom_terms": [],
  "ai_enabled": true
}
```

AI 使用成功時回傳：

```json
{
  "ai_clean_srt": "1\n00:00:00,000 --> 00:00:02,000\n問答對練。\n",
  "rule_based_srt": "1\n00:00:00,000 --> 00:00:02,000\n問答對練\n",
  "changes": [
    {
      "index": 1,
      "before": "問答對練",
      "after": "問答對練。",
      "type": "ai_text_correction"
    }
  ],
  "ai_used": true,
  "fallback_reason": null
}
```

如果 AI disabled、AI server 不可用、timeout、provider 回傳錯誤、AI output 無法通過完整驗證，API 不會因為 AI 失敗而回傳壞掉的 SRT；它會回退到 rule-based clean：

```json
{
  "ai_clean_srt": "<rule-based clean srt>",
  "rule_based_srt": "<rule-based clean srt>",
  "changes": [],
  "ai_used": false,
  "fallback_reason": "AI clean disabled by environment."
}
```

若 AI 回傳的 JSON 整體結構正確，但只有單一 block 的 `clean_text` 不合格，該 block 會回退到 rule-based text；其他合格 block 仍可使用 AI 修正。最終 SRT 會重新 parse 驗證，block count、index、block order、start time、end time 都必須和原始 SRT 一致。

AI Clean Text 的環境變數：

```text
AI_CLEAN_ENABLED=true
AI_CLEAN_PROVIDER=ollama
AI_CLEAN_BASE_URL=http://localhost:11434
AI_CLEAN_MODEL=qwen3:8b
AI_CLEAN_TIMEOUT_SECONDS=120
AI_CLEAN_TEMPERATURE=0
```

預設 provider 是 `ollama`，預設 model 是 `qwen3:8b`。model name 只是一個 config value；endpoint logic 不綁定任何特定模型。要切換模型只需改環境變數，例如：

```bash
AI_CLEAN_MODEL=qwen3:14b
AI_CLEAN_MODEL=qwen3:32b
AI_CLEAN_MODEL=another-local-model
```

AI client 透過 `backend/ai_clients/base.py` 抽象，provider-specific HTTP logic 放在 `backend/ai_clients/ollama_client.py`，`backend/main.py` 只呼叫 `AICleaner` service。AI server 是 optional；沒有 Ollama 或 model 時，API 會回退到 rule-based clean。測試使用 fake AI client，不需要真實 Ollama server、不需要 internet，也不新增重型模型 dependency。

重要限制：

- AI Clean Text 不會修復跨 block phrase splitting，例如 `深入探 / 討`、`空 / 間路径`、`自 / 動反擊`、`十分钟对 / 话`。
- AI Clean Text 不會 merge subtitle blocks。
- AI Clean Text 不會 split subtitle blocks。
- AI Clean Text 不會改 timestamps。
- timing optimization、subtitle block boundary repair、cross-block repair 屬於未來的 Optimize Subtitle Repair 任務，不在這個 endpoint 內。

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
