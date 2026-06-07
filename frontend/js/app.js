const API_BASE_URL = getApiBaseUrl();
const JOB_API_URL = `${API_BASE_URL}/transcribe-job`;
const CLEAR_OUTPUT_URL = `${API_BASE_URL}/output`;
const MODEL_STATUS_URL = `${API_BASE_URL}/model-status`;
const ALLOWED_LANGUAGES = new Set(['auto', 'mixed', 'ko', 'ja', 'vi', 'zh', 'en']);
const LANGUAGE_LABELS = {
  auto: 'Auto',
  mixed: 'Mixed',
  ko: 'Korean',
  ja: 'Japanese',
  vi: 'Vietnamese',
  zh: 'Chinese',
  en: 'English',
};
const LANGUAGE_CODE_PATTERN = '(auto|mixed|ko|ja|vi|zh|en)';
const ALLOWED_AUDIO_EXTENSIONS = new Set(['aac', 'flac', 'm4a', 'mp3', 'mp4', 'oga', 'ogg', 'opus', 'wav', 'webm']);
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

let audioFile = null;
let audioObjectUrl = null;
let srtContent = '';
let currentLanguage = 'auto';
let abortController = null;
let progressTimer = null;
let transcriptionStartedAt = 0;

const $ = id => document.getElementById(id);

function getApiBaseUrl() {
  if (window.SRT_API_BASE_URL) return window.SRT_API_BASE_URL.replace(/\/$/, '');
  if (window.SRT_API_URL) return window.SRT_API_URL.replace(/\/transcribe\/?$/, '').replace(/\/$/, '');

  const hostname = location.hostname;
  if (!hostname || hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://127.0.0.1:8001';
  }

  return `${location.protocol}//${hostname}:8001`;
}

if (location.protocol === 'file:') {
  $('fileWarn').style.display = 'block';
}

$('fileInput').addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) setFile(file);
});

$('transcribeBtn').addEventListener('click', () => {
  if (abortController) {
    cancelTranscription();
    return;
  }

  startTranscription();
});
$('downloadBtn').addEventListener('click', downloadSRT);
$('copyBtn').addEventListener('click', copySRT);
$('clearOutputBtn').addEventListener('click', clearOutputFolder);
$('checkModelBtn').addEventListener('click', checkModelStatus);
$('useMixedRanges').addEventListener('change', event => {
  updateRangeControls(event.target.checked);
});
$('customRangeLanguages').addEventListener('change', () => {
  updateRangeControls($('useMixedRanges').checked);
});
document.querySelectorAll('input[name="defaultExceptionLanguage"]').forEach(input => {
  input.addEventListener('change', () => {
    updateDefaultExceptionLanguageOptions();
    updateRangeHelp();
  });
});
updateRangeControls($('useMixedRanges').checked);

document.querySelectorAll('[data-language]').forEach(option => {
  option.addEventListener('click', () => selectLanguage(option.dataset.language));
});

document.querySelectorAll('input[name="language"]').forEach(input => {
  input.addEventListener('change', event => selectLanguage(event.target.value));
});

const dropZone = $('dropZone');

dropZone.addEventListener('dragover', event => {
  event.preventDefault();
  dropZone.classList.add('over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('over');
});

dropZone.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('over');

  const file = event.dataTransfer.files[0];
  if (file && isAllowedAudioFile(file)) {
    setFile(file);
  } else {
    setStatus('請選擇音訊檔案', 'error');
  }
});

function selectLanguage(language) {
  if (!ALLOWED_LANGUAGES.has(language)) return;

  currentLanguage = language;
  const radio = document.querySelector(`input[name="language"][value="${language}"]`);
  if (radio) radio.checked = true;

  document.querySelectorAll('[data-language]').forEach(option => {
    const isSelected = option.dataset.language === language;
    option.classList.toggle('selected', isSelected);
    option.setAttribute('aria-checked', String(isSelected));
  });

  setStatus(audioFile ? `已選擇語言：${language}` : '請先上傳音訊檔案', audioFile ? 'active' : '');
}

function setFile(file) {
  if (!isAllowedAudioFile(file)) {
    clearFileState();
    setStatus('不支援的檔案格式，請選擇 MP3、M4A、WAV、FLAC 等音訊檔案', 'error');
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    clearFileState();
    setStatus('檔案太大，請選擇 500 MB 以下的音訊檔案', 'error');
    return;
  }

  audioFile = file;

  const dropName = $('dropName');
  dropName.style.display = 'block';
  dropName.textContent = '✓ ' + file.name;

  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = URL.createObjectURL(file);

  const audioPreview = $('audioPreview');
  audioPreview.src = audioObjectUrl;
  audioPreview.style.display = 'block';

  $('transcribeBtn').disabled = false;
  $('outputSection').style.display = 'none';
  $('srtBox').textContent = '';
  srtContent = '';
  setProgress(0);
  setStatus('已載入：' + file.name, 'active');
}

function isAllowedAudioFile(file) {
  if (!file) return false;
  if (file.type && (file.type.startsWith('audio/') || file.type === 'video/mp4' || file.type === 'video/webm')) {
    return true;
  }

  const extension = file.name.split('.').pop()?.toLowerCase();
  return ALLOWED_AUDIO_EXTENSIONS.has(extension);
}

function clearFileState() {
  audioFile = null;
  srtContent = '';

  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
    audioObjectUrl = null;
  }

  $('fileInput').value = '';

  const dropName = $('dropName');
  dropName.style.display = 'none';
  dropName.textContent = '';

  const audioPreview = $('audioPreview');
  audioPreview.removeAttribute('src');
  audioPreview.style.display = 'none';

  $('transcribeBtn').disabled = true;
  $('outputSection').style.display = 'none';
  $('srtBox').textContent = '';
  setProgress(0);
}

function setStatus(message, type = 'active') {
  const status = $('status');
  status.textContent = message;
  status.className = 'status-bar' + (type ? ' ' + type : '');
}

function setProgress(value) {
  $('progTrack').classList.remove('indeterminate');
  $('progFill').style.width = Math.max(0, Math.min(value, 100)) + '%';
}

function setIndeterminateProgress() {
  $('progFill').style.width = '';
  $('progTrack').classList.add('indeterminate');
}

function stopProgressTimer() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function startProgressTimer() {
  transcriptionStartedAt = Date.now();
  stopProgressTimer();
  progressTimer = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - transcriptionStartedAt) / 1000);
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = String(elapsedSeconds % 60).padStart(2, '0');
    setStatus(`後端正在轉錄中... 已等待 ${minutes}:${seconds}。CPU large-v3 可能需要一段時間。`, 'active');
  }, 5000);
}

function setTranscribingState(isTranscribing) {
  const button = $('transcribeBtn');
  if (isTranscribing) {
    button.disabled = false;
    button.textContent = '■ 取消轉錄';
    button.classList.add('btn-cancel');
    return;
  }

  button.textContent = '▶ 開始轉錄';
  button.classList.remove('btn-cancel');
  button.disabled = !audioFile;
}

async function startTranscription() {
  if (!audioFile) {
    setStatus('請先上傳音訊檔案', 'error');
    return;
  }

  const saveOutput = $('saveOutput').checked;
  const professionalOptimization = $('professionalOptimization').checked;
  const rangeSettings = getRangeSettings();
  const rangeError = validateRangeSettings(rangeSettings);
  if (rangeError) {
    setStatus(rangeError, 'error');
    return;
  }

  abortController = new AbortController();
  setTranscribingState(true);
  $('outputSection').style.display = 'none';
  setProgress(10);
  setStatus('正在上傳音訊到 FastAPI 後端...', 'active');

  try {
    const srt = await transcribeWithFastAPI(
      audioFile,
      currentLanguage,
      saveOutput,
      rangeSettings.text,
      rangeSettings.custom,
      rangeSettings.defaultLanguage,
      professionalOptimization,
      abortController.signal,
    );

    srtContent = srt;
    $('srtBox').textContent = srtContent;
    $('outputSection').style.display = 'block';
    setProgress(100);
    setStatus('轉錄完成，可以下載或複製 SRT。', 'success');
  } catch (error) {
    if (error.name === 'AbortError') {
      setProgress(0);
      setStatus('已取消等待後端回覆。若後端已開始轉錄，伺服器會在目前工作結束或 timeout 後釋放資源。', 'warn');
    } else {
      setProgress(0);
      setStatus('轉錄失敗：' + error.message, 'error');
    }
  } finally {
    stopProgressTimer();
    abortController = null;
    setTranscribingState(false);
  }
}

function cancelTranscription() {
  if (abortController) {
    abortController.abort();
  }
}

function updateRangeControls(isEnabled) {
  $('mixedRangePanel').classList.toggle('open', isEnabled);
  $('customRangeLanguages').disabled = !isEnabled;
  if (!isEnabled) {
    $('customRangeLanguages').checked = false;
  }

  const isCustom = isEnabled && $('customRangeLanguages').checked;
  $('defaultExceptionLanguagePanel').hidden = !isCustom;
  if (!isCustom) {
    resetDefaultExceptionLanguage();
  }

  updateDefaultExceptionLanguageOptions();
  updateRangeHelp();
}

function updateRangeHelp() {
  const isCustom = $('useMixedRanges').checked && $('customRangeLanguages').checked;
  const defaultLanguage = getDefaultExceptionLanguage();

  if (!isCustom) {
    $('mixedRanges').placeholder = '例如：0-8\n8-01:12\n01:12-00:01:20\n00:00:00,000 --> 00:00:03,000';
    $('rangeHint').textContent = [
      '未勾自定義：每行只輸入時間段，這些區間會用 mixed 逐段偵測。',
      '格式例子：0-8 = 0秒到8秒；8-01:12 = 8秒到1分12秒；01:12-00:01:20 = 1分12秒到1分20秒；00:00:00,000 --> 00:00:03,000 = 標準 SRT 時間。',
    ].join('\n');
    return;
  }

  if (defaultLanguage) {
    $('mixedRanges').placeholder = '例如：0-3\n8-01:12\n01:12-00:01:20\n00:00:00,000 --> 00:00:03,000';
    $('rangeHint').textContent = [
      `已選預設例外語言：${LANGUAGE_LABELS[defaultLanguage]}。每行只輸入時間段，後端會自動套用 ${defaultLanguage}。`,
      '格式例子：0-3 = 0秒到3秒；8-01:12 = 8秒到1分12秒；01:12-00:01:20 = 1分12秒到1分20秒；00:00:00,000 --> 00:00:03,000 = 標準 SRT 時間。',
    ].join('\n');
    return;
  }

  $('mixedRanges').placeholder = '例如：0-3 ja\n8-01:12 mixed\n01:12-00:01:20 en\n00:00:00,000 --> 00:00:03,000 ja';
  $('rangeHint').textContent = [
    '自定義模式：每行必須是「時間段 語言碼」。語言碼支援 auto、mixed、ko、ja、vi、zh、en。',
    '格式例子：0-3 ja = 0秒到3秒用日文；8-01:12 mixed = 8秒到1分12秒逐段偵測；01:12-00:01:20 en = 1分12秒到1分20秒用英文。',
  ].join('\n');
}

function resetDefaultExceptionLanguage() {
  document.querySelectorAll('input[name="defaultExceptionLanguage"]').forEach(input => {
    input.checked = input.value === '';
  });
}

function updateDefaultExceptionLanguageOptions() {
  document.querySelectorAll('input[name="defaultExceptionLanguage"]').forEach(input => {
    input.closest('.exception-language-opt')?.classList.toggle('selected', input.checked);
  });
}

function getDefaultExceptionLanguage() {
  if (!$('useMixedRanges').checked || !$('customRangeLanguages').checked) {
    return '';
  }

  const selected = document.querySelector('input[name="defaultExceptionLanguage"]:checked');
  return selected?.value || '';
}

function getRangeSettings() {
  if (!$('useMixedRanges').checked) {
    return { enabled: false, text: '', custom: false, defaultLanguage: '' };
  }

  return {
    enabled: true,
    text: $('mixedRanges').value.trim(),
    custom: $('customRangeLanguages').checked,
    defaultLanguage: getDefaultExceptionLanguage(),
  };
}

function validateRangeSettings(rangeSettings) {
  if (!rangeSettings.enabled) {
    return '';
  }

  if (!rangeSettings.text) {
    return '請輸入例外時間段，或取消勾選「指定混合 / 例外語言時間段」。';
  }

  const lines = rangeSettings.text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  for (const line of lines) {
    let error = '';
    if (rangeSettings.custom && rangeSettings.defaultLanguage) {
      error = validateDefaultLanguageRangeLine(line);
    } else if (rangeSettings.custom) {
      error = validateCustomRangeLine(line);
    } else {
      error = validateSimpleRangeLine(line);
    }
    if (error) return error;
  }

  return '';
}

function validateSimpleRangeLine(line) {
  if (new RegExp(`\\s+${LANGUAGE_CODE_PATTERN}$`, 'i').test(line)) {
    return '如果要輸入語言碼，請先勾選「自定義每段語言」。格式：0-3 ja';
  }

  return validateTimeRange(line, '未勾自定義時，每行請只輸入時間段，例如：0-3');
}

function validateDefaultLanguageRangeLine(line) {
  if (new RegExp(`\\s+${LANGUAGE_CODE_PATTERN}$`, 'i').test(line)) {
    return '已選預設例外語言時，每行只需要輸入時間段，例如：0-3';
  }

  return validateTimeRange(line, '已選預設例外語言時，每行請只輸入時間段，例如：0-3');
}

function validateCustomRangeLine(line) {
  const match = line.match(new RegExp(`^(.+?)\\s+${LANGUAGE_CODE_PATTERN}$`, 'i'));
  if (!match) {
    return '自定義模式每行必須是「時間段 語言碼」，例如：0-3 ja 或 00:00:00,000 --> 00:00:03,000 ja';
  }

  const rangeText = match[1].trim();
  const language = match[2].toLowerCase();
  if (!ALLOWED_LANGUAGES.has(language.toLowerCase())) {
    return '語言碼只支援：auto、mixed、ko、ja、vi、zh、en';
  }

  return validateTimeRange(rangeText, '自定義模式的時間段格式錯誤，例如：0-3 ja 或 00:00:00,000 --> 00:00:03,000 ja');
}

function validateTimeRange(rangeText, errorMessage) {
  const parts = splitTimeRangeForValidation(rangeText);
  if (!parts) return errorMessage;

  const start = parseTimeValueForValidation(parts[0]);
  const end = parseTimeValueForValidation(parts[1]);
  if (start === null || end === null) return errorMessage;
  if (end <= start) return '時間段的結束時間必須大於開始時間。';

  return '';
}

function splitTimeRangeForValidation(rangeText) {
  const trimmed = rangeText.trim();
  if (!trimmed) return null;

  const arrowParts = trimmed.split(/\s*-->\s*/);
  if (arrowParts.length === 2 && arrowParts[0].trim() && arrowParts[1].trim()) {
    return [arrowParts[0].trim(), arrowParts[1].trim()];
  }
  if (arrowParts.length > 2) return null;

  let match = trimmed.match(/^(.+?)\s*(?:~|到|至)\s*(.+)$/);
  if (match) return [match[1].trim(), match[2].trim()];

  match = trimmed.match(/^(.+?)\s*-\s*(.+)$/);
  if (match) return [match[1].trim(), match[2].trim()];

  return null;
}

function parseTimeValueForValidation(value) {
  const normalized = value.replace(',', '.');
  if (!normalized) return null;

  if (!normalized.includes(':')) {
    const seconds = Number(normalized);
    return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
  }

  const rawParts = normalized.split(':');
  const parts = rawParts.map(part => Number(part));
  if (![2, 3, 4].includes(parts.length) || parts.some(part => !Number.isFinite(part) || part < 0)) {
    return null;
  }

  if (parts.length === 2) {
    if (!isIntegerTimePart(rawParts[0]) || !isSecondsTimePart(rawParts[1]) || parts[1] >= 60) {
      return null;
    }
    return (parts[0] * 60) + parts[1];
  }

  if (parts.length === 3) {
    if (!isIntegerTimePart(rawParts[0]) || !isIntegerTimePart(rawParts[1]) || !isSecondsTimePart(rawParts[2])) {
      return null;
    }
    if (parts[1] >= 60 || parts[2] >= 60) {
      return null;
    }
    return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
  }

  if (
    !isIntegerTimePart(rawParts[0])
    || !isIntegerTimePart(rawParts[1])
    || !isIntegerTimePart(rawParts[2])
    || !/^\d{1,3}$/.test(rawParts[3])
    || parts[1] >= 60
    || parts[2] >= 60
    || parts[3] > 999
  ) {
    return null;
  }

  return (parts[0] * 3600) + (parts[1] * 60) + parts[2] + (parts[3] / 1000);
}

function isIntegerTimePart(value) {
  return /^\d+$/.test(value);
}

function isSecondsTimePart(value) {
  return /^\d+(?:\.\d{1,3})?$/.test(value);
}

async function transcribeWithFastAPI(
  audioFile,
  language,
  saveOutput,
  mixedRanges,
  mixedRangesCustom,
  mixedRangesDefaultLanguage,
  professionalOptimization,
  signal,
) {
  const formData = new FormData();
  formData.append('file', audioFile, audioFile.name);
  formData.append('language', language);
  formData.append('save_output', saveOutput ? 'true' : 'false');
  formData.append('mixed_ranges', mixedRanges);
  formData.append('mixed_ranges_custom', mixedRangesCustom ? 'true' : 'false');
  formData.append('mixed_ranges_default_language', mixedRangesDefaultLanguage || '');
  formData.append('professional_optimization', professionalOptimization ? 'true' : 'false');

  setIndeterminateProgress();
  setStatus('正在上傳音訊並建立後端轉錄工作...', 'active');

  const jobResponse = await fetch(JOB_API_URL, {
    method: 'POST',
    body: formData,
    signal,
  });
  const jobText = await jobResponse.text();

  if (!jobResponse.ok) {
    throw new Error(extractErrorMessage(jobText, jobResponse.status));
  }

  const job = parseJsonResponse(jobText);
  if (!job.job_id) {
    throw new Error('後端沒有回傳 job_id');
  }

  setStatus('後端已收到工作，正在使用 faster-whisper large-v3 轉錄...', 'active');
  startProgressTimer();

  while (true) {
    await waitWithAbort(3000, signal);

    const statusResponse = await fetch(`${API_BASE_URL}/jobs/${job.job_id}`, { signal });
    const statusText = await statusResponse.text();

    if (!statusResponse.ok) {
      throw new Error(extractErrorMessage(statusText, statusResponse.status));
    }

    const status = parseJsonResponse(statusText);
    if (status.status === 'error') {
      throw new Error(status.error || '轉錄工作失敗');
    }

    if (status.status === 'done') {
      const srtResponse = await fetch(`${API_BASE_URL}/jobs/${job.job_id}/srt`, { signal });
      const srtText = await srtResponse.text();

      if (!srtResponse.ok) {
        throw new Error(extractErrorMessage(srtText, srtResponse.status));
      }

      return srtText;
    }
  }
}

function extractErrorMessage(text, status) {
  try {
    const parsed = JSON.parse(text);
    if (parsed.detail) return parsed.detail;
  } catch {
    // The backend normally returns JSON for HTTPException, but keep plain text readable.
  }

  return text || `HTTP ${status}`;
}

function parseJsonResponse(text) {
  try {
    return JSON.parse(text);
  } catch {
    throw new Error('後端回傳格式錯誤');
  }
}

async function checkModelStatus() {
  const button = $('checkModelBtn');
  const status = $('modelStatus');

  button.disabled = true;
  status.className = 'model-status-text active';
  status.textContent = '正在檢查 Hugging Face 遠端版本...';

  try {
    const response = await fetch(MODEL_STATUS_URL);
    const text = await response.text();

    if (!response.ok) {
      throw new Error(extractErrorMessage(text, response.status));
    }

    renderModelStatus(parseJsonResponse(text));
  } catch (error) {
    status.className = 'model-status-text error';
    status.textContent = '模型版本檢查失敗：' + error.message;
  } finally {
    button.disabled = false;
  }
}

function renderModelStatus(data) {
  const status = $('modelStatus');
  const localRevision = shortRevision(data.local_revision);
  const remoteRevision = shortRevision(data.remote_revision);

  if (data.status === 'latest') {
    status.className = 'model-status-text success';
    status.textContent = `模型已是最新：${data.model_name} / ${data.repo_id}\n本地 ${localRevision} = 遠端 ${remoteRevision}`;
    return;
  }

  if (data.status === 'outdated') {
    status.className = 'model-status-text warn';
    status.textContent = `模型不是最新：${data.model_name} / ${data.repo_id}\n本地 ${localRevision}，遠端 ${remoteRevision}\n要更新時請刪除 hf_cache volume 後重啟。`;
    return;
  }

  if (data.status === 'not_cached') {
    status.className = 'model-status-text warn';
    status.textContent = `${data.message}\n遠端版本：${remoteRevision}`;
    return;
  }

  status.className = 'model-status-text warn';
  status.textContent = `${data.message || '無法判定模型是否最新'}\n本地版本：${localRevision}`;
}

function shortRevision(revision) {
  if (!revision) return 'unknown';
  return String(revision).slice(0, 12);
}

function waitWithAbort(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortError());
      return;
    }

    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(createAbortError());
    }, { once: true });
  });
}

function createAbortError() {
  const error = new Error('AbortError');
  error.name = 'AbortError';
  return error;
}

function getCurrentSRT() {
  return $('srtBox').textContent || srtContent || '';
}

function downloadSRT() {
  const text = getCurrentSRT();
  if (!text.trim()) {
    setStatus('目前沒有可下載的 SRT 內容', 'error');
    return;
  }

  const baseName = audioFile
    ? audioFile.name.replace(/\.[^.]+$/, '')
    : 'subtitle';
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');

  link.href = url;
  link.download = `${baseName}.srt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function copySRT() {
  const text = getCurrentSRT();
  if (!text.trim()) {
    setStatus('目前沒有可複製的 SRT 內容', 'error');
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setStatus('已複製到剪貼簿', 'success');
  } catch {
    setStatus('複製失敗，請手動選取 SRT 內容', 'error');
  }
}

async function clearOutputFolder() {
  if (abortController) {
    setStatus('轉錄中不能清空 output/，請先取消或等目前工作完成。', 'warn');
    return;
  }

  const confirmed = window.confirm('確定要清空後端 output/ 裡的所有內容嗎？這個動作不能復原。');
  if (!confirmed) return;

  const button = $('clearOutputBtn');
  button.disabled = true;
  setStatus('正在清空 output/...', 'active');

  try {
    const response = await fetch(CLEAR_OUTPUT_URL, { method: 'DELETE' });
    const text = await response.text();

    if (!response.ok) {
      throw new Error(extractErrorMessage(text, response.status));
    }

    let deleted = 0;
    try {
      deleted = JSON.parse(text).deleted || 0;
    } catch {
      deleted = 0;
    }

    setStatus(`已清空 output/，刪除 ${deleted} 個項目。`, 'success');
  } catch (error) {
    setStatus('清空 output/ 失敗：' + error.message, 'error');
  } finally {
    button.disabled = false;
  }
}
