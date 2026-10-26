# SIP test client (pyVoIP) для голосового бота

Минимальный SIP-клиент на Python: регистрация на АТС, входящие и исходящие
звонки, запись разговора в WAV, эхо-бот, бот-проигрыватель WAV и голосовой
диалоговый бот `VAD → ASR → LLM → TTS`. Задуман как «испытательный стенд»:
SIP-транспорт отделён от обработки звука, поэтому сеть не нужно переделывать
при замене модели/промпта/голоса.

## Что умеет

| Команда | Назначение |
|---|---|
| `register` | только регистрация (проверка логина/пароля/сети) |
| `listen` | принимать входящие, отвечать, писать в WAV, запускать бота |
| `dial` | исходящий звонок, проигрывание WAV, запись ответа |
| `incoming` | многоканальный приём, запись, UI+ASR и AI-диалог с TTS-приветствием |

Боты (`--bot`): `auto`, `ai`, `silence`, `echo`, `tone`, `tone:<Гц>:<мс>`,
`wav:<путь>[:loop]`, `placeholder`, `placeholder:<мс>`. Бот получает входящий
кадр и возвращает кадр для отправки. `auto`/`ai` доступны в `incoming`:
`auto` выбирает диалогового бота, если настроены ASR/LLM/TTS. Полные таблицы
ключей — в разделе «Запуск и ключи».

Отдельные точки входа: `scripts/prompt_editor.py` (редактор промптов),
`scripts/live_incoming.py` (локальный harness входящего звонка),
`scripts/offline_smoke.py` (оффлайн-проверка) и `scripts/sip_trace.py`
(трассировка SIP/RTP).

## Веб-интерфейс с живой расшифровкой

У `incoming` есть локальный веб-интерфейс: реплики абонента распознаются через
OpenAI-совместимый ASR и появляются в браузере пузырями в реальном времени.
Ни Flask/FastAPI, ни websockets не нужны — только стандартная библиотека
(`ThreadingHTTPServer` + Server-Sent Events).

```powershell
# в .env должны быть OPENAI_BASE_URL и OPENAI_API_KEY
python main.py incoming --ui
# открыть http://127.0.0.1:8765/
```

Параметры:

| Флаг | Назначение |
|---|---|
| `--ui` | включить веб-интерфейс и ASR для входящих звонков |
| `--ui-host 127.0.0.1` | адрес прослушивания (по умолчанию только localhost) |
| `--ui-port 8765` | порт веб-интерфейса |
| `--no-asr` | не звать ASR; UI покажет только события звонка |

Как это работает: `MediaBridge` отдаёт 20-мс кадры в `DialogBot` (или в
`TranscribingBot` для обычного бота) → `EnergyVAD` находит конец реплики →
WAV уходит в `AiClient.transcribe()` в фоновом потоке → текст попадает в
историю диалога и `TranscriptStore` → `AiClient.chat()` формирует ответ →
TTS синтезируется и проигрывается по кадрам → SSE отправляет реплики в
браузер. Аудио звонка при этом не блокируется. В интерфейсе есть статус
линии, Call-ID, счётчик реплик, автоскролл и очистка ленты.

Для AI-диалога (по умолчанию `--bot auto`) при старте один раз синтезируются
и кэшируются приветствие `AI_GREETING` и no-input подсказка. Ответ
синтезируется по предложениям,
поэтому первый звук уходит в линию раньше. Если абонент молчит, бот через
`DIALOG_NO_INPUT_PROMPT_MS` мягко переспрашивает, а ещё через
`DIALOG_NO_INPUT_HANGUP_MS` завершает звонок. Если ASR/LLM/TTS не настроены,
`auto` откатывается на `placeholder` с предупреждением в логе.

Для проверки на локальном harness можно отправить клиенту настоящую речь:

```powershell
python scripts/live_incoming.py --bot ai --ui --caller-wav echo_test.wav
```

Интерфейс по умолчанию слушает только `127.0.0.1`. Если открыть его через
`--ui-host 0.0.0.0`, встроенной авторизации нет — не публикуйте порт наружу.

## Редактор промптов

Промпты из `prompts/` можно читать и править в отдельном локальном интерфейсе.
Это самостоятельный скрипт: SIP-клиент и ASR при этом не запускаются.

```powershell
python scripts/prompt_editor.py
# открыть http://127.0.0.1:8766/
```

Что есть в редакторе:

* список файлов с поиском, размером и датой изменения;
* подсветка Markdown-разметки в режиме «Чтение» (заголовки, списки, код,
  цитаты, ссылки) без внешних CDN;
* создание нового промпта (`.md`, `.txt`, `.prompt`);
* сохранение по `Ctrl+S` / `Cmd+S`, подтверждение перезаписи, если файл
  изменили на диске;
* защита от несохранённых изменений при переключении файла и закрытии вкладки;
* атомарная запись: при сбое не остаётся частично записанного промпта.

Параметры:

| Флаг | Назначение |
|---|---|
| `--dir DIR` | каталог с промптами (по умолчанию `prompts/` в корне проекта) |
| `--host HOST` | адрес прослушивания (по умолчанию `127.0.0.1`) |
| `--port PORT` | порт веб-интерфейса (по умолчанию `8766`; `0` — свободный) |
| `--no-browser` | не открывать браузер автоматически |
| `--verbose` | печатать запросы сервера |

Полный справочник по этому и остальным скриптам — в разделе «Запуск и ключи».

По умолчанию сервер доступен только с этой машины и не имеет авторизации.
Не открывайте его в интернет. Для правки промптов достаточно прав на запись
в каталог `prompts/`; содержимое сохраняется в UTF-8.

## Промпты для LLM

`PromptStore` умеет собирать несколько файлов в один system prompt. По
умолчанию берутся все файлы предметной области в фиксированном порядке:

```text
support_evp.md → connection_terms.md → billing_account_data.md → internet_private.md
```

Порядок можно переопределить в `SYSTEM_PROMPT_FILES` (запятая, точка с
запятой или перевод строки). Например:

```ini
SYSTEM_PROMPT_FILES=support_evp.md,connection_terms.md,billing_account_data.md,internet_private.md
```

Пути вида `prompts/support_evp.md` и `prompts\support_evp.md` приводятся к
имени файла; дубликаты удаляются с сохранением первого вхождения. Если файл
не найден, загрузка падает с `PromptNotFound`, а не отдаёт LLM неполный
промпт. Части разделяются строкой `\n\n---\n\n`.

Сборка в коде:

```python
from sipbot.prompts import PromptStore

store = PromptStore("prompts")
bundle = store.load_system_prompt_from_env()  # SYSTEM_PROMPT_FILES
system_prompt = bundle.content
```

`SYSTEM_PROMPT_FILE` поддерживается как устаревший одиночный вариант: он
используется, только если `SYSTEM_PROMPT_FILES` не задан или пуст.

## Ограничения pyVoIP 1.6.8 (важно!)

* только **G.711 (PCMU/PCMA), 8 кГц, моно** — без Opus/G.722;
* нет SRTP, нет эхоподавления (AEC), нет автоматической регулировки усиления;
* **нет исходящего DTMF** (в 1.6.8 нет `send_dtmf`), входящий DTMF — только
  через `call.get_dtmf()`;
* **не отвечает на OPTIONS** (keepalive АТС) и путает его с ответом на REGISTER —
  обойдено в `sipbot/pyvoip_fixes.py`, см. ниже;
* лицензия **GPLv3** — для внутреннего прототипа ок, для продукта смотрите
  pjsua2 / FreeSWITCH;
* на Windows медиа идёт по UDP через порты из `RTP_PORT_LOW..RTP_PORT_HIGH`.

Для продакшена с Opus/SRTP/эхоподавлением этот код — только референс:
транспорт придётся заменить на pjsua2 (см. «Дальше»).

## Патчи для живой АТС (`sipbot/pyvoip_fixes.py`)

При первом подключении к Asterisk 11.7 вылезли два связанных дефекта pyVoIP 1.6.8:

1. `SIPMessage` разбирает только `pyVoIP.SIPCompatibleMethods` =
   `INVITE/ACK/BYE/CANCEL`, поэтому OPTIONS (qualify-keepalive Asterisk) падает
   с `SIPParseError` и в recv-цикле молча выбрасывается. АТС не получает ответ
   и считает абонента недоступным.
2. Если OPTIONS приходит между нашим REGISTER и ответом на него, внутренний
   `__register` принимает его за ответ: `SIPParseError` → `registerFailures += 1`
   → таймер с `delay=0` → в сеть уходит **лишний REGISTER без авторизации**.
   После трёх таких ошибок статус становится `FAILED` (воспроизводилось примерно
   в каждом втором запуске).

`sipbot/pyvoip_fixes.py` вызывается из `SipPhone.__init__` и идемпотентен:

* добавляет keepalive-методы в `pyVoIP.SIPCompatibleMethods`, чтобы `SIPMessage`
  мог их разобрать;
* отвечает `200 OK` на `OPTIONS/NOTIFY/INFO/MESSAGE/SUBSCRIBE/REFER/UPDATE`;
* подменяет `SIPClient.recv` и `SIPClient.__register` на версии, которые
  обслуживают «чужие» сообщения и ждут именно ответ на REGISTER;
* не даёт упасть потоку приёма: `404/503` на неизвестный Call-ID (например,
  звонок на несуществующий номер) игнорируются, а любая другая ошибка
  обработчика логируется и не убивает recv-поток (в pyVoIP 1.6.8 там
  `KeyError` в `_callback_RESP_NotFound`).

После патча в трейсе ровно один цикл регистрации, `200 OK` на каждый OPTIONS,
`registerFailures = 0`, перерегистрация через 115 с (`Expires: 120` от АТС).
Тесты на это — `tests/test_pyvoip_fixes.py` (сырые пакеты взяты из реального
обмена с Asterisk).

## Формат аудио внутри pyVoIP

Это главный подводный камень: `read_audio`/`write_audio` работают **не** с
16-битным PCM, а с **8-битным unsigned PCM**:

* 1 сэмпл = 1 байт, 8000 Гц, моно;
* 160 байт = 20 мс;
* `0x80` (128) — тишина.

`VoIPCall.read_audio(160)` возвращает ровно 160 байт, `write_audio()` ждёт
такие же 160 байт. Конвертация в/из обычного 16-битного PCM (WAV, ASR, TTS)
живёт в `sipbot/audio_utils.py` (`u8_to_s16`, `s16_to_u8`, `read_wav`,
`resample_s16`).

## Установка

```powershell
cd E:\programs\sip_test_client
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# отредактируйте .env
```

Требуется Python 3.10–3.12. На Python 3.13+ модуль `audioop` удалён —
поставьте `pip install audioop-lts` (код выдаст понятную ошибку).

## Настройка `.env`

`main.py` читает `.env` (путь можно изменить ключом `--env FILE`);
`scripts/live_incoming.py` и `scripts/sip_trace.py` также используют `.env`.
Уже экспортированные переменные оболочки имеют приоритет над файлом. Если
`python-dotenv` не установлен, используется встроенный минимальный парсер.

```ini
# --- SIP/АТС ---
SIP_SERVER=192.168.1.10     # адрес АТС/Asterisk/FreeSWITCH
SIP_PORT=5060               # порт АТС (UDP)
SIP_USER=1000               # логин/внутренний номер
SIP_PASS=secret
SIP_MY_IP=192.168.1.50      # ОБЯЗАТЕЛЬНО: локальный IPv4 этой машины
SIP_LOCAL_PORT=5060         # локальный порт SIP (UDP)
RTP_PORT_LOW=10000
RTP_PORT_HIGH=10100
MAX_INCOMING_LINES=16       # значение по умолчанию для incoming --max-lines

# --- ASR/LLM/TTS (OpenAI-совместимый API) ---
OPENAI_BASE_URL=https://api.aitunnel.ru/v1
OPENAI_API_KEY=
ASR_MODEL=whisper-large-v3
ASR_LANGUAGE=ru
LLM_MODEL=deepseek-v4-flash
LLM_MAX_TOKENS=600
LLM_TEMPERATURE=0.3
AI_TIMEOUT=30

# --- TTS ---
TTS_MODEL=tts-1
TTS_VOICE=nova
# AiTunnel/gemini требуют pcm; если провайдер отдаёт WAV, поставьте wav.
TTS_RESPONSE_FORMAT=pcm
TTS_SAMPLE_RATE=24000
TTS_TIMEOUT=20
TTS_CACHE_DIR=cache/tts

# --- Голосовой диалог (incoming --bot auto/ai) ---
AI_GREETING=Здравствуйте! Вы позвонили в службу поддержки. Чем я могу вам помочь?
AI_FALLBACK_REPLY=Извините, не удалось получить ответ. Повторите, пожалуйста.
VAD_SPEECH_MS=200
VAD_SILENCE_MS=900
VAD_MAX_UTTERANCE_MS=10000
VAD_RMS_THRESHOLD=0.02
VAD_PREROLL_MS=300
DIALOG_HISTORY_TURNS=12
DIALOG_MAX_REPLY_CHARS=1200
DIALOG_MAX_REPLY_SENTENCES=2
DIALOG_MAX_TOOL_ROUNDS=3
DIALOG_NO_INPUT_PROMPT_MS=10000
DIALOG_NO_INPUT_HANGUP_MS=60000
DIALOG_NO_INPUT_PROMPT=Вы ещё здесь? Если вопрос остался, скажите, пожалуйста.
DIALOG_BARGE_IN=1

# Биллинг: POST {BILLING_BASE_URL}, Authorization: Bearer {BILLING_TOKEN}
BILLING_BASE_URL=
BILLING_TOKEN=
BILLING_TIMEOUT=10
```

Дополнительные переменные AI-клиента (все необязательны):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `AI_BASE_URL` | — | синоним `OPENAI_BASE_URL` |
| `AI_API_KEY` | — | синоним `OPENAI_API_KEY` |
| `OPENAI_ASR_MODEL` | `whisper-large-v3` | синоним `ASR_MODEL` |
| `LLM_MODEL` / `OPENAI_LLM_MODEL` | `deepseek-v4-flash` | модель LLM |
| `LLM_MAX_TOKENS` | `600` | лимит completion-токенов; часть уходит на reasoning модели |
| `LLM_TEMPERATURE` | `0.3` | вариативность ответа LLM (0.0–2.0) |
| `TTS_MODEL` / `OPENAI_TTS_MODEL` | `gemini-3.8-flash-lite-tts` | модель TTS |
| `TTS_VOICE` / `OPENAI_TTS_VOICE` | `Zephyr` | голос TTS |
| `TTS_RESPONSE_FORMAT` | `pcm` | формат ответа TTS: `pcm` (raw s16le) или `wav` |
| `TTS_SAMPLE_RATE` | `24000` | частота raw PCM от TTS до ресемплинга в 8 кГц |
| `TTS_TIMEOUT` | `20` | таймаут одного запроса TTS, с |
| `TTS_CACHE_DIR` | `cache/tts` | дисковый кэш WAV; пустое значение отключает кэш |
| `AI_GREETING` | встроенная фраза | TTS-приветствие в начале звонка; пусто — без приветствия |
| `AI_FALLBACK_REPLY` | «Извините…» | фраза при ошибке LLM |
| `VAD_SPEECH_MS` | `200` | сколько речи подтверждает начало реплики |
| `VAD_SILENCE_MS` | `900` | сколько тишины завершает реплику |
| `VAD_MAX_UTTERANCE_MS` | `10000` | максимальная длительность речи в реплике (тишина не учитывается) |
| `VAD_RMS_THRESHOLD` | `0.02` | порог энергии речи (0.0–1.0) |
| `VAD_PREROLL_MS` | `300` | сколько аудио до начала речи сохранить в WAV для ASR |
| `DIALOG_HISTORY_TURNS` | `12` | сколько пар «вопрос-ответ» держать в контексте LLM |
| `DIALOG_MAX_REPLY_CHARS` | `1200` | ограничение длины ответа для TTS |
| `DIALOG_MAX_REPLY_SENTENCES` | `2` | сколько первых предложений ответа оставлять для телефонии |
| `DIALOG_MAX_TOOL_ROUNDS` | `3` | максимум раундов tool-calls LLM за одну реплику |
| `DIALOG_NO_INPUT_PROMPT_MS` | `10000` | через сколько тишины подсказать «Вы ещё здесь?» (`0` — выключить) |
| `DIALOG_NO_INPUT_HANGUP_MS` | `60000` | через сколько тишины после подсказки завершить звонок (`0` — не завершать) |
| `DIALOG_NO_INPUT_PROMPT` | текст | фраза мягкой подсказки при молчании |
| `DIALOG_BARGE_IN` | `1` | останавливать ответ, когда абонент начинает говорить |
| `BILLING_BASE_URL` | — | POST-URL биллинга для `get_billing_info` |
| `BILLING_TOKEN` | — | Bearer-токен биллинга |
| `BILLING_TIMEOUT` | `10` | таймаут запроса к биллингу, с |
| `OPENAI_ASR_LANGUAGE` | `ru` | синоним `ASR_LANGUAGE` |
| `SYSTEM_PROMPT_FILES` | все 4 файла из `prompts/` | список файлов system prompt |
| `SYSTEM_PROMPT_FILE` | — | устаревший одиночный файл (fallback) |
| `SSL_CERT_FILE` | системный | путь к CA-файлу для HTTPS к AI-API |

TTS-кэш хранит WAV целиком, поэтому приветствие и no-input подсказка
синтезируются один раз, а повторные фразы (например, «Извините…») не тратят
запрос. Ключ кэша включает
модель, голос и текст. `TTS_RESPONSE_FORMAT=pcm` — обязательный вариант для
`gemini-*-tts` в AiTunnel; для OpenAI `tts-1` он тоже поддерживается. Если
провайдер вернёт WAV, `TtsService` определит это по заголовку `RIFF`.

Barge-in работает по энергии входящего сигнала. В этом клиенте нет AEC, поэтому
эхо собственного TTS в линии может ложно останавливать ответ; для телефонии с
эхом используйте `DIALOG_BARGE_IN=0` или внешний эхоподавитель.

Если `OPENAI_BASE_URL`/`OPENAI_API_KEY` (или их `AI_*` синонимы) не заданы,
ASR/LLM/TTS отключаются; `incoming --ui` работает, но без расшифровки.

`SIP_MY_IP` проверяется при старте: `0.0.0.0`, `127.0.0.1` и не-IP значения
отклоняются. Причина — pyVoIP подставляет `myIP` в SDP (`c=IN IP4 ...`);
с `0.0.0.0` АТС не знает, куда слать RTP, и получается one-way audio.
`MAX_INCOMING_LINES` не должен превышать число портов в
`RTP_PORT_LOW..RTP_PORT_HIGH`.

Брандмауэр Windows: разрешите входящий UDP на `SIP_LOCAL_PORT` и на диапазон
`RTP_PORT_LOW..RTP_PORT_HIGH` для `python.exe`.

## Запуск и ключи

Все команды выполняются из корня проекта. Основной CLI — `main.py`; отдельные
инструменты в `scripts/` запускаются самостоятельно и SIP-клиент не подменяют.

| Точка входа | Назначение |
|---|---|
| `python main.py ...` | регистрация, входящие/исходящие звонки, многоканальный приём |
| `python scripts/prompt_editor.py` | локальный веб-редактор промптов |
| `python scripts/live_incoming.py` | локальный harness: ненастоящая АТС + один входящий звонок |
| `python scripts/offline_smoke.py` | оффлайн-проверка моста и записи без сети |
| `python scripts/sip_trace.py` | трассировка SIP/RTP для диагностики |

### `main.py`: общие ключи

```powershell
python main.py [--env FILE] [-v|--verbose] <команда> [ключи]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--env FILE` | `.env` | путь к файлу с настройками |
| `-v`, `--verbose` | выкл. | отладочные логи (DEBUG) |

Общие ключи указываются **до** команды, например:
`python main.py --env .env.test -v register`.

У каждой подкоманды `main.py` и у каждого скрипта есть `-h`/`--help` с
актуальным списком ключей: `python main.py incoming --help`,
`python scripts/prompt_editor.py --help` и т. п.

Если одноимённая переменная уже экспортирована в оболочке, она имеет приоритет
над `.env` (`load_dotenv` не перезаписывает окружение). Проверить мешающую
переменную можно через `printenv SIP_MY_IP`, убрать — `unset SIP_MY_IP`.

### `register` — только регистрация

```powershell
python main.py register [--timeout SEC] [--hold SEC]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--timeout SEC` | `20` | сколько ждать статуса `REGISTERED` |
| `--hold SEC` | `0` | сколько держать регистрацию после успеха |

```powershell
python main.py register
python main.py register --hold 5
python main.py register --timeout 30 --hold 10
```

### `listen` — одноканальный приём входящих

```powershell
python main.py listen [--bot SPEC] [--record FILE] [--timeout SEC] [--duration SEC]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--bot SPEC` | `silence` | что отвечать абоненту (см. «Боты») |
| `--record FILE` | нет | писать звук абонента в WAV |
| `--timeout SEC` | `0` | общее время ожидания звонков; `0` — бесконечно |
| `--duration SEC` | `0` | максимальная длительность звонка; `0` — до отбоя |

`listen` обрабатывает звонки последовательно и после каждого снова ждёт
следующий. Остановка — `Ctrl+C`, либо по `--timeout`.

```powershell
# эхо + запись, максимум 60 секунд на звонок
python main.py listen --bot echo --record incoming.wav --duration 60

# приветствие в цикле, ждать звонка не дольше 10 минут
python main.py listen --bot wav:greeting.wav:loop --record incoming.wav --timeout 600

# молчать и ждать до отбоя
python main.py listen --bot silence --timeout 0 --duration 0
```

### `dial` — исходящий звонок

```powershell
python main.py dial TARGET [--play FILE] [--bot SPEC] [--record FILE] [--duration SEC] [--timeout SEC]
```

| Аргумент/ключ | По умолчанию | Назначение |
|---|---|---|
| `TARGET` | обязателен | номер/логин; `sip:1001@host` тоже принимается, хост игнорируется |
| `--play FILE` | нет | проиграть WAV абоненту после ответа |
| `--bot SPEC` | нет | бот для входящего звука, если не задан `--play` |
| `--record FILE` | нет | писать ответ абонента в WAV |
| `--duration SEC` | `20` | длительность разговора после ответа; `0` — до отбоя |
| `--timeout SEC` | `20` | ожидание ответа на INVITE |

`--play` имеет приоритет над `--bot`: при одновременном указании будет
проигрываться WAV. Если не задано ни то, ни другое, клиент только слушает.

```powershell
python main.py dial 1001 --play greeting.wav --record answer.wav --duration 20
python main.py dial 1001 --bot echo --duration 15
python main.py dial 1001 --bot tone --duration 10
python main.py dial 1001 --bot tone:800:500 --record answer.wav --duration 0
```

### `incoming` — многоканальный приём входящих

```powershell
python main.py incoming [--bot SPEC] [--record-dir DIR] [--max-lines N] [--ui]
                        [--ui-host HOST] [--ui-port PORT] [--no-asr]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--bot SPEC` | `auto` | обработчик каждого звонка (см. «Боты»); `auto` = `ai`, если AI настроен, иначе `placeholder` |
| `--record-dir DIR` | `recordings` | каталог для WAV каждого входящего звонка |
| `--max-lines N` | `MAX_INCOMING_LINES` (`16`) | сколько звонков обслуживать одновременно |
| `--ui` | выкл. | включить веб-интерфейс и живую расшифровку (ASR) |
| `--ui-host HOST` | `127.0.0.1` | адрес веб-интерфейса |
| `--ui-port PORT` | `8765` | порт веб-интерфейса |
| `--no-asr` | выкл. | отключить ASR и AI-бота; UI покажет только события звонка |

`incoming` работает до `Ctrl+C` и не завершается после первого звонка. Каждый
звонок пишется отдельным WAV в `--record-dir`. `--max-lines` не должен
превышать число портов в `RTP_PORT_LOW..RTP_PORT_HIGH`. Для AI-диалога нужны
`OPENAI_BASE_URL`/`OPENAI_API_KEY`, файлы из `SYSTEM_PROMPT_FILES` и рабочий
TTS. Если TTS недоступен, бот всё равно запустится, но приветствие будет
пустым — причина попадёт в лог.

```powershell
# AI-диалог: приветствие, ASR, LLM, TTS и веб-лента (рекомендуется)
python main.py incoming --ui --max-lines 1

# явно выбрать AI-бота без веб-интерфейса
python main.py incoming --bot ai --record-dir recordings

# эхо + ASR в UI без LLM; удобно для проверки входящего звука
python main.py incoming --ui --bot echo --max-lines 1

# только события звонка в UI, без обращения к ASR
python main.py incoming --ui --no-asr --ui-port 9000

# обычный WAV-бот без AI
python main.py incoming --bot placeholder --record-dir recordings --max-lines 4
```

После запуска с `--ui` откройте `http://127.0.0.1:8765/` (или свой `--ui-port`).
С `--ui-host 0.0.0.0` интерфейс доступен из сети, но авторизации нет.

### Боты (`--bot`)

| Спецификация | Поведение |
|---|---|
| `auto` | в `incoming`: AI-диалог, если настроены ASR/LLM/TTS, иначе `placeholder` |
| `ai` | в `incoming`: `DialogBot` — VAD → ASR → LLM → TTS по предложениям, приветствие, barge-in, no-input подсказка и завершение звонка |
| `silence` | ничего не отправляет (в `listen` это значение по умолчанию) |
| `echo` | возвращает входящий звук обратно |
| `tone` | тон 1000 Гц по 300 мс с паузой 2700 мс |
| `tone:<Гц>:<мс>` | тон заданной частоты и длительности, пауза 2700 мс |
| `wav:<путь>` | проигрывает WAV один раз, затем молчит |
| `wav:<путь>:loop` | проигрывает WAV по кругу |
| `placeholder` | короткое приветствие 500 мс (800 Гц), затем молчит |
| `placeholder:<мс>` | приветствие заданной длительности, затем молчит |

WAV ресемплируется в 8 кГц. В `dial` ключ `--play FILE` — это сокращение для
`--bot wav:FILE` без цикла. `auto`/`ai` работают только в `incoming`; для
`listen`/`dial` используйте обычные боты. Неизвестная спецификация завершает
запуск ошибкой конфигурации.

### `scripts/prompt_editor.py` — отдельный редактор промптов

```powershell
python scripts/prompt_editor.py [--dir DIR] [--host HOST] [--port PORT]
                                [--no-browser] [--verbose]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--dir DIR` | `<корень проекта>/prompts` | каталог, который показывать и редактировать |
| `--host HOST` | `127.0.0.1` | адрес прослушивания |
| `--port PORT` | `8766` | порт HTTP; `0` — выбрать свободный |
| `--no-browser` | выкл. | не открывать браузер автоматически |
| `--verbose` | выкл. | подробный лог HTTP-запросов |

Редактор не запускает SIP-клиент и не требует `.env`. Остановка — `Ctrl+C`.
Каталог `--dir` создаётся при запуске, если его нет; если путь указывает на
файл или каталог недоступен для записи, будет ошибка.

```powershell
# обычный запуск: откроется браузер на http://127.0.0.1:8766/
python scripts/prompt_editor.py

# другой каталог и порт, без автозапуска браузера
python scripts/prompt_editor.py --dir prompts --port 9000 --no-browser

# доступ из локальной сети (без авторизации!)
python scripts/prompt_editor.py --host 0.0.0.0 --port 9000

# свободный порт: адрес будет напечатан в консоли
python scripts/prompt_editor.py --port 0 --no-browser
```

### `scripts/live_incoming.py` — локальный harness одного входящего

Скрипт поднимает минимальную SIP-АТС и вызывающего на локальном IPv4,
запускает `main.py incoming`, делает один звонок, обменивается RTP, кладёт
трубку и сохраняет отчёт. Настоящая АТС и проброс портов не нужны.

```powershell
python scripts/live_incoming.py [ключи]
```

| Ключ | По умолчанию | Назначение |
|---|---|---|
| `--my-ip IP` | автоопределение | IPv4 интерфейса для SIP/RTP |
| `--sip-port PORT` | `15060` | порт harness-АТС |
| `--client-sip-port PORT` | `15061` | локальный SIP-порт клиента |
| `--client-user USER` | `SIP_USER` или `503` | SIP-логин тестируемого клиента |
| `--caller-user USER` | `508` | номер вызывающего абонента |
| `--rtp-low PORT` | `12100` | нижняя граница RTP-портов клиента |
| `--rtp-high PORT` | `12120` | верхняя граница RTP-портов клиента |
| `--bot SPEC` | `tone` | бот клиента (`tone`, `placeholder`, `echo`, `silence`, `wav:...`) |
| `--caller-wav FILE` | нет | WAV, который вызывающий отправляет клиенту (проверка ASR) |
| `--ui` | выкл. | включить веб-интерфейс клиента |
| `--ui-port PORT` | `8765` | порт веб-интерфейса клиента |
| `--no-asr` | выкл. | не включать ASR в клиенте |
| `--ui-hold SEC` | `0` | держать клиент и UI после звонка |
| `--duration SEC` | `8` | длительность разговора после ответа |
| `--record-dir DIR` | `recordings/live_incoming_one` | каталог результатов |
| `--keep` | выкл. | не очищать каталог результатов перед запуском |
| `--reg-timeout SEC` | `25` | ожидание регистрации клиента |

Результаты: `client.log`, `caller_heard.wav`, запись клиента и `report.txt` с
итогом `PASS/FAIL`.

```powershell
# базовый прогон без веба
python scripts/live_incoming.py

# проверить живую расшифровку на заранее записанной речи
python scripts/live_incoming.py --bot tone --ui --caller-wav echo_test.wav --ui-hold 120

# другой бот и каталог, сохранить предыдущие результаты
python scripts/live_incoming.py --bot echo --duration 15 --record-dir recordings/echo --keep
```

### `scripts/offline_smoke.py` — оффлайн-проверка

```powershell
python scripts/offline_smoke.py
```

Ключей нет. Проверяет `SipPhone` + `MediaBridge` + `Recorder` + `WavFileBot`
без сети, создаёт файлы в `.pytest_tmp/` и печатает статистику обмена.

### `scripts/sip_trace.py` — трассировка SIP/RTP

```powershell
python scripts/sip_trace.py [СЕКУНДЫ] [--rtp] [--stack] [--debug]
```

| Аргумент/ключ | По умолчанию | Назначение |
|---|---|---|
| `СЕКУНДЫ` | `10` | сколько секунд держать регистрацию и слушать сокеты |
| `--rtp` | выкл. | печатать каждый RTP-пакет (иначе только считать) |
| `--stack` | выкл. | печатать стек вызовов для исходящих SIP-сообщений |
| `--debug` | выкл. | включить хуки `register()` и таймера перерегистрации |

Скрипт читает `.env`, печатает каждое SIP-сообщение и итоговые счётчики
пакетов. Полезен, когда АТС не отвечает на REGISTER или «пропадает» звук.

```powershell
python scripts/sip_trace.py 15
python scripts/sip_trace.py 15 --rtp
python scripts/sip_trace.py 30 --debug --stack
```

## Структура проекта

```
main.py                 CLI: register / listen / dial / incoming
sipbot/config.py        Settings, чтение .env, валидация (SIP_MY_IP, порты)
sipbot/phone.py         SipPhone: VoIPPhone, регистрация, очередь входящих,
                        нормализация номера (pyVoIP ждёт номер, не URI)
sipbot/media.py         MediaBridge (2 потока: чтение/запись, такт 20 мс)
                        Recorder (потоковая запись в WAV)
sipbot/bots.py          SilenceBot / EchoBot / ToneBot / PlaceholderBot / WavFileBot + build_bot
sipbot/vad.py           EnergyVAD: сегментация речи для ASR и barge-in
sipbot/transcript.py    TranscriptStore: история реплик + подписки для UI
sipbot/transcribing_bot.py  обёртка обычного бота: VAD -> ASR -> лента разговора
sipbot/dialog.py        DialogBot: VAD -> ASR -> LLM -> TTS, история, метрики
sipbot/tts.py           TtsService: raw PCM/WAV -> 8 кГц, дисковый кэш
sipbot/prompts.py       безопасное чтение/создание/атомарное сохранение промптов
sipbot/prompt_ui_server.py  HTTP-сервер редактора промптов
sipbot/prompt_ui/index.html страница редактора с Markdown-просмотром
sipbot/ui_server.py     HTTP/SSE-сервер веб-интерфейса
sipbot/ui/index.html    страница с живыми пузырями разговора
sipbot/ai_client.py     OpenAI-совместимые ASR/LLM/TTS
sipbot/audio_utils.py   u8 <-> s16, G.711, ресемплинг, WAV, кадры
sipbot/pyvoip_fixes.py  патчи pyVoIP: 200 OK на OPTIONS, устойчивая регистрация
scripts/offline_smoke.py  оффлайн-проверка моста и записи на реальном API pyVoIP
scripts/live_incoming.py  живой harness: локальная АТС + один входящий звонок
scripts/prompt_editor.py  отдельный запуск веб-редактора промптов
scripts/sip_trace.py    трассировка SIP/RTP по сокетам (диагностика live-тестов)
tests/                  108 тестов без сети: аудио, боты, диалог, TTS, мост, UI
```

### Как устроен обмен звуком

```
АТС ──RTP──> pyVoIP (jitter buffer) ──read_audio(160, blocking=False)──> MediaBridge
                                                                          │
                                                    Recorder (WAV) <──────┤
                                                                          │
                                                    Bot.on_frame() ───────┤
                                                                          ▼
АТС <──RTP── pyVoIP <──write_audio(160)── очередь без потерь ───── MediaBridge
```

`MediaBridge` пишет тишину, когда у бота нет кадра, и держит исходящую
очередь неограниченной: длинное TTS-приветствие не обрезается. `DialogBot`
при этом отдаёт TTS по одному кадру на входящий кадр, поэтому очередь не
растёт, а при перебивании несыгранные кадры можно выбросить.

Чтение **неблокирующее** и пейсится ровно 20 мс на кадр. Так запись сохраняет
реальный таймлайн (тишина тоже попадает в WAV), а бот получает равномерный
поток кадров — это важно для VAD/ASR. `read_audio(blocking=True)` в pyVoIP,
наоборот, «съедает» тишину и отдаёт только непустые фрагменты, поэтому его
использовать не стоит.

Исходящий поток тоже идёт с тактом 20 мс; когда бот молчит, отправляется
явная тишина (`0x80 * 160`).

## Тесты

```powershell
python -m pytest -q
```

Тесты не требуют АТС и не открывают сокеты: используется фейковый звонок
(`FakeCall`) и фейковый UDP-сокет, проверяются конвертеры, WAV, боты, мост,
нормализация номеров, валидация конфига и патчи pyVoIP (OPTIONS, регистрация).
`--basetemp=.pytest_tmp` в `pytest.ini` держит временные
файлы внутри проекта (в песочницах `%TEMP%` может быть недоступен).

Дополнительно есть оффлайн-смоук, который проверяет связку `SipPhone` →
`MediaBridge` → `Recorder` на реальном API pyVoIP (сокеты не открываются):

```powershell
python scripts/offline_smoke.py
# status before start: INACTIVE
# recording: 4160 frames @ 8000 Hz
# stats: {'frames_in': 26, 'frames_out': 10, 'dropped': 0, 'level': 0.0}
# frames written to call: 25
# greeting frames sent: 10
```

Числа зависят от таймингов планировщика (кадры 20 мс, ±несколько кадров),
важны сами факты: запись идёт, `frames_out > 0`, `dropped == 0`.


## Проверено на живой АТС

Стенд: Asterisk 11.7 (`sip.evpanet.com:5060/UDP`), клиент в сети, где АТС видит
его локальный адрес (`received=172.18.185.162` в Via), т.е. NAT на пути нет.

| Проверка | Результат |
|---|---|
| `register` | `401` → `REGISTER` с Digest → `200 OK`, `Expires: 120` |
| Contact в ответе АТС | `<sip:503@172.18.185.162:5060;transport=UDP>;expires=120` |
| OPTIONS keepalive от АТС | `200 OK` на каждый запрос, ретрансмиссий от АТС нет |
| Лишние REGISTER | нет, `registerFailures = 0`, перерегистрация через 115 с |
| Deregister при выходе | `REGISTER` с `Expires: 0` → `401` → `200 OK` |
| Обмен SIP за сессию | 5 отправлено / 5 получено, ошибок нет |
| `dial` на несуществующий номер | `INVITE` → `404 Not Found` → `state=ENDED`, recv-поток жив, без traceback |

Входящий/исходящий звонок и RTP в обе стороны требуют второго абонента —
после звонка на `503` смотрите статистику `MediaBridge.stats()`
(`frames_in`/`frames_out` должны расти в обе стороны).

Трассировка обмена по сокетам (каждый пакет, тред и при необходимости стек):

```powershell
python scripts/sip_trace.py 8                  # 8 секунд регистрации
python scripts/sip_trace.py 8 --rtp            # плюс каждый RTP-пакет
python scripts/sip_trace.py 8 --debug --stack  # хуки register() + стек отправителя
```

## Диагностика

| Симптом | Причина / что делать |
|---|---|
| `Регистрация не удалась` | неверный логин/пароль, АТС недоступна, UDP 5060 закрыт, неверный `SIP_PORT` |
| Регистрация ок, но тишина в обе стороны | `SIP_MY_IP` не тот интерфейс; RTP-порты закрыты брандмауэром |
| `REGISTERED`, но входящие не приходят | АТС помечает абонента недоступным (не ответили на OPTIONS/qualify) — патч `pyvoip_fixes` должен быть применён, проверьте `python scripts/sip_trace.py 5` |
| Статус иногда `FAILED` сразу после старта | дефект pyVoIP с OPTIONS во время регистрации (закрыт патчем); заодно убедитесь, что не запущен второй экземпляр на том же `SIP_LOCAL_PORT` |
| Слышно только в одну сторону | в SDP уходит `0.0.0.0` (пустой `SIP_MY_IP`) или NAT между клиентом и АТС |
| Звонок идёт, но собеседник не слышит бота | проверьте `--bot tone`: если тон слышно — проблема в боте, а не в SIP |
| `Address already in use` | занят `SIP_LOCAL_PORT`/RTP-диапазон (запущен второй экземпляр) |
| Рвётся через ~30 с | АТС не получает RTP: порты/адрес, либо `--duration` меньше времени разговора |
| Запись короче разговора | записывается только входящий поток и только пока звонок в состоянии `ANSWERED` |

Полезно: `python main.py -v dial 1001 --bot tone --duration 5` — подробные
логи pyVoIP и статистика кадров (`frames_in`, `frames_out`, `dropped`, `level`).

## Что дальше

1. **Качество VAD/ASR**: сейчас `EnergyVAD` и Whisper Large; для шумной линии
   имеет смысл `webrtcvad`/Silero и streaming ASR, чтобы отвечать до паузы 900 мс.
2. **Скорость TTS**: ответ теперь синтезируется по предложениям, и первое
   предложение отдаётся в линию сразу. Логи `Задержка ответа` показывают
   `queue`, ASR, LLM, полный TTS, время до первого звука (`first_audio_ms`)
   и итог по каждой реплике. Следующий шаг — streaming TTS внутри предложения
   и параллельный синтез, чтобы убрать паузу между предложениями.
3. **Микрофон/колонки**: `sounddevice` уже в зависимостях; нужен ресемплер
   48 кГц <-> 8 кГц (`resample_s16` есть) и подавление эха (иначе петля).
4. **DTMF**: в pyVoIP 1.6.8 исходящего DTMF нет — либо патч RTP (RFC 4733),
   либо переход на pjsua2.
5. **Продакшен**: pjsua2 (pjproject) или FreeSWITCH + внешний media-сервер;
   там есть Opus, SRTP, AEC, VAD и полноценный DTMF.

## Лицензия

Код проекта — учебный/внутренний. pyVoIP распространяется под GPLv3:
учитывайте это при распространении бинарников/сервиса.

