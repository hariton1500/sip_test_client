# SIP test client (pyVoIP) для голосового бота

Минимальный SIP-клиент на Python: регистрация на АТС, входящие и исходящие
звонки, запись разговора в WAV, эхо-бот и бот-проигрыватель WAV.
Задуман как «испытательный стенд»: SIP-транспорт отделён от обработки звука,
поэтому VAD/ASR/LLM/TTS подключаются без переделки сети.

## Что умеет

| Команда | Назначение |
|---|---|
| `register` | только регистрация (проверка логина/пароля/сети) |
| `listen` | принимать входящие, отвечать, писать в WAV, запускать бота |
| `dial` | исходящий звонок, проигрывание WAV, запись ответа |

Боты (`--bot`): `silence`, `echo`, `tone`, `tone:<Гц>:<мс>`,
`wav:<путь>[:loop]`. Бот получает входящий кадр и возвращает кадр для отправки.

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

```ini
SIP_SERVER=192.168.1.10     # адрес АТС/Asterisk/FreeSWITCH
SIP_PORT=5060               # порт АТС (UDP)
SIP_USER=1000               # логин/внутренний номер
SIP_PASS=secret
SIP_MY_IP=192.168.1.50      # ОБЯЗАТЕЛЬНО: локальный IPv4 этой машины
SIP_LOCAL_PORT=5060         # локальный порт SIP (UDP)
RTP_PORT_LOW=10000
RTP_PORT_HIGH=10100
```

`SIP_MY_IP` проверяется при старте: `0.0.0.0`, `127.0.0.1` и не-IP значения
отклоняются. Причина — pyVoIP подставляет `myIP` в SDP (`c=IN IP4 ...`);
с `0.0.0.0` АТС не знает, куда слать RTP, и получается one-way audio.

Брандмауэр Windows: разрешите входящий UDP на `SIP_LOCAL_PORT` и на диапазон
`RTP_PORT_LOW..RTP_PORT_HIGH` для `python.exe`.

## Запуск

```powershell
# 1. Только регистрация (быстрая проверка учётки и сети)
python main.py register --hold 5

# 2. Приём звонка: эхо + запись
python main.py listen --bot echo --record incoming.wav --duration 60

# 3. Приём звонка с проигрыванием приветствия (в цикле)
python main.py listen --bot wav:greeting.wav:loop --record incoming.wav

# 4. Исходящий звонок: проиграть файл, записать ответ, 20 секунд
python main.py dial 1001 --play greeting.wav --record answer.wav --duration 20

# 5. Исходящий звонок: эхо (проверка двустороннего аудио)
python main.py dial 1001 --bot echo --duration 15

# 6. Проверка исходящего звука тоном 1 кГц
python main.py dial 1001 --bot tone --duration 10
```

`--timeout` в `dial` — ожидание ответа, в `listen` — общее время ожидания
звонков (0 = бесконечно). `--duration 0` — говорить до отбоя.

Логи идут в stderr, запись — в указанный WAV (моно, 16 бит, 8 кГц; при
необходимости пересемплируйте: `ffmpeg -i in.wav -ar 16000 out.wav`).

## Структура проекта

```
main.py                 CLI: register / listen / dial
sipbot/config.py        Settings, чтение .env, валидация (SIP_MY_IP, порты)
sipbot/phone.py         SipPhone: VoIPPhone, регистрация, очередь входящих,
                        нормализация номера (pyVoIP ждёт номер, не URI)
sipbot/media.py         MediaBridge (2 потока: чтение/запись, такт 20 мс)
                        Recorder (потоковая запись в WAV)
sipbot/bots.py          SilenceBot / EchoBot / ToneBot / WavFileBot + build_bot
sipbot/audio_utils.py   u8 <-> s16, G.711, ресемплинг, WAV, кадры
sipbot/pyvoip_fixes.py  патчи pyVoIP: 200 OK на OPTIONS, устойчивая регистрация
scripts/offline_smoke.py  оффлайн-проверка моста и записи на реальном API pyVoIP
scripts/sip_trace.py    трассировка SIP/RTP по сокетам (диагностика live-тестов)
tests/                  69 тестов без сети: аудио, боты, мост, конфиг, патчи
```

### Как устроен обмен звуком

```
АТС ──RTP──> pyVoIP (jitter buffer) ──read_audio(160, blocking=False)──> MediaBridge
                                                                          │
                                                    Recorder (WAV) <──────┤
                                                                          │
                                                    Bot.on_frame() ───────┤
                                                                          ▼
АТС <──RTP── pyVoIP <──write_audio(160)── очередь (drop-oldest) ── MediaBridge
```

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

1. **VAD + ASR**: в `MediaBridge._read_loop` кадры уже приходят по 20 мс;
   подключите `webrtcvad`/`silero-vad`, затем Whisper (faster-whisper) — только
   на сегментах речи.
2. **TTS**: синтез в 8 кГц (или 16/24 кГц + `resample_s16`) и `WavFileBot`-подобная
   очередь; стриминг — тем же способом, что `ToneBot`.
3. **Микрофон/колонки**: `sounddevice` уже в зависимостях; нужен ресемплер
   48 кГц <-> 8 кГц (`resample_s16` есть) и подавление эха (иначе петля).
4. **DTMF**: в pyVoIP 1.6.8 исходящего DTMF нет — либо патч RTP (RFC 4733),
   либо переход на pjsua2.
5. **Продакшен**: pjsua2 (pjproject) или FreeSWITCH + внешний media-сервер;
   там есть Opus, SRTP, AEC, VAD и полноценный DTMF.

## Лицензия

Код проекта — учебный/внутренний. pyVoIP распространяется под GPLv3:
учитывайте это при распространении бинарников/сервиса.

