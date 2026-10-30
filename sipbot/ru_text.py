"""Подготовка русского текста для локального TTS.

Kokoro-ru получает не текст, а фонемы, поэтому весь «фронтенд» (числа,
аббревиатуры, латиница, разметка) нужно привести к обычной русской речи
заранее. Облачный Gemini делает это сам, локальной модели такой слой нужен
явно — иначе «л/с 123456789» читается побуквенно, а «1 500 руб.» теряет
согласование.

Модуль намеренно не зависит от TTS-движка: его можно тестировать отдельно.
"""

from __future__ import annotations

import re

_UNITS_M = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)
_UNITS_F = ("ноль", "одна", "две") + _UNITS_M[3:]
_UNITS_N = ("ноль", "одно", "два") + _UNITS_M[3:]
_UNITS = {"m": _UNITS_M, "f": _UNITS_F, "n": _UNITS_N}

_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)

#: Разряды: (единственное, 2-4, много, род числительного).
_SCALES = (
    ("", "", "", "m"),
    ("тысяча", "тысячи", "тысяч", "f"),
    ("миллион", "миллиона", "миллионов", "m"),
    ("миллиард", "миллиарда", "миллиардов", "m"),
)

_DIGIT_WORDS = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)

_LATIN_DIGRAPHS = (
    ("sch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("ph", "ф"),
    ("th", "т"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ya", "я"),
    ("yu", "ю"),
    ("yo", "ё"),
    ("je", "е"),
    ("oo", "у"),
)
_LATIN_LETTERS = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
}
#: Как русский диктор называет латинские буквы (EVP -> «е вэ пэ»).
_LATIN_LETTER_NAMES = {
    "a": "эй",
    "b": "би",
    "c": "си",
    "d": "ди",
    "e": "е",
    "f": "эф",
    "g": "джи",
    "h": "эйч",
    "i": "ай",
    "j": "джей",
    "k": "кей",
    "l": "эл",
    "m": "эм",
    "n": "эн",
    "o": "оу",
    "p": "пи",
    "q": "кью",
    "r": "ар",
    "s": "эс",
    "t": "ти",
    "u": "ю",
    "v": "вэ",
    "w": "дабл-ю",
    "x": "экс",
    "y": "уай",
    "z": "зет",
}

_ABBREVIATIONS = (
    (re.compile(r"\bл\s*/\s*с\b", re.IGNORECASE), "лицевой счёт"),
    (re.compile(r"\bт\.\s*д\.", re.IGNORECASE), "так далее"),
    (re.compile(r"\bт\.\s*п\.", re.IGNORECASE), "тому подобное"),
    (re.compile(r"\bт\.\s*е\.", re.IGNORECASE), "то есть"),
    (re.compile(r"\bт\.\s*к\.", re.IGNORECASE), "так как"),
    (re.compile(r"\bдр\.", re.IGNORECASE), "другие"),
    (re.compile(r"\bул\.", re.IGNORECASE), "улица"),
    (re.compile(r"\bкв\.", re.IGNORECASE), "квартира"),
    (re.compile(r"\bдом\.", re.IGNORECASE), "дом"),
    (re.compile(r"\bтел\.", re.IGNORECASE), "телефон"),
    (re.compile(r"\bг\.", re.IGNORECASE), "город"),
    (re.compile(r"\bруб\b\.?", re.IGNORECASE), "рублей"),
)

_MARKUP_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MARKDOWN_CHARS = re.compile(r"[*_`#>~|]+")
_EMOJI = re.compile(
    "[^0-9A-Za-zА-Яа-яЁё\\s.,!?;:—…()«»\"'\\-+/%₽№]",
    re.UNICODE,
)
_SPACES = re.compile(r"\s+")
_CURRENCY = re.compile(
    r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,](\d{1,2}))?\s*"
    r"(?:₽|руб\.?|рублей|рубля|рубль)(?=\W|$)",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*%")
_KOPECKS = re.compile(
    r"(?<!\d)(\d+)\s*(?:коп\.?|копеек|копейки|копейка)(?=\W|$)",
    re.IGNORECASE,
)
_THOUSANDS = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+)(?!\d)")
_DECIMAL = re.compile(r"(?<!\d)(\d+)[.,](\d+)(?!\d)")
_INTEGER = re.compile(r"(?<!\d)(\d+)(?!\d)")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_PHONE_PLUS = re.compile(r"\+\s*(?=\d)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:…)\]»])")


def _plural(value: int, one: str, few: str, many: str) -> str:
    """Выбрать форму существительного для числа (1 рубль / 2 рубля / 5 рублей)."""
    value = abs(value) % 100
    if 11 <= value <= 19:
        return many
    value %= 10
    if value == 1:
        return one
    if 2 <= value <= 4:
        return few
    return many


def _triad_words(value: int, gender: str = "m") -> list:
    """Числа 0..999 словами с учётом рода (одна тысяча, два миллиона)."""
    words = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        tens, units = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if units:
            words.append(_UNITS[gender][units])
    return words


def int_to_words(value: int) -> str:
    """Целое неотрицательное число словами (до миллиардов)."""
    if value < 0:
        return "минус " + int_to_words(-value)
    if value == 0:
        return "ноль"
    groups = []
    index = 0
    while value > 0:
        value, group = divmod(value, 1000)
        if group:
            if index >= len(_SCALES):
                raise ValueError("Число слишком велико для озвучивания")
            one, few, many, gender = _SCALES[index]
            words = _triad_words(group, gender)
            if index:
                words.append(_plural(group, one, few, many))
            groups.insert(0, " ".join(words))
        index += 1
    return " ".join(groups)


def digits_to_words(digits: str) -> str:
    """Прочитать последовательность цифр по одной (номера счетов, телефоны)."""
    return " ".join(_DIGIT_WORDS[int(ch)] for ch in digits if ch.isdigit())


def _number_by_context(digits: str) -> str:
    """Длинные последовательности читаем по цифрам, короткие — числом."""
    if len(digits) >= 7:
        return digits_to_words(digits)
    return int_to_words(int(digits))


def _spoken_integer(raw: str) -> str:
    """Число из текста; группы по три разряда читаем как число, не по цифрам."""
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if any(ch.isspace() for ch in raw):
        try:
            return int_to_words(int(digits))
        except ValueError:
            return digits_to_words(digits)
    return _number_by_context(digits)


def _currency_words(match: re.Match) -> str:
    raw = match.group(1)
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return match.group(0)
    value = int(digits)
    result = f"{_spoken_integer(raw)} {_plural(value, 'рубль', 'рубля', 'рублей')}"
    kopecks = match.group(2)
    if kopecks:
        amount = int(kopecks)
        result += (
            f" {int_to_words(amount)} "
            f"{_plural(amount, 'копейка', 'копейки', 'копеек')}"
        )
    return result


def _kopecks_words(match: re.Match) -> str:
    amount = int(match.group(1))
    words = "ноль" if amount == 0 else " ".join(_triad_words(amount, "f"))
    return f"{words} {_plural(amount, 'копейка', 'копейки', 'копеек')}"


def _percent_words(match: re.Match) -> str:
    raw = match.group(1)
    if "," in raw or "." in raw:
        whole, fraction = re.split(r"[.,]", raw, maxsplit=1)
        spoken = f"{int_to_words(int(whole))} запятая {digits_to_words(fraction)}"
        value = int(whole)
    else:
        spoken = _number_by_context(raw)
        value = int(raw)
    return f"{spoken} {_plural(value, 'процент', 'процента', 'процентов')}"


def _decimal_words(match: re.Match) -> str:
    whole, fraction = match.group(1), match.group(2)
    return f"{_number_by_context(whole)} запятая {digits_to_words(fraction)}"


def _thousands_words(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(1))
    return int_to_words(int(digits))


def _integer_words(match: re.Match) -> str:
    return _number_by_context(match.group(1))


def _translit_latin_word(word: str) -> str:
    """Латиница -> кириллица; аббревиатуры читаются названиями букв."""
    letters = word.replace("-", "").replace(" ", "")
    if letters.isupper() and 2 <= len(letters) <= 6 and not any(
        ch.isdigit() for ch in letters
    ):
        return " ".join(_LATIN_LETTER_NAMES.get(ch.lower(), ch) for ch in letters)

    result = []
    lower = word.lower()
    index = 0
    while index < len(lower):
        for latin, cyrillic in _LATIN_DIGRAPHS:
            if lower.startswith(latin, index):
                result.append(cyrillic)
                index += len(latin)
                break
        else:
            char = lower[index]
            if char.isdigit() or char == "-":
                result.append(char)
            else:
                result.append(_LATIN_LETTERS.get(char, ""))
            index += 1
    return "".join(result)


def _latin_words(match: re.Match) -> str:
    return _translit_latin_word(match.group(0))


def normalize_for_tts(text: str) -> str:
    """Привести ответ LLM к виду, пригодному для kokoro-ru.

    Что делается:

    * убирается markdown, ссылки и эмодзи;
    * раскрываются частые сокращения (``л/с``, ``т.д.``, ``руб.``);
    * числа, суммы, проценты и длинные номера читаются словами;
    * латиница транслитерируется, аббревиатуры — по буквам.

    Функция идемпотентна по смыслу: повторный вызов на уже нормализованном
    тексте ничего не меняет, кроме безопасной нормализации пробелов.
    """
    if not text:
        return ""

    result = _MARKUP_LINK.sub(r"\1", text)
    result = _URL.sub(" ", result)
    result = _MARKDOWN_CHARS.sub(" ", result)
    result = _EMOJI.sub(" ", result)
    result = _PHONE_PLUS.sub("", result)

    for pattern, replacement in _ABBREVIATIONS:
        result = pattern.sub(replacement, result)

    result = _CURRENCY.sub(_currency_words, result)
    result = _KOPECKS.sub(_kopecks_words, result)
    result = _PERCENT.sub(_percent_words, result)
    result = re.sub(r"№\s*", "номер ", result)
    result = _THOUSANDS.sub(_thousands_words, result)
    result = _DECIMAL.sub(_decimal_words, result)
    result = _INTEGER.sub(_integer_words, result)
    result = _LATIN_WORD.sub(_latin_words, result)

    result = result.replace("\u00a0", " ")
    result = _SPACES.sub(" ", result)
    result = _SPACE_BEFORE_PUNCT.sub(r"\1", result)
    return result.strip()


__all__ = [
    "digits_to_words",
    "int_to_words",
    "normalize_for_tts",
]
