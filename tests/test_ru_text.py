"""Тесты нормализации русского текста для локального TTS."""

from __future__ import annotations

import pytest

from sipbot.ru_text import digits_to_words, int_to_words, normalize_for_tts


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "ноль"),
        (1, "один"),
        (11, "одиннадцать"),
        (21, "двадцать один"),
        (100, "сто"),
        (1000, "одна тысяча"),
        (2000, "две тысячи"),
        (5000, "пять тысяч"),
        (2026, "две тысячи двадцать шесть"),
        (1_500_000, "один миллион пятьсот тысяч"),
        (2_000_001, "два миллиона один"),
    ],
)
def test_int_to_words(value, expected):
    assert int_to_words(value) == expected


def test_digits_to_words():
    assert digits_to_words("150") == "один пять ноль"


def test_normalize_currency_agreement():
    assert (
        normalize_for_tts("Баланс 1500 руб.")
        == "Баланс одна тысяча пятьсот рублей"
    )
    assert normalize_for_tts("1 руб.") == "один рубль"
    assert normalize_for_tts("2 руб.") == "два рубля"
    assert normalize_for_tts("5 руб.") == "пять рублей"
    assert (
        normalize_for_tts("429,56 руб.")
        == "четыреста двадцать девять рублей пятьдесят шесть копеек"
    )
    assert normalize_for_tts("1 коп.") == "одна копейка"
    assert normalize_for_tts("2 коп.") == "две копейки"
    assert normalize_for_tts("5 коп.") == "пять копеек"


def test_normalize_grouped_and_long_numbers():
    assert (
        normalize_for_tts("1 500 000 рублей")
        == "один миллион пятьсот тысяч рублей"
    )
    # Номер лицевого счёта читается по цифрам, а не как число.
    assert (
        normalize_for_tts("л/с 123456789")
        == "лицевой счёт один два три четыре пять шесть семь восемь девять"
    )


def test_normalize_percent_and_decimal():
    assert normalize_for_tts("12,5 %") == "двенадцать запятая пять процентов"
    assert normalize_for_tts("15%") == "пятнадцать процентов"
    assert normalize_for_tts("2,5") == "два запятая пять"


def test_normalize_abbreviations_and_punctuation():
    assert normalize_for_tts("т.е. да") == "то есть да"
    assert normalize_for_tts("№ 42") == "номер сорок два"
    assert normalize_for_tts("  привет   мир  ") == "привет мир"


def test_normalize_markdown_links_and_emoji():
    text = "**[Тариф](https://example.com) 😊**"
    assert normalize_for_tts(text) == "Тариф"


def test_normalize_latin():
    assert normalize_for_tts("EVP") == "е вэ пи"
    assert normalize_for_tts("Wi-Fi") == "ви-фи"


def test_normalize_is_idempotent():
    once = normalize_for_tts("Здравствуйте! Баланс 1500 руб.")
    assert normalize_for_tts(once) == once


def test_normalize_empty():
    assert normalize_for_tts("") == ""
    assert normalize_for_tts("   ") == ""
