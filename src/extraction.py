"""Extraccion de duracion de sintomas con regex y un transductor (FST)."""

import re

# numeros escritos en palabra -> numero
_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "few": 3, "several": 4, "many": 5,
}

# unidad de tiempo -> dias
_UNIT_TO_DAYS = {
    "second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24,
    "day": 1, "week": 7, "month": 30, "year": 365, "decade": 3650,
}

# la regex busca: disparador (for/since/...) + cantidad opcional + unidad
DURATION_RE = re.compile(
    r"""
    \b(?P<trig>for|since|over|past|last|about|around|been)\s+
    (?:the\s+)?
    (?:a\s+|an\s+)?
    (?P<qty>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|
            eleven|twelve|couple|few|several|many)?
    \s*(?:of\s+)?
    (?P<unit>second|minute|hour|day|week|month|year|decade)s?
    (?=\b|\W|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _transduce_quantity(qty_token: str | None) -> int | None:
    if qty_token is None:
        return 1  # "last week" = 1 semana
    qty_token = qty_token.lower()
    if qty_token.isdigit():
        return int(qty_token)
    return _WORD_NUM.get(qty_token)


def _transduce_unit(unit_token: str) -> float | None:
    return _UNIT_TO_DAYS.get(unit_token.lower())


def extract_durations(text: str) -> list[dict]:
    # encuentra cada duracion y la convierte a dias
    results = []
    for m in DURATION_RE.finditer(text):
        qty = _transduce_quantity(m.group("qty"))
        factor = _transduce_unit(m.group("unit"))
        if qty is None or factor is None:
            continue
        qty_str = m.group("qty") if m.group("qty") else "1"
        results.append({
            "match":    f"{qty_str} {m.group('unit')}".strip(),
            "trigger":  m.group("trig").lower(),
            "quantity": qty,
            "unit":     m.group("unit").lower(),
            "days":     round(qty * factor, 2),
        })
    return results


if __name__ == "__main__":
    pruebas = [
        "I have felt empty for 3 years and tired since last week.",
        "This has been going on for about a couple of months now.",
        "Symptoms for several days, and over the past decade it worsened.",
        "I went to the store and bought milk.",
    ]
    for p in pruebas:
        print("TEXTO:", p)
        for d in extract_durations(p):
            print("   ->", d)
        print()
