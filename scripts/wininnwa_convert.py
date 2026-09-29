"""WinInnwa -> Unicode conversion for Than Tun, Royal Orders of Burma (born-digital PDFs).

Uses python-myanmar's WinInnwa table with two patches:
  1. '0' is mapped both to the letter wa and to digit zero; the digit wins and the
     vowel signs after it are left unconverted (e.g. '0g' -> '၀g'). Here '0' is always
     read as wa, and wa is turned back into digit zero only next to other digits.
  2. 'Å' (a stack-ta glyph variant) is missing from the table; it is added as stack-ta.
"""
import json
import pkgutil
import re

from myanmar import converter, encodings

_DIGITS = "၁၂၃၄၅၆၇၈၉၀"
_WA = "ဝ"


def _build_patched_encoding():
    orig = pkgutil.get_data
    data = json.loads(orig("myanmar", "data/wininnwa.json").decode("utf-8"))
    data["digit"].pop("zero", None)
    data["stack"]["stack-ta_alt"] = "Å"  # 'Å'
    blob = json.dumps(data).encode("utf-8")

    # WininnwaEncoding reads its table via pkgutil.get_data in __init__; serve the patched one.
    def patched(pkg, name):
        return blob if name == "data/wininnwa.json" else orig(pkg, name)

    encodings.pkgutil.get_data = patched
    try:
        return encodings.WininnwaEncoding()
    finally:
        encodings.pkgutil.get_data = orig


converter.encoders["wininnwa_rob"] = _build_patched_encoding()


def _restore_zero(u: str) -> str:
    # wa standing next to a digit is digit zero (e.g. ၅၀, ၁၀၀)
    prev = None
    while prev != u:
        prev = u
        u = re.sub(rf"(?<=[{_DIGITS}]){_WA}|{_WA}(?=[{_DIGITS}])", "၀", u)
    return u


# typing-order variants: i/ii vowel keyed before ha-htoe ('dS' for 'Sd')
_PRE = [("dS", "Sd"), ("DS", "SD")]


def wininnwa_to_unicode(text: str) -> str:
    for a, b in _PRE:
        text = text.replace(a, b)
    return _restore_zero(converter.convert(text, "wininnwa_rob", "unicode"))
