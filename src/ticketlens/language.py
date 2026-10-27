"""Lightweight, dependency-free language check for incoming tickets.

The models are trained on English tickets only, so non-English input is still classified but
flagged and routed to human review. The check is intentionally conservative: it only fires on a
clear signal (non-Latin script, or more foreign than English function words) to avoid flagging
terse English technical text such as "VPN TLS error 504".
"""

import re
import unicodedata
from typing import Optional

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_ENGLISH = frozenset(
    "the a an and or but is are was were be been to of in on at for with from by this that it its "
    "i we you my our your me us can could would should will not no please have has had do does did "
    "cannot can't don't isn't since after when why how what where which there here all any".split()
)

# Frequent function words that are not also common English words.
_FOREIGN = {
    "German": frozenset("der die das und ist nicht ich wir sie mit für auf ein eine einen dem den des bei "
                        "kann können wurde werden bitte unser unsere mein meine kein keine auch noch".split()),
    "Spanish": frozenset("el la los las y es está están por para con una uno del al que pero mi mis "
                         "nuestro nuestra puedo podemos funciona favor hola gracias también".split()),
    "French": frozenset("le la les et est sont pour avec une des du au aux que qui ne pas mon ma mes "
                        "nous vous je fonctionne merci bonjour aussi".split()),
    "Portuguese": frozenset("o os as e é está estão para com uma um do da dos das ao que não meu minha "
                            "nosso nossa obrigado olá também funciona".split()),
    "Italian": frozenset("il lo gli e è sono con una uno del della che non mio mia nostro nostra "
                         "grazie ciao anche funziona".split()),
    "Dutch": frozenset("de het een en is zijn niet ik wij met voor op van dat mijn onze kan graag "
                       "bedankt ook werkt".split()),
}

# Foreign function words that also appear in English IT tickets ("as", "OS", "I/O", ".com", "MIT",
# "die", "den", "van", "op", "um") and must never count as evidence of another language.
_AMBIGUOUS = frozenset("as os com mit die den van op um al lo".split())


def _non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
    return non_latin / len(letters)


def detect_non_english(text: str) -> Optional[str]:
    """Return a human-readable warning if the text does not look like English, else None."""
    if _non_latin_ratio(text) > 0.3:
        return "Ticket appears to use a non-Latin script; models are trained on English tickets only."

    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 3:
        return None

    english_hits = sum(1 for w in words if w in _ENGLISH)
    candidates = [w for w in words if len(w) > 1 and w not in _ENGLISH and w not in _AMBIGUOUS]
    best_lang, best_hits = max(
        ((lang, sum(1 for w in candidates if w in vocab)) for lang, vocab in _FOREIGN.items()),
        key=lambda item: item[1],
    )
    if best_hits >= 2 and best_hits > english_hits:
        return f"Ticket appears to be written in {best_lang}; models are trained on English tickets only."
    return None
