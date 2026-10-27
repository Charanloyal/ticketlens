"""ITIL keyword cues used to correct the statistical model on short, hand-written tickets.

The training dataset is long-form and synthetic, and its labels do not consistently follow ITIL
definitions (e.g. "recurring outage, find the root cause" is often labelled Incident). These cues
encode the ITIL definitions directly and are applied as a bounded log-probability boost, so the
model still decides when no cue matches, and a single cue can never fully override it.
"""

import math
import re
from typing import Dict, List, Tuple

_FLAGS = re.IGNORECASE

# (cue name, pattern) per ITIL category. Patterns describe the *intent* of each category:
#   Request  - asking for information, access, documents or standard items
#   Incident - something is broken or unavailable right now
#   Problem  - recurring/persistent issues or requests for root-cause analysis
#   Change   - planned modification of systems, configuration or infrastructure
ITIL_CUES: Dict[str, List[Tuple[str, "re.Pattern[str]"]]] = {
    "Request": [
        ("billing document", re.compile(r"\b(invoices?|receipts?|quotes?|quotation|pricing|price list|statement)\b", _FLAGS)),
        ("access request", re.compile(r"\b(access to|permissions? (on|to|for)|grant (me |us )?access|read-only)\b", _FLAGS)),
        ("new account/item", re.compile(r"\b(create|set ?up|need|provision|order)\b.{0,30}\b(account|mailbox|laptop|monitors?|licen[cs]es?|seats?|phone|headset)\b", _FLAGS)),
        ("password reset", re.compile(r"\b(reset (my |the )?password|password reset|forgot (my )?password)\b", _FLAGS)),
        ("information request", re.compile(r"\b(information|details|documentation|guide)\b.{0,15}\b(on|about|for|of)\b", _FLAGS)),
        ("how-to question", re.compile(r"^\s*(how (do|can|to|should)|what (is|are)|where (can|do|is)|when (is|are|will)|is there)\b", _FLAGS)),
        ("send/share request", re.compile(r"\b(send|share|email|forward|provide)\b.{0,12}\b(me|us)\b", _FLAGS)),
    ],
    "Incident": [
        ("service down", re.compile(r"\b(is|are|went|goes) down\b|\boutage\b(?!s)|\bunavailable\b", _FLAGS)),
        ("not working", re.compile(r"\b(not (working|loading|responding)|stopped (working|responding)|cannot|can't|unable to)\b", _FLAGS)),
        ("error now", re.compile(r"\b(right now|since (this|today|yesterday)|just now|at the moment)\b", _FLAGS)),
    ],
    "Problem": [
        ("root cause", re.compile(r"\b(root[- ]cause|underlying cause|permanent fix|find out why|investigate why|why)\b", _FLAGS)),
        # "intermittent" is deliberately excluded: an intermittent symptom of a live outage is an Incident.
        ("recurrence", re.compile(r"\b(recurring|recurrent|recur|keeps?|kept|repeatedly|again|persistent|ongoing|frequent(ly)?|consistently|every (day|week|month|monday|morning|few))\b", _FLAGS)),
        ("multiple occurrences", re.compile(r"\b((\d+|two|three|four|five|several|multiple)(th)? times|(second|third|fourth|fifth) time|multiple incidents|pattern)\b", _FLAGS)),
    ],
    "Change": [
        ("upgrade/migration", re.compile(r"\b(upgrade|migrat(e|ion)|move (our|the)|switch (to|our|from)|roll ?out|deploy (the|a)? ?new|decommission)\b", _FLAGS)),
        ("configuration change", re.compile(r"\b(reconfigure|configure|modify|change the|update the|enable|disable|add (sso|mfa|support)|implement)\b", _FLAGS)),
        ("planned work", re.compile(r"\b(maintenance window|schedule (a|the)|plan(ned)? to|next (quarter|weekend|month)|this weekend|patch(es|ing)?)\b", _FLAGS)),
    ],
}

# Log-probability boost per matched cue (capped), tuned so a strong cue can flip a low-confidence
# prediction but not a confident one with an opposing signal.
CUE_WEIGHT = 1.5
MAX_CUES_PER_CLASS = 2


def match_itil_cues(text: str) -> Dict[str, List[str]]:
    """Return the matched cue names per category (only categories with at least one match)."""
    matches: Dict[str, List[str]] = {}
    for category, cues in ITIL_CUES.items():
        hit = [name for name, pattern in cues if pattern.search(text)]
        if hit:
            matches[category] = hit
    return matches


def apply_itil_cues(
    probabilities: Dict[str, float],
    text: str,
    weight: float = CUE_WEIGHT,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Re-weight class probabilities with ITIL cues. Returns (adjusted probabilities, matched cues)."""
    matches = match_itil_cues(text)
    if not matches or weight <= 0:
        return dict(probabilities), matches

    logits = {}
    for label, p in probabilities.items():
        boost = weight * min(len(matches.get(label, [])), MAX_CUES_PER_CLASS)
        logits[label] = math.log(max(p, 1e-9)) + boost

    max_logit = max(logits.values())
    exp = {label: math.exp(v - max_logit) for label, v in logits.items()}
    total = sum(exp.values())
    return {label: v / total for label, v in exp.items()}, matches
