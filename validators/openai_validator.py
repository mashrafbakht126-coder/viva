"""
validators/openai_validator.py
──────────────────────────────────────────────────────────────────────────────
OpenAI AI Validator for ALL diagram types:
  - Class Diagram
  - Use Case Diagram
  - Sequence Diagram

Model: gpt-4o-2024-08-06 (pinned stable version)
Rate limit (429): auto wait + retry

FIX: Shape sanitizer added — strips ToolType. prefix, filters shapes by
     diagram type so OpenAI never sees actor/systemBoundary shapes when
     validating a class diagram (and vice versa).
     Prompt headers now explicitly forbid cross-diagram rules.
"""

import os
import json
import logging
import math
import re
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

_log = logging.getLogger(__name__)

_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
]

_OPENAI_API_BASE = "https://api.openai.com/v1"
_TIMEOUT_SECONDS = 120
_RETRY_WAIT      = 65

_SYSTEM_MESSAGE = (
    "You are a strict, deterministic UML diagram validator. "
    "ALL name matching is CASE-INSENSITIVE — 'login' and 'Login' are identical. "
    "NEVER report missing/extra elements due to capitalisation differences. "
    "DISCONNECTED_ACTOR and ISOLATED_USE_CASE are checked by rule engine — do NOT duplicate them. "
    "Only report scenario-based structural errors you are 100% certain about. "
    "Return valid JSON only — no markdown, no prose."
)


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE SANITIZER — strips Flutter ToolType prefix, filters by diagram type
# so Gemini cannot be confused by cross-diagram shape types
# ─────────────────────────────────────────────────────────────────────────────

# Flutter ToolType enum value → clean readable name for Gemini
_TOOLTYPE_MAP = {
    "classfullshape":       "class",
    "classshape":           "class",
    "generalization":       "generalization_arrow",
    "association":          "association_arrow",
    "aggregation":          "aggregation_arrow",
    "composition":          "composition_arrow",
    "dependency":           "dependency_arrow",
    "dashedarrow":          "dashed_arrow",
    "dashedopenarow":       "dashed_open_arrow",
    "dottedarrow":          "dotted_arrow",
    "excludearrow":         "include_extend_arrow",
    "arrow":                "arrow",
    "straightline":         "line",
    "actor":                "actor",
    "usecase":              "use_case_oval",
    "systemboundary":       "system_boundary",
    "lifeline":             "lifeline",
    "object":               "object_lifeline",
    "activation":           "activation_box",
    "deletion":             "deletion_marker",
    "fragment":             "combined_fragment",
    "selfmessagearrow":     "self_message_arrow",
    "selfmessagedottedarrow": "self_message_dotted_arrow",
    "text":                 "text_label",
    "circle":               "circle",
}

# Shape types that are valid (expected) in each diagram type
_CLASS_VALID_TYPES = {
    "class", "generalization_arrow", "association_arrow",
    "aggregation_arrow", "composition_arrow", "dependency_arrow",
    "dashed_arrow", "dashed_open_arrow", "dotted_arrow",
    "arrow", "line", "text_label",
}
_USECASE_VALID_TYPES = {
    "actor", "use_case_oval", "system_boundary",
    "include_extend_arrow", "generalization_arrow",
    "dashed_arrow", "dashed_open_arrow", "dotted_arrow",
    "arrow", "line", "text_label",
}
_SEQUENCE_VALID_TYPES = {
    "lifeline", "object_lifeline", "actor", "activation_box",
    "deletion_marker", "combined_fragment",
    "arrow", "dashed_arrow", "dotted_arrow",
    "self_message_arrow", "self_message_dotted_arrow",
    "line", "text_label",
}

_VALID_TYPES_BY_DIAGRAM = {
    "class":    _CLASS_VALID_TYPES,
    "usecase":  _USECASE_VALID_TYPES,
    "sequence": _SEQUENCE_VALID_TYPES,
}


def _clean_type(raw_type: str) -> str:
    """Strip 'ToolType.' prefix and return a readable name."""
    t = str(raw_type).strip().lower()
    if "." in t:
        t = t.split(".")[-1]            # 'ToolType.classFullShape' → 'classfullshape'
    t = re.sub(r"[_\-]", "", t)        # drop separators before lookup
    return _TOOLTYPE_MAP.get(t, t)


def _sanitize_shapes(shapes: List[Dict], diagram_type: str) -> List[Dict]:
    """
    Prepare shapes for Gemini:
    1. Clean 'type' field — strip ToolType. prefix, map to readable name.
    2. Remove shapes that don't belong to this diagram type (they confuse Gemini).
    3. Keep only fields Gemini needs; drop internal Flutter state fields.
    """
    valid_types = _VALID_TYPES_BY_DIAGRAM.get(diagram_type)
    cleaned = []

    for s in shapes:
        clean_t = _clean_type(str(s.get("type", "")))

        # Filter out foreign-diagram shapes UNLESS they carry connection info
        if valid_types and clean_t not in valid_types:
            has_conn = any(s.get(f) for f in ("from", "to", "startLifeline", "endLifeline"))
            if not has_conn:
                continue  # skip this shape — it doesn't belong here

        # Build a minimal, clean dict for Gemini
        entry: Dict[str, Any] = {"type": clean_t}
        for field in ("text", "label", "name", "id", "from", "to",
                      "startLifeline", "endLifeline", "lifelineRef",
                      "multiplicity_start", "multiplicity_end",
                      "relationship_label", "attributes", "methods",
                      "position", "endPosition", "size"):
            val = s.get(field)
            if val is not None and str(val).strip().lower() not in ("", "none", "null", "undefined"):
                # classFullShape text = "ClassName\n---attrs---\n..." — preserve full text for class shapes
                # but send only class name for the "name" equivalent
                if field == "text" and "\n" in str(val):
                    # Keep full text so LLM can see attributes/methods
                    entry[field] = val
                    # Also expose the class name separately
                    entry["_class_name"] = str(val).split("\n")[0].strip()
                else:
                    entry[field] = val
        cleaned.append(entry)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK AUTO-FIX BUILDER
# Jab Gemini fixable: true nahi deta, hum error_type se apna fix banate hain
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_fix(error_type: str, element: str, raw_error: dict, diagram_type: str, scenario: str = "") -> dict:
    """
    Gemini ka auto_fix agar incomplete/missing ho — error_type se fallback fix banao.
    Ye ensure karta hai ke common fixable errors hamesha fixable rahein.
    """
    et = error_type.upper()
    desc = str(raw_error.get("description", "")).lower()
    suggestion = str(raw_error.get("suggestion", ""))

    # ── CLASS DIAGRAM ─────────────────────────────────────────────────────────
    if "MISSING_CLASS" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "class", "name": element or "NewClass"}

    if "DUPLICATE_CLASS" in et:
        return {"fixable": True, "action": "merge_shapes", "name": element}

    if "EMPTY_CLASS_NAME" in et:
        return {"fixable": True, "action": "rename_shape", "name": element or "ClassName"}

    if "MISSING_ASSOCIATION_NAME" in et or "MISSING_ASSOCIATION_LABEL" in et:
        # element is usually "ClassA → ClassB" (or "ClassA - ClassB")
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if not (from_el and to_el):
            sep = "\u2192" if "\u2192" in element else ("-" if "-" in element else None)
            if sep:
                parts = element.split(sep)
                if len(parts) >= 2:
                    from_el, to_el = parts[0].strip(), parts[1].strip()
        if from_el and to_el:
            label = _suggest_relationship_label(scenario, from_el, to_el)
            if label:
                return {"fixable": True, "action": "add_label",
                        "from_element": from_el, "to_element": to_el, "name": label}
        return {"fixable": False}

    if "MISSING_RELATIONSHIP" in et or "MISSING_RELATIONSHIP" in et:
        # Try to parse from_element and to_element from description/suggestion
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = _guess_arrow_type(desc + " " + suggestion.lower(), diagram_type)
        if from_el and to_el:
            return {"fixable": True, "action": "add_arrow",
                    "from_element": from_el, "to_element": to_el, "arrow_type": arrow}
        # element might be "ClassA — ClassB"
        if "—" in element or "-" in element:
            parts = element.replace("—", "-").split("-")
            if len(parts) >= 2:
                return {"fixable": True, "action": "add_arrow",
                        "from_element": parts[0].strip(), "to_element": parts[1].strip(),
                        "arrow_type": arrow}
        return {"fixable": False}

    if "WRONG_RELATIONSHIP" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = _guess_arrow_type(desc + " " + suggestion.lower(), diagram_type)
        if from_el and to_el:
            return {"fixable": True, "action": "change_arrow_type",
                    "from_element": from_el, "to_element": to_el, "arrow_type": arrow}
        return {"fixable": False}

    if "MISSING_MULTIPLICITY" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        if from_el and to_el:
            return {"fixable": True, "action": "add_label",
                    "from_element": from_el, "to_element": to_el,
                    "multiplicity_from": "1", "multiplicity_to": "*"}
        return {"fixable": False}

    # ── USE CASE DIAGRAM ──────────────────────────────────────────────────────
    if "MISSING_ACTOR" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "actor", "name": element or "Actor"}

    if "MISSING_USE_CASE" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "use_case_oval", "name": element or "UseCase"}

    if "MISSING_SYSTEM_BOUNDARY" in et:
        return {"fixable": True, "action": "add_boundary", "shape_type": "system_boundary"}

    if "DISCONNECTED_ACTOR" in et:
        return {"fixable": True, "action": "add_arrow",
                "from_element": element, "to_element": "", "arrow_type": "association"}

    if "ISOLATED_USE_CASE" in et:
        return {"fixable": True, "action": "add_arrow",
                "from_element": "", "to_element": element, "arrow_type": "association"}

    if "DUPLICATE_ACTOR" in et or "DUPLICATE_USE_CASE" in et:
        return {"fixable": True, "action": "merge_shapes", "name": element}

    if "MISSING_VERB_IN_USE_CASE" in et:
        # Try to add a verb from suggestion
        name = element or "DoAction"
        if suggestion:
            import re as _re
            m = _re.search(r"'([^']+)'", suggestion)
            if m: name = m.group(1)
        return {"fixable": True, "action": "rename_shape", "name": name}

    # ── SEQUENCE DIAGRAM ──────────────────────────────────────────────────────
    if "MISSING_LIFELINE" in et:
        return {"fixable": True, "action": "add_shape", "shape_type": "lifeline", "name": element or "Participant"}

    if "EMPTY_LIFELINE_NAME" in et:
        return {"fixable": True, "action": "rename_shape", "name": element or "Participant"}

    if "MISSING_MESSAGE" in et or "MISSING_RETURN" in et:
        from_el, to_el = _parse_from_to(desc + " " + suggestion.lower())
        arrow = "dashed_arrow" if "return" in et.lower() or "response" in desc else "arrow"
        if from_el and to_el:
            return {"fixable": True, "action": "add_arrow",
                    "from_element": from_el, "to_element": to_el,
                    "message_label": element or "", "arrow_type": arrow}
        return {"fixable": False}

    return {"fixable": False}


def _parse_from_to(text: str):
    """Try to extract 'from X to Y' or 'X and Y' or quoted names from text."""
    import re as _re
    # Pattern: from 'X' to 'Y'
    m = _re.search(r"from ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? to ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern: between 'X' and 'Y'
    m = _re.search(r"between ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]? and ['\"]?([A-Za-z][A-Za-z0-9_\s]*?)['\"]?(?:\s|$|\.)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Pattern: 'X' class and the 'Y' class
    m = _re.search(r"['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"] class and the ['\"]([A-Za-z][A-Za-z0-9_\s]*?)['\"]", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _guess_arrow_type(text: str, diagram_type: str) -> str:
    """Guess the best arrow type from description/suggestion text."""
    if "composition" in text:   return "composition"
    if "aggregation" in text:   return "aggregation"
    if "generalization" in text or "inherit" in text: return "generalization"
    if "dependency" in text or "depend" in text:      return "dependency"
    if "include" in text:       return "include"
    if "extend" in text:        return "extend"
    return "association"


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED VALIDATOR — 100% deterministic, no LLM
# ─────────────────────────────────────────────────────────────────────────────

def _n(s) -> str:
    """Normalize name: lowercase + strip. Used for ALL comparisons."""
    return str(s).strip().lower()


def _shape_name(s: Dict) -> str:
    """Get the display name of a shape from any possible field."""
    for f in ("text", "label", "name"):
        v = str(s.get(f) or "").strip()
        if v and _n(v) not in ("none", "null", "undefined", ""):
            return v
    return ""


def _arrow_endpoint(s: Dict, side: str) -> str:
    """
    Get the normalized name of an arrow endpoint (from or to side).
    Tries all possible Flutter field names for connection endpoints.
    side = 'from' or 'to'
    """
    if side == "from":
        candidates = ("from", "startShape", "sourceId", "fromActor", "startLifeline")
    else:
        candidates = ("to", "endShape", "targetId", "toUseCase", "endLifeline")
    for f in candidates:
        v = str(s.get(f) or "").strip()
        if v and _n(v) not in ("none", "null", "undefined", ""):
            return _n(v)
    return ""


def _build_connection_map(shapes: List[Dict]) -> Dict[str, set]:
    """
    Build a map: normalized_element_name -> set of normalized names it connects to.
    Works for ANY arrow type by scanning all arrow shapes for from/to fields.
    This is the ground truth for connection checking — no LLM needed.
    """
    connections: Dict[str, set] = {}
    arrow_types = {
        "arrow", "association_arrow", "dashed_arrow", "dashed_open_arrow",
        "dotted_arrow", "include_extend_arrow", "generalization_arrow",
        "line", "straightline",
    }
    for s in shapes:
        if s.get("type") not in arrow_types:
            continue
        frm = _arrow_endpoint(s, "from")
        to  = _arrow_endpoint(s, "to")
        if frm and to:
            connections.setdefault(frm, set()).add(to)
            connections.setdefault(to,  set()).add(frm)
    return connections


# ─────────────────────────────────────────────────────────────────────────────
#  POS tagging helper — used for accurate MISSING_NOUN / MISSING_VERB_IN_USE_CASE
#  / ACTOR_SHOULD_BE_NOUN checks. Uses spaCy (already a project dependency,
#  same model nlp_extractor.py loads) with a lazy, cached load so the model
#  is only loaded once per process. Falls back to a keyword list if spaCy or
#  the model isn't available, so these checks never hard-crash the request.
# ─────────────────────────────────────────────────────────────────────────────
_POS_MODEL = None
_POS_MODEL_TRIED = False

# Much larger fallback verb list than before (used only if spaCy unavailable).
_FALLBACK_VERBS = {
    "add","apply","approve","authenticate","authorize","browse","buy",
    "calculate","cancel","change","check","close","complete","confirm",
    "create","delete","display","download","edit","enter","export",
    "filter","generate","get","handle","import","initiate","insert",
    "launch","list","log","login","logout","make","manage","modify","monitor",
    "notify","open","pay","perform","place","print","process","provide",
    "purchase","read","receive","register","remove","request","reset",
    "retrieve","review","save","search","select","send","set","show",
    "sign","start","submit","track","update","upload","validate","verify",
    "view","withdraw","write",
    # ── previously missing verbs ──
    "book","reserve","checkout","return","renew","borrow","assign","grant",
    "deposit","transfer","order","ship","deliver","schedule","unlock","lock",
    "compare","rate","share","invite","join","leave","subscribe",
    "unsubscribe","publish","post","comment","like","follow","unfollow",
    "block","report","finish","install","uninstall","configure","customize",
    "sync","backup","restore","encrypt","decrypt","scan","fax","email",
    "message","chat","call","record","pause","resume","stop","rewind",
    "forward","upgrade","downgrade","activate","deactivate","enable","disable",
    "redeem","refund","dispute","escalate","assign","allocate","withdraw",
    "deposit","transfer","exchange","swap","merge","split","archive",
    "restore","recover","terminate","suspend","resume","renew","extend",
}

# Common actor/role nouns that must NEVER be flagged as verbs, even if a POS
# tagger mistags a bare standalone word (spaCy has no sentence context for a
# single shape label and sometimes guesses VERB for short unknown words like
# "admin"). This whitelist always wins over the tagger's guess.
_COMMON_ROLE_NOUNS = {
    "admin", "administrator", "customer", "user", "client", "manager",
    "system", "bank", "employee", "staff", "guest", "visitor", "operator",
    "supervisor", "student", "teacher", "instructor", "vendor", "supplier",
    "seller", "buyer", "member", "moderator", "owner", "driver", "rider",
    "passenger", "doctor", "nurse", "patient", "librarian", "cashier",
    "auditor", "agent", "clerk", "receptionist", "technician", "engineer",
    "developer", "analyst", "director", "president", "ceo", "hr",
    "accountant", "warehouse", "merchant", "shopkeeper", "tenant",
    "landlord", "guardian", "parent", "child", "applicant", "recruiter",
    "payment gateway", "third party", "external system",
}


def _is_known_role_noun(name: str) -> bool:
    return (name or "").strip().lower() in _COMMON_ROLE_NOUNS


def _get_pos_model():
    """Lazily load & cache the spaCy model. Returns None if unavailable."""
    global _POS_MODEL, _POS_MODEL_TRIED
    if _POS_MODEL_TRIED:
        return _POS_MODEL
    _POS_MODEL_TRIED = True
    try:
        import spacy  # type: ignore
        _POS_MODEL = spacy.load("en_core_web_sm")
    except Exception as e:
        _log.warning("POS model unavailable, falling back to verb keyword list: %s", e)
        _POS_MODEL = None
    return _POS_MODEL


def _pos_analyze(name: str) -> Dict[str, bool]:
    """
    Analyze a shape name and return {'has_verb': bool, 'has_noun': bool}.
    Uses real POS tagging when spaCy is available; otherwise falls back to
    a keyword-list heuristic (still checks BOTH verb presence and noun
    presence, unlike the old first-word-only check).
    """
    name = (name or "").strip()
    if not name:
        return {"has_verb": False, "has_noun": False}

    # Safety override: known role/entity nouns are never verbs, regardless
    # of what the POS tagger guesses for a bare standalone word.
    if _is_known_role_noun(name):
        return {"has_verb": False, "has_noun": True}

    nlp = _get_pos_model()
    if nlp is not None:
        doc = nlp(name)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
        has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in doc)
        return {"has_verb": has_verb, "has_noun": has_noun}

    # ── Fallback: keyword-list heuristic ──
    # spaCy unavailable — use a conservative convention-based heuristic instead
    # of pure list-membership, since words like "order"/"report" can be BOTH
    # noun and verb and a naive list lookup would misclassify "Manage Order".
    words = [w.lower() for w in name.split()]
    if not words:
        return {"has_verb": False, "has_noun": False}
    has_verb = words[0] in _FALLBACK_VERBS
    if len(words) > 1:
        # Standard UML naming convention is "Verb Noun" (e.g. "Manage Order") —
        # trust that a second word is the object/noun.
        has_noun = True
    else:
        has_noun = words[0] not in _FALLBACK_VERBS
    return {"has_verb": has_verb, "has_noun": has_noun}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY-BASED CONNECTION DETECTION — ported from usecase_validator.py
# (proven spatial-detection logic: position + size + line endpoints touching
# a shape's bounding box). Used as a SECOND, independent way to detect
# connections, alongside the existing name-field (from/to) based check.
# A shape counts as "connected" if EITHER method finds a connection.
# ─────────────────────────────────────────────────────────────────────────────

# Max px distance from a line endpoint to a shape's bbox to count as "touching"
_HIT_RADIUS = 40.0

# Line/arrow types (already-cleaned names, post _clean_type) that can carry a
# spatial connection in a use case diagram.
_USECASE_LINE_TYPES = {
    "arrow", "line", "dashed_arrow", "dashed_open_arrow", "dotted_arrow",
    "include_extend_arrow", "generalization_arrow", "association_arrow",
}


def _geo_pos(shape: Dict) -> Optional[tuple]:
    """Shape's (x, y) position — top-left for shapes, start-point for lines."""
    p = shape.get("position")
    if not isinstance(p, dict):
        return None
    try:
        return float(p.get("dx", 0)), float(p.get("dy", 0))
    except (TypeError, ValueError):
        return None


def _geo_end_abs(shape: Dict) -> Optional[tuple]:
    """Absolute end point of a line: position + endPosition (relative offset)."""
    ep = shape.get("endPosition")
    if not isinstance(ep, dict):
        return None
    pos = _geo_pos(shape)
    if pos is None:
        return None
    try:
        return pos[0] + float(ep.get("dx", 0)), pos[1] + float(ep.get("dy", 0))
    except (TypeError, ValueError):
        return None


def _geo_size(shape: Dict) -> tuple:
    s = shape.get("size")
    if not isinstance(s, dict):
        return 80.0, 60.0
    try:
        return float(s.get("width", 80)), float(s.get("height", 60))
    except (TypeError, ValueError):
        return 80.0, 60.0


def _geo_bbox(shape: Dict) -> Optional[tuple]:
    pos = _geo_pos(shape)
    if pos is None:
        return None
    w, h = _geo_size(shape)
    x, y = pos
    return x, y, x + w, y + h


def _pt_near_bbox(px: float, py: float, bbox: tuple, radius: float = _HIT_RADIUS) -> bool:
    """True if point (px, py) is within `radius` of the axis-aligned bbox."""
    x1, y1, x2, y2 = bbox
    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))
    return math.hypot(px - cx, py - cy) <= radius


def _line_touches(line: Dict, shape: Dict, radius: float = _HIT_RADIUS) -> bool:
    """
    True if either endpoint of `line` is within `radius` of `shape`'s bbox.
    Returns False (not True) when position/size data is missing, so this
    check never produces false positives when geometry simply isn't present
    in the payload — the name-based check still covers that case.
    """
    bbox = _geo_bbox(shape)
    if bbox is None:
        return False

    start = _geo_pos(line)
    if start is not None and _pt_near_bbox(start[0], start[1], bbox, radius):
        return True

    end = _geo_end_abs(line)
    if end is not None and _pt_near_bbox(end[0], end[1], bbox, radius):
        return True

    return False


def _rule_check_usecase(shapes: List[Dict]) -> List[Dict]:
    """
    Deterministic rule-based checks for use case diagrams.
    Handles: empty names, duplicates, disconnected actors/use-cases, capitalisation,
    and noun/verb correctness (via spaCy POS tagging, with keyword-list fallback).
    Does NOT need LLM — all checks are from shape data directly.
    """
    errors = []

    # Collect actors and use cases
    actors   = {}  # norm_name -> original_name
    use_cases = {} # norm_name -> original_name

    for s in shapes:
        t    = s.get("type", "")
        name = _shape_name(s)
        n    = _n(name)

        if t == "actor":
            if not n:
                errors.append({
                    "error_type": "UNLABELLED_ACTOR", "severity": "ERROR",
                    "element": "(unnamed actor)",
                    "description": "An actor has no name.",
                    "suggestion": "Give this actor a meaningful name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Actor"},
                })
            elif n in actors:
                errors.append({
                    "error_type": "DUPLICATE_ACTOR", "severity": "ERROR",
                    "element": name,
                    "description": f"Actor '{name}' appears more than once.",
                    "suggestion": f"Remove the duplicate '{name}' actor.",
                    "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
                })
            else:
                actors[n] = name

        elif t == "use_case_oval":
            if not n:
                errors.append({
                    "error_type": "UNLABELLED_USE_CASE", "severity": "ERROR",
                    "element": "(unnamed use case)",
                    "description": "A use case has no label.",
                    "suggestion": "Give this use case a descriptive action name.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Use Case"},
                })
            elif n in use_cases:
                errors.append({
                    "error_type": "DUPLICATE_USE_CASE", "severity": "ERROR",
                    "element": name,
                    "description": f"Use case '{name}' appears more than once.",
                    "suggestion": f"Remove the duplicate '{name}' use case.",
                    "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
                })
            else:
                use_cases[n] = name

                # ── Use case naming format check: STRICT "Verb + Noun" only ──
                # A use case oval's label must be exactly a verb followed by a
                # noun/object (e.g. "Manage Order", "Place Order"). Any name
                # that does not satisfy BOTH parts is reported as ONE error.
                _pos = _pos_analyze(name)
                if not (_pos["has_verb"] and _pos["has_noun"]):
                    errors.append({
                        "error_type": "MISSING_VERB_IN_USE_CASE", "severity": "WARNING",
                        "element": name,
                        "description": f"Use case '{name}' must follow the 'Verb + Noun' naming format only (e.g. 'Manage Order', 'Place Order') — a verb followed by a noun, nothing more.",
                        "suggestion": f"Rename '{name}' to a verb + noun, e.g. 'Manage {name}' or 'Process {name}'.",
                        "auto_fix": {"fixable": True, "action": "rename_shape", "name": f"Manage {name}"},
                    })

    # ── Blank/Empty Use Case check ──────────────────────────────────────────
    # Check every use case oval — if it has no text → EMPTY_USE_CASE_LABEL error
    for s in shapes:
        t = s.get("type", "").lower()
        if "usecase" in t or "use_case" in t or "oval" in t or "ellipse" in t:
            label = str(s.get("text", "") or s.get("label", "") or s.get("name", "") or "").strip()
            if not label or label.lower() in ("", "none", "null", "undefined", "use case", "usecase"):
                errors.append({
                    "error_type": "EMPTY_USE_CASE_LABEL", "severity": "ERROR",
                    "element": "Unlabelled use case",
                    "description": "A use case oval has no label or an empty name.",
                    "suggestion": "Give this use case a meaningful name e.g. 'Place Order', 'Browse Products'.",
                    "auto_fix": {"fixable": False},
                })

    # ── Connection checking (name-based + geometry-based) ──────────────────
    # Method 1: name-based — arrow shapes that carry explicit from/to (or
    # equivalent) fields naming the connected elements.
    connected = set()
    for s in shapes:
        t = s.get("type", "")
        if any(k in t for k in ("arrow", "line", "association", "connector")):
            for key in ("from", "to", "startShape", "endShape", "source", "target"):
                val = _n(str(s.get(key) or ""))
                if val:
                    connected.add(val)

    # Method 2: geometry-based (ported from usecase_validator.py) — a shape is
    # "connected" if any line's endpoint (position / position+endPosition)
    # lands within _HIT_RADIUS of that shape's bbox (position + size). This
    # catches diagrams where lines are drawn visually touching a shape but
    # don't carry explicit from/to name fields — the exact false-positive
    # ("connected use case wrongly flagged as isolated") this was fixing.
    # A shape only needs to satisfy ONE of the two methods to count as connected.
    line_shapes = [s for s in shapes if _n(str(s.get("type", ""))) in _USECASE_LINE_TYPES]

    if line_shapes:
        for norm_name, orig_name in actors.items():
            if norm_name in connected:
                continue
            shape = next(
                (s for s in shapes if s.get("type") == "actor"
                 and _n(_shape_name(s)) == norm_name),
                None,
            )
            if shape and any(_line_touches(ln, shape) for ln in line_shapes):
                connected.add(norm_name)

        for norm_name, orig_name in use_cases.items():
            if norm_name in connected:
                continue
            shape = next(
                (s for s in shapes if s.get("type") == "use_case_oval"
                 and _n(_shape_name(s)) == norm_name),
                None,
            )
            if shape and any(_line_touches(ln, shape) for ln in line_shapes):
                connected.add(norm_name)

    # DISCONNECTED_ACTOR: actor with no arrow connecting to any use case
    for norm_name, orig_name in actors.items():
        if norm_name not in connected:
            errors.append({
                "error_type": "DISCONNECTED_ACTOR", "severity": "ERROR",
                "element": orig_name,
                "description": f"Actor '{orig_name}' is not connected to any use case.",
                "suggestion": f"Draw an association line from '{orig_name}' to at least one use case.",
                "auto_fix": {"fixable": False},
            })

    # ISOLATED_USE_CASE: use case oval with no arrow connecting to any actor
    for norm_name, orig_name in use_cases.items():
        if norm_name not in connected:
            errors.append({
                "error_type": "ISOLATED_USE_CASE", "severity": "ERROR",
                "element": orig_name,
                "description": f"Use case '{orig_name}' is not connected to any actor.",
                "suggestion": f"Connect '{orig_name}' to at least one actor.",
                "auto_fix": {"fixable": False},
            })

    # Check: actor names should be nouns/roles, not verbs/actions (POS-based)
    for n, orig in actors.items():
        _pos = _pos_analyze(orig)
        if _pos["has_verb"] and not _pos["has_noun"]:
            errors.append({
                "error_type": "ACTOR_SHOULD_BE_NOUN", "severity": "WARNING",
                "element": orig,
                "description": f"Actor '{orig}' appears to be a verb/action. Actors should represent roles or entities (nouns), not actions.",
                "suggestion": f"Rename '{orig}' to a role or entity name, e.g. 'Customer', 'Admin', or 'System'.",
                "auto_fix": {"fixable": False},
            })

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# ASSOCIATION-NAME SUGGESTION — for class diagram MISSING_ASSOCIATION_NAME.
# Scans the scenario text for the sentence connecting two class names and
# suggests the actual verb/phrase that should be used as the label.
# ─────────────────────────────────────────────────────────────────────────────

_REL_LINK_PHRASES = [
    "works for", "worked for", "placed by", "created by", "owned by",
    "assigned to", "reports to", "belongs to", "consists of", "made up of",
    "has a", "has an", "have a", "have an", "is a", "is an", "are a", "are an",
]

_REL_MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
_REL_IRREGULAR_PRESENT = {
    "have": "has", "be": "is", "do": "does", "go": "goes",
    "has": "has", "is": "is", "does": "does", "goes": "goes",
}


def _to_present_tense(verb: str) -> str:
    """Normalize a verb to lowercase, 3rd-person-singular present tense."""
    v = verb.lower().strip()
    if not v:
        return v
    if v in _REL_IRREGULAR_PRESENT:
        return _REL_IRREGULAR_PRESENT[v]
    if v in _REL_MODAL_VERBS:
        return v
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    if v.endswith("e"):
        return v + "s"          # place -> places, manage -> manages
    return v + "s"               # work -> works, contain -> contains


def _name_variants(name: str) -> set:
    n = name.lower().strip()
    variants = {n}
    if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
        variants.add(n[:-1] + "ies")
    else:
        variants.add(n + "s")
    if n.endswith("s"):
        variants.add(n[:-1])
    return variants


def _suggest_relationship_label(scenario: str, name_a: str, name_b: str) -> Optional[str]:
    """
    Find the sentence in `scenario` that mentions both `name_a` and `name_b`,
    and suggest the verb/phrase connecting them as a lowercase, present-tense
    association-name label.

    Priority:
      1. Known linking phrases ("has a", "works for", "placed by", ...) —
         used verbatim as the label if present.
      2. spaCy-tagged main verb between the two names (modal verbs like
         "can"/"may" are skipped in favour of the actual action verb, and
         the verb is normalized to present tense, e.g. "can place" -> "places").
      3. Keyword-list fallback verb if spaCy isn't available.

    Returns None if no relevant sentence / verb can be found — caller should
    fall back to a generic suggestion in that case.
    """
    if not scenario or not name_a or not name_b:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", scenario.strip())
    va, vb = _name_variants(name_a), _name_variants(name_b)

    for sent in sentences:
        s_low = sent.lower()
        if not (any(x in s_low for x in va) and any(x in s_low for x in vb)):
            continue

        # 1) Known has-a / is-a / role phrases — use directly as the label
        for phrase in _REL_LINK_PHRASES:
            if phrase in s_low:
                return phrase

        # 2) spaCy main-verb extraction (skip modals/aux, prefer the last
        #    substantive verb — i.e. the more specific one when 2 verbs exist)
        nlp = _get_pos_model()
        if nlp is not None:
            doc = nlp(sent)
            verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
            main_verbs = [t for t in verbs if t.lemma_.lower() not in _REL_MODAL_VERBS]
            chosen = main_verbs[-1] if main_verbs else (verbs[-1] if verbs else None)
            if chosen is not None:
                return _to_present_tense(chosen.lemma_)

        # 3) Fallback: scan words for a known action verb from the keyword list
        words = re.findall(r"[a-zA-Z']+", s_low)
        fallback_hits = [w for w in words if w in _FALLBACK_VERBS and w not in _REL_MODAL_VERBS]
        if fallback_hits:
            return _to_present_tense(fallback_hits[-1])

    return None


def _rule_check_class(shapes: List[Dict], scenario: str = "") -> List[Dict]:
    """Deterministic rule checks for class diagrams."""
    errors = []
    class_names = {}  # norm -> original

    for s in shapes:
        if s.get("type") != "class":
            continue
        name = _shape_name(s)
        n    = _n(name)
        if not n or n in ("class 1", "classname", "class"):
            errors.append({
                "error_type": "EMPTY_CLASS_NAME", "severity": "ERROR",
                "element": name or "(unnamed)",
                "description": "Class has no name or has a placeholder name.",
                "suggestion": "Give this class a meaningful name.",
                "auto_fix": {"fixable": True, "action": "rename_shape", "name": "NewClass"},
            })
        elif n in class_names:
            errors.append({
                "error_type": "DUPLICATE_CLASS", "severity": "ERROR",
                "element": name,
                "description": f"Class '{name}' appears more than once.",
                "suggestion": f"Remove the duplicate '{name}' class.",
                "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
            })
        else:
            class_names[n] = name
            # Capitalisation: WARNING only
            if name and name[0].islower():
                correct = name[0].upper() + name[1:]
                errors.append({
                    "error_type": "WRONG_CLASS_CAPITALISATION", "severity": "WARNING",
                    "element": name,
                    "description": f"Class '{name}' should start with an uppercase letter.",
                    "suggestion": f"Rename '{name}' to '{correct}'.",
                    "auto_fix": {"fixable": True, "action": "rename_shape", "name": correct},
                })
    # ── Multiplicity + Association-Name checks (previously missing entirely) ──
    # Every association/aggregation/composition arrow must have valid
    # multiplicity on BOTH ends and a name label at its midpoint.
    # Generalization/dependency/realization arrows are excluded — they never
    # carry multiplicity or a name by UML convention.
    _VALID_MULT_RE = re.compile(r"^(\d+|\*)(\.\.(\d+|\*))?$")
    _NO_MULT_KEYWORDS = ("generalization", "inheritance", "extend", "realization",
                         "dependency", "depend", "include")
    _MULT_ARROW_KEYWORDS = ("association", "aggregation", "composition")

    for s in shapes:
        t = str(s.get("type", "")).lower()
        if not any(k in t for k in _MULT_ARROW_KEYWORDS):
            continue
        if any(k in t for k in _NO_MULT_KEYWORDS):
            continue

        frm_raw = str(s.get("from") or "")
        to_raw  = str(s.get("to")   or "")
        frm_n, to_n = _n(frm_raw), _n(to_raw)
        if not frm_n or not to_n:
            continue  # can't identify endpoints — skip rather than guess

        frm_disp = class_names.get(frm_n, frm_raw)
        to_disp  = class_names.get(to_n,  to_raw)

        m_start = str(s.get("multiplicity_start") or "").strip()
        m_end   = str(s.get("multiplicity_end")   or "").strip()

        # ── MISSING_MULTIPLICITY ──
        if not m_start and not m_end:
            errors.append({
                "error_type": "MISSING_MULTIPLICITY", "severity": "ERROR",
                "element": f"{frm_disp} \u2192 {to_disp}",
                "description": f"Relationship '{frm_disp}' \u2192 '{to_disp}' is missing multiplicity on both ends.",
                "suggestion": "Add multiplicity labels (e.g., '1', '0..*', '1..*') on both sides of the relationship.",
                "auto_fix": {"fixable": True, "action": "update_multiplicity",
                             "from_element": frm_disp, "to_element": to_disp,
                             "multiplicity_from": "1", "multiplicity_to": "*"},
            })
        elif not m_start:
            errors.append({
                "error_type": "MISSING_MULTIPLICITY", "severity": "ERROR",
                "element": f"{frm_disp} \u2192 {to_disp}",
                "description": f"Relationship '{frm_disp}' \u2192 '{to_disp}' is missing multiplicity on the '{frm_disp}' side.",
                "suggestion": f"Add a multiplicity label (e.g., '1', '0..*') on the '{frm_disp}' side.",
                "auto_fix": {"fixable": True, "action": "update_multiplicity",
                             "from_element": frm_disp, "to_element": to_disp,
                             "multiplicity_from": "1", "multiplicity_to": m_end},
            })
        elif not m_end:
            errors.append({
                "error_type": "MISSING_MULTIPLICITY", "severity": "ERROR",
                "element": f"{frm_disp} \u2192 {to_disp}",
                "description": f"Relationship '{frm_disp}' \u2192 '{to_disp}' is missing multiplicity on the '{to_disp}' side.",
                "suggestion": f"Add a multiplicity label (e.g., '1', '0..*') on the '{to_disp}' side.",
                "auto_fix": {"fixable": True, "action": "update_multiplicity",
                             "from_element": frm_disp, "to_element": to_disp,
                             "multiplicity_from": m_start, "multiplicity_to": "*"},
            })
        else:
            # ── INVALID_MULTIPLICITY — both present but format is malformed ──
            bad_start = not _VALID_MULT_RE.match(m_start)
            bad_end   = not _VALID_MULT_RE.match(m_end)
            if bad_start or bad_end:
                bad_side = frm_disp if bad_start else to_disp
                bad_val  = m_start if bad_start else m_end
                errors.append({
                    "error_type": "INVALID_MULTIPLICITY", "severity": "ERROR",
                    "element": f"{frm_disp} \u2192 {to_disp}",
                    "description": (f"Multiplicity '{bad_val}' on the '{bad_side}' side of "
                                     f"'{frm_disp}' \u2192 '{to_disp}' is not a valid UML multiplicity."),
                    "suggestion": "Use a valid format such as '1', '0..1', '*', '0..*', or '1..*'.",
                    "auto_fix": {"fixable": False},
                })

        # ── MISSING_ASSOCIATION_NAME — unconditional, independent of multiplicity ──
        label = str(s.get("relationship_label") or "").strip()
        if not label:
            text = str(s.get("text") or "").strip()
            if "|" in text:
                parts = text.split("|")
                if len(parts) >= 2:
                    label = parts[1].strip()
        if not label:
            suggested = _suggest_relationship_label(scenario, frm_disp, to_disp)
            if suggested:
                errors.append({
                    "error_type": "MISSING_ASSOCIATION_NAME", "severity": "ERROR",
                    "element": f"{frm_disp} \u2192 {to_disp}",
                    "description": "Association Name is required",
                    "suggestion": (
                        f"Add '{suggested}' as the label at the midpoint of the line "
                        f"between '{frm_disp}' and '{to_disp}' (based on the scenario)."
                    ),
                    "auto_fix": {"fixable": True, "action": "add_label",
                                 "from_element": frm_disp, "to_element": to_disp,
                                 "name": suggested},
                })
            else:
                errors.append({
                    "error_type": "MISSING_ASSOCIATION_NAME", "severity": "ERROR",
                    "element": f"{frm_disp} \u2192 {to_disp}",
                    "description": "Association Name is required",
                    "suggestion": (
                        f"Add a name label at the midpoint of the line between "
                        f"'{frm_disp}' and '{to_disp}', e.g. a verb describing "
                        f"their relationship (e.g. 'manages', 'has a')."
                    ),
                    "auto_fix": {"fixable": False},
                })

    return errors


def _rule_check_sequence(shapes: List[Dict]) -> List[Dict]:
    """Deterministic rule checks for sequence diagrams."""
    errors = []
    lifelines = {}

    for s in shapes:
        t = s.get("type", "")
        if t not in ("lifeline", "object_lifeline", "actor"):
            continue
        name = _shape_name(s)
        n    = _n(name)
        if not n:
            errors.append({
                "error_type": "UNLABELLED_LIFELINE", "severity": "ERROR",
                "element": "(unnamed)",
                "description": "A lifeline has no name.",
                "suggestion": "Give this lifeline a meaningful name.",
                "auto_fix": {"fixable": True, "action": "rename_shape", "name": "Participant"},
            })
        elif n in lifelines:
            errors.append({
                "error_type": "DUPLICATE_LIFELINE", "severity": "WARNING",
                "element": name,
                "description": f"Lifeline '{name}' appears more than once.",
                "suggestion": f"Remove the duplicate '{name}' lifeline.",
                "auto_fix": {"fixable": True, "action": "merge_shapes", "name": name},
            })
        else:
            lifelines[n] = name

    for s in shapes:
        if s.get("type") not in ("arrow", "dashed_arrow"):
            continue
        label = str(s.get("label") or s.get("text") or "").strip()
        if _n(label) in ("", "none", "null", "undefined"):
            errors.append({
                "error_type": "UNLABELLED_ARROW", "severity": "WARNING",
                "element": "(unlabelled arrow)",
                "description": "A message arrow has no label.",
                "suggestion": "Add a message name to this arrow.",
                "auto_fix": {"fixable": True, "action": "rename_shape", "name": "message"},
            })
    return errors


def _run_rule_checks(shapes: List[Dict], diagram_type: str, scenario: str = "") -> List[Dict]:
    dt = diagram_type.lower()
    if "usecase" in dt or "use_case" in dt or "use case" in dt:
        return _rule_check_usecase(shapes)
    elif "class" in dt:
        return _rule_check_class(shapes, scenario)
    elif "sequence" in dt:
        return _rule_check_sequence(shapes)
    return []


def _existing_relationships(shapes: List[Dict]) -> set:
    """
    Return a set of frozensets {norm_from, norm_to} for all drawn arrows.
    Used to filter hallucinated MISSING_RELATIONSHIP / MISSING_MULTIPLICITY errors.
    """
    rels = set()
    # Accept both raw cleaned types and original type strings
    arrow_keywords = {
        "arrow", "association", "dashed", "dotted", "include",
        "generalization", "composition", "aggregation", "dependency", "line",
    }
    for s in shapes:
        t = str(s.get("type", "")).lower().replace("_", "").replace("-", "")
        is_arrow = any(kw.replace("_","") in t for kw in arrow_keywords)
        if not is_arrow:
            continue
        frm = _n(str(s.get("from") or s.get("startLifeline") or ""))
        to  = _n(str(s.get("to")   or s.get("endLifeline")   or ""))
        if frm and to:
            rels.add(frozenset({frm, to}))
    return rels


def _existing_multiplicities(shapes: List[Dict]) -> set:
    """
    Return frozensets {norm_from, norm_to} for arrows that ALREADY HAVE multiplicity set.
    An arrow has multiplicity if multiplicity_start OR multiplicity_end is non-empty,
    OR if the text field has pipe-separated format 'startMult|label|endMult'.
    """
    has_mult = set()
    # Arrow types that can carry multiplicity (association, aggregation, composition)
    mult_arrow_keywords = {"association", "aggregation", "composition", "arrow", "line"}
    for s in shapes:
        t = str(s.get("type", "")).lower().replace("_", "").replace("-", "")
        is_mult_arrow = any(kw.replace("_","") in t for kw in mult_arrow_keywords)
        if not is_mult_arrow:
            continue
        frm = _n(str(s.get("from") or ""))
        to  = _n(str(s.get("to")   or ""))
        if not (frm and to):
            continue

        m_start = str(s.get("multiplicity_start") or "").strip()
        m_end   = str(s.get("multiplicity_end")   or "").strip()
        text    = str(s.get("text") or "").strip()

        # Check pipe-format text: "1|label|*"
        has_pipe_mult = False
        if "|" in text:
            parts = text.split("|")
            if len(parts) >= 2 and (parts[0].strip() or parts[-1].strip()):
                has_pipe_mult = True

        if (m_start and m_start.lower() not in ("none", "null", "")) or \
           (m_end   and m_end.lower()   not in ("none", "null", "")) or \
           has_pipe_mult:
            has_mult.add(frozenset({frm, to}))

    return has_mult


def _existing_names(shapes: List[Dict]) -> set:
    """All normalized names present in the diagram — for hallucination filtering."""
    names = set()
    for s in shapes:
        n = _n(_shape_name(s))
        if n:
            names.add(n)
        # Also extract class name from _class_name field
        cn = _n(str(s.get("_class_name") or ""))
        if cn:
            names.add(cn)
    return names


_STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "and", "or", "in", "on", "at",
    "by", "with", "is", "are", "be", "will", "can", "should", "must",
    "their", "his", "her", "its", "system", "user",
}


def _stem(w: str) -> str:
    """Very light stemmer so 'tracks'/'tracking'/'tracked' all match 'track'."""
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _scenario_mentions(scenario: str, element: str) -> bool:
    """
    Fuzzy, deterministic check: does the scenario text actually contain
    (a close variant of) this element name? Used to stop the LLM from
    reporting EXTRA_USE_CASE / EXTRA_ACTOR for things that ARE in the
    scenario just written with different wording/case/tense.
    """
    if not scenario or not element:
        return False
    scenario_words = {_stem(w) for w in re.findall(r"[a-zA-Z]+", scenario)}
    elem_words = [w for w in re.findall(r"[a-zA-Z]+", element) if _n(w) not in _STOPWORDS]
    if not elem_words:
        return False
    matched = sum(1 for w in elem_words if _stem(w) in scenario_words)
    # Majority of the significant words must appear in the scenario text.
    return matched >= max(1, math.ceil(len(elem_words) / 2))


def _merge_results(rule_errors: List[Dict], llm_errors: List[Dict],
                   llm_warnings: List[Dict], llm_info: List[Dict],
                   clean_shapes: List[Dict], llm_score: int, llm_summary: str,
                   ignored_errors: Optional[List[str]] = None,
                   scenario: str = "") -> Dict:
    """
    Merge rule-based errors with LLM errors.
    - Rule errors: always trusted (deterministic).
    - LLM errors: filtered to remove hallucinations and capitalisation noise.
    - ignored_errors: list of error fingerprints that user has dismissed — never re-report them.
    """
    SKIP_FROM_LLM = {
        "wrong_class_capitalisation", "wrong_capitalisation",
        "wrong_actor_capitalisation", "wrong_use_case_capitalisation",
        "duplicate_actor", "duplicate_use_case", "duplicate_class",
        "unlabelled_actor", "unlabelled_use_case", "unlabelled_lifeline",
        "empty_class_name",
        "disconnected_actor", "isolated_use_case",  # handled by rule engine above
        "missing_multiplicity", "invalid_multiplicity",  # handled by rule engine above
        "missing_association_name",  # handled by rule engine above
        "missing_noun",  # handled by rule engine above
        # Self-referential / actor-connection hallucinations
        "self_referential_relationship", "incorrect_self_referential_relationship",
        "self_referential", "incorrect_self_reference",
        "incorrect_relationship", "incorrect_actor_relationship",
        "actor_self_relationship", "invalid_relationship",
        # wrong_multiplicity is now ENABLED — removed from skip list
    }

    existing      = _existing_names(clean_shapes)
    existing_rels = _existing_relationships(clean_shapes)
    existing_mult = _existing_multiplicities(clean_shapes)
    ignored_set   = set(ignored_errors or [])

    def _error_fingerprint(e: Dict) -> str:
        """Unique key for an error — used for ignored_errors matching."""
        et   = _n(str(e.get("error_type", "")))   # lowercase
        elem = _n(str(e.get("element", "")))
        fix  = e.get("auto_fix") or {}
        frm  = _n(str(fix.get("from_element", "")))
        to   = _n(str(fix.get("to_element",   "")))
        return f"{et}|{elem}|{frm}|{to}"

    def keep_llm(e):
        et   = _n(str(e.get("error_type", "")))   # lowercase
        elem = _n(str(e.get("element", "")))
        fix  = e.get("auto_fix") or {}
        frm  = _n(str(fix.get("from_element", "")))
        to   = _n(str(fix.get("to_element",   "")))

        # Drop if rule layer already handles this error type
        if et in SKIP_FROM_LLM:
            return False

        # Drop if user has ignored this error previously
        if _error_fingerprint(e) in ignored_set:
            return False

        # Drop any relationship error where both endpoints are actors in the diagram
        # (LLM hallucinates self-referential or actor-actor errors from spatial arrow lines)
        if frm and to and frm == to:
            # Self-referential: from and to are the same element — always a hallucination for actors
            actor_names = {_n(_shape_name(s)) for s in clean_shapes if s.get("type") == "actor"}
            if frm in actor_names:
                return False

        if frm and to and frm != to:
            actor_names = {_n(_shape_name(s)) for s in clean_shapes if s.get("type") == "actor"}
            if frm in actor_names and to in actor_names:
                return False  # actor-to-actor relationship — always a hallucination

        # Drop WRONG_SYSTEM_BOUNDARY_NAME if the names differ only by case
        if et == "wrong_system_boundary_name":
            desc = str(e.get("description", ""))
            # Extract both names from description: "named 'X' but scenario calls it 'Y'"
            import re as _re2
            names_found = _re2.findall(r"'([^']+)'", desc)
            if len(names_found) >= 2:
                if _n(names_found[0]) == _n(names_found[1]):
                    return False  # same name, just different case — not a real error

        # Drop EXTRA_USE_CASE / EXTRA_ACTOR (and MISSING_USE_CASE / MISSING_ACTOR)
        # if a deterministic fuzzy word-match shows the element name IS actually
        # present in the scenario text — this is a pure LLM hallucination guard,
        # independent of the model's own (sometimes wrong) judgement.
        if et in ("extra_use_case", "extra_actor") and elem:
            if _scenario_mentions(scenario, e.get("element", "")):
                return False

        if et in ("missing_use_case", "missing_actor") and elem:
            if _scenario_mentions(scenario, e.get("element", "")) and elem in existing:
                return False

        # Drop if LLM says element is MISSING but it exists in shapes (hallucination)
        if "missing" in et and elem and elem in existing:
            # Allow MISSING_MULTIPLICITY / MISSING_RELATIONSHIP even if element name matches
            # ONLY if the relationship itself is NOT already in the diagram
            if et in ("missing_multiplicity", "missing_relationship",
                      "missing_association_label", "wrong_multiplicity"):
                if frm and to:
                    pair = frozenset({frm, to})
                    if et == "missing_multiplicity":
                        # Only report if multiplicity is genuinely absent
                        if pair in existing_mult:
                            return False  # multiplicity already exists — false positive
                    elif et in ("missing_relationship", "missing_association_label"):
                        # Only report if the relationship does NOT exist at all
                        if pair in existing_rels:
                            return False  # relationship already drawn — false positive
                    # wrong_multiplicity: keep it (it's a value correction, not missing)
                    return True
            return False  # generic MISSING_X when element exists = hallucination

        # Drop MISSING_MULTIPLICITY if pair already has multiplicity (no from/to in fix)
        if et == "missing_multiplicity" and frm and to:
            if frozenset({frm, to}) in existing_mult:
                return False

        # Drop MISSING_RELATIONSHIP / MISSING_ASSOCIATION_LABEL if relationship already drawn
        if et in ("missing_relationship", "missing_association_label") and frm and to:
            if frozenset({frm, to}) in existing_rels:
                return False

        return True

    filtered_errors   = [e for e in llm_errors   if keep_llm(e)]
    filtered_warnings = [e for e in llm_warnings if keep_llm(e)]
    filtered_info     = [e for e in llm_info     if keep_llm(e)]

    # Spelling mistake dedup: if a SPELLING_MISTAKE is reported for an element,
    # suppress any MISSING_* or EXTRA_* error for that same element — only ONE error allowed.
    _spelled_elements = set()
    for _e in filtered_errors + filtered_warnings + filtered_info:
        if _n(str(_e.get("error_type", ""))) == "spelling_mistake":
            _spelled_elements.add(_n(str(_e.get("element", ""))))

    def _not_dup_spelling(e):
        et   = _n(str(e.get("error_type", "")))
        elem = _n(str(e.get("element", "")))
        if et == "spelling_mistake":
            return True
        if elem in _spelled_elements and ("missing" in et or "extra" in et):
            return False
        return True

    filtered_errors   = [e for e in filtered_errors   if _not_dup_spelling(e)]
    filtered_warnings = [e for e in filtered_warnings if _not_dup_spelling(e)]
    filtered_info     = [e for e in filtered_info     if _not_dup_spelling(e)]

    # Also filter rule errors against ignored set
    def keep_rule(r):
        return _error_fingerprint(r) not in ignored_set

    rule_errors_filtered = [r for r in rule_errors if keep_rule(r)]

    def to_item(r):
        return {
            "error_type":  r["error_type"],
            "severity":    r.get("severity", "ERROR"),
            "element":     r.get("element", ""),
            "description": r.get("description", ""),
            "suggestion":  r.get("suggestion", ""),
            "auto_fix":    r.get("auto_fix", {"fixable": False}),
        }

    rule_e = [to_item(r) for r in rule_errors_filtered if r.get("severity") == "ERROR"]
    rule_w = [to_item(r) for r in rule_errors_filtered if r.get("severity") == "WARNING"]
    rule_i = [to_item(r) for r in rule_errors_filtered if r.get("severity") == "INFO"]

    final_errors   = rule_e + filtered_errors
    final_warnings = rule_w + filtered_warnings
    final_info     = rule_i + filtered_info
    all_items      = final_errors + final_warnings + final_info

    fixable = sum(1 for i in all_items if i.get("auto_fix", {}).get("fixable"))

    if not all_items:
        return {
            "is_valid": True, "score": 100,
            "summary": "Diagram is correct",
            "errors": [], "warnings": [], "info": [],
            "total_issues": 0, "fixable_count": 0, "source": "openai+rules",
        }

    score = max(0, 100 - len(final_errors) * 15 - len(final_warnings) * 5)
    return {
        "is_valid":      len(final_errors) == 0,
        "score":         score,
        "summary":       llm_summary or f"{len(final_errors)} error(s), {len(final_warnings)} warning(s)",
        "errors":        final_errors,
        "warnings":      final_warnings,
        "info":          final_info,
        "total_issues":  len(all_items),
        "fixable_count": fixable,
        "source":        "openai+rules",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — alag diagram type ke liye alag prompt
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_class(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Class Diagram validator using SEMANTIC analysis.

## TASK
This is a CLASS DIAGRAM. Validate it using ONLY class diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS — violating these makes your response wrong:
- Do NOT check for actors, stick figures, system boundaries, use cases, lifelines, or messages.
- Do NOT report MISSING_ACTOR, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE, MISSING_LIFELINE.
- Do NOT apply use case or sequence diagram rules of any kind.
- ONLY apply the rules listed below.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ FLUTTER SHAPES — CRITICAL:

### Arrow/Relationship shapes:
- `"type"`: e.g. `"ToolType.association"`, `"ToolType.aggregation"`, `"ToolType.composition"`, `"ToolType.generalization"`, `"ToolType.dependency"`
- `"multiplicity_start"`: Multiplicity at source end e.g. `"1"`, `"0..*"`, `"1..*"`
- `"multiplicity_end"`: Multiplicity at target end e.g. `"*"`, `"1"`, `"0..1"`
- `"relationship_label"`: Label on the arrow e.g. `"manages"`, `"contains"`
- `"from"` / `"to"`: Source and target class names
- `"text"`: Raw text in format `"startMult|label|endMult"` e.g. `"1|manages|*"`

### Class shapes:
- `"type"`: `"ToolType.classShape"` or `"ToolType.classFullShape"`
- `"text"`: Class name (top section)

### MULTIPLICITY VALIDATION RULES:
- If `multiplicity_start` OR `multiplicity_end` field exists with a non-empty value → multiplicity EXISTS.
- If BOTH are null/empty on an association/aggregation/composition arrow → report MISSING_MULTIPLICITY.
- Check `"text"` field too — format is `"startMult|label|endMult"`. If the text has pipe-separated values, those are the multiplicities.
- If multiplicity is present but WRONG vs scenario → report WRONG_MULTIPLICITY with correct expected values.

### RELATIONSHIP TYPE VALIDATION:
- Read `"type"` field: `composition`, `aggregation`, `generalization`, `association`, `dependency`
- If scenario says "is a" / "inherits" / "type of" / "kind of" → expect GENERALIZATION
- If scenario says "consists of" / "composed of" / "cannot exist without" → expect COMPOSITION
- If scenario says "contains" / "collection of" / "holds" / "is made up of" → expect AGGREGATION
- If scenario says "has" / "uses" / "is related to" / "is associated with" → expect ASSOCIATION
- Wrong type drawn → report WRONG_RELATIONSHIP_TYPE

## SEMANTIC ANALYSIS — READ THIS CAREFULLY:
You must use SEMANTIC reasoning, not just keyword matching. Different students describe the same correct diagram in different ways. A diagram is valid if its OVERALL LOGIC matches the scenario's intent, even if exact wording differs.

Examples of semantically equivalent descriptions:
- "Customer places Order" and "Order is placed by Customer" → same relationship
- "Bank manages Accounts" and "Bank has multiple Accounts" → same aggregation
- Multiplicity "1 to many" = "1..*" = "one to many" — all mean the same

## RELATIONSHIP RULES — CRITICAL:
- ONLY report MISSING_RELATIONSHIP if the scenario EXPLICITLY states a relationship between two specific classes using trigger words (has, contains, inherits, etc.).
- If the scenario does NOT describe any relationship between two classes, do NOT invent one.
- Do NOT report relationships that are merely implied or logically reasonable — only what is written.
- A diagram with classes but NO relationships drawn is valid if the scenario does not describe relationships.

## CLASS NAME CASE RULES — CRITICAL:
- If a class name exists in the diagram but with wrong capitalisation (e.g. "customer" instead of "Customer"):
  → Report WRONG_CLASS_CAPITALISATION with suggestion to "Capitalise the first letter: rename 'customer' to 'Customer'."
  → Do NOT report it as EXTRA_CLASS or say to remove it.
  → Do NOT report it as MISSING_CLASS for the correctly-capitalised version.
- Class names MUST start with an uppercase letter in UML. Lowercase first letter = capitalisation error only.

## SPELLING MISTAKE RULES — CRITICAL:
- SPELLING_MISTAKE applies ONLY when a class IS ACTUALLY DRAWN on the canvas and its name is a NEAR-IDENTICAL misspelling of a scenario class name — differing by only 1-2 letters, a swapped letter, or one missing/extra letter, and clearly the same word (e.g. "Custmer" vs "Customer", "Odrr" vs "Order").
- Do NOT treat two names as a "spelling mistake" match just because they are both nouns in the same domain or start with the same letter. If the drawn name and the scenario name do not share almost all the same letters in the same order, they are NOT a spelling match.
- If a scenario class has NO drawn class whose name is a near-identical match by the rule above — including when it was never drawn at all — you MUST report MISSING_CLASS for it. NEVER invent a SPELLING_MISTAKE against some unrelated existing class name just to explain the gap.
- When SPELLING_MISTAKE genuinely applies (per the strict criteria above):
  → Report EXACTLY ONE error of type SPELLING_MISTAKE.
  → element: the misspelled name as drawn.
  → description: "Class name 'X' appears to be a misspelling of 'Y' from the scenario."
  → suggestion: "Try changing 'X' to 'Y'."
  → auto_fix: fixable: true, action: rename_shape, name: <correct spelling from scenario>
  → Do NOT ALSO report it as MISSING_CLASS for the correctly-spelled version.
  → Do NOT ALSO report it as EXTRA_CLASS for the misspelled version.
  → ONE error only — either SPELLING_MISTAKE or MISSING_CLASS, never both for the same element.

## MISSING LABEL RULES:
- If a relationship arrow exists between ClassA and ClassB, and the scenario explicitly names a label for that relationship (e.g. "manages", "contains", "employs"), but the drawn arrow has no label → report MISSING_ASSOCIATION_LABEL.
- Description should say: "The relationship between 'ClassA' and 'ClassB' should have the label 'X' as described in the scenario."
- Suggestion: "Add the label 'X' to the arrow between 'ClassA' and 'ClassB'."
- If the scenario does NOT name a specific label, do NOT report missing label — labels are optional.

## RULES TO CHECK (class diagram ONLY)
1. MISSING_CLASS              — Important nouns in scenario must be classes. Only explicitly named entities.
2. EXTRA_CLASS                — Class in diagram not mentioned in scenario (warning).
3. WRONG_CLASS_CAPITALISATION — Class name exists but starts with lowercase — suggest capitalising, never suggest removing.
4. WRONG_RELATIONSHIP_TYPE    — Wrong arrow type vs scenario (association/aggregation/composition/generalization).
5. MISSING_RELATIONSHIP       — Relationship EXPLICITLY described in scenario but not drawn.
6. MISSING_MULTIPLICITY       — Association/aggregation/composition arrow drawn but both multiplicity fields are empty.
7. WRONG_MULTIPLICITY         — Multiplicity present but value is wrong vs scenario.
8. MISSING_ASSOCIATION_LABEL  — Scenario names a label for a relationship but it is not on the drawn arrow.
9. WRONG_INHERITANCE_DIRECTION — Inheritance arrow reversed (child should point TO parent).
10. DUPLICATE_CLASS           — Same class name appears twice.
11. CIRCULAR_INHERITANCE      — A inherits B and B inherits A.
12. EMPTY_CLASS_NAME          — Class has no name or placeholder like "Class 1".
13. SELF_ASSOCIATION          — Class connected to itself (warn unless scenario says so).
14. SPELLING_MISTAKE          — A class name closely resembles a scenario class name but is misspelled (e.g. "Custmer" instead of "Customer"). ONE error only — do NOT also report MISSING_CLASS or EXTRA_CLASS for the same element.
15. WRONG_ASSOCIATION_LABEL   — An association/aggregation/composition arrow HAS a label, but that label does not match what the scenario describes for that relationship. ALWAYS include the correct word/phrase taken directly from the scenario in the "suggestion" field.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
For each error, provide an "auto_fix" object:
- "action": one of "add_shape", "delete_shape", "rename_shape", "add_arrow", "delete_arrow", "change_arrow_type", "add_label", "merge_shapes"
- "shape_type": (for add_shape) "class"
- "name": new name or label
- "from_element": source class name
- "to_element": target class name
- "arrow_type": "association", "aggregation", "composition", "generalization", "dependency"
- "multiplicity_from": multiplicity at source e.g. "1"
- "multiplicity_to": multiplicity at target e.g. "*"
- "fixable": true/false

AUTO-FIX RULES:
- MISSING_CLASS → fixable: true, action: add_shape, shape_type: class, name: <missing class>
- WRONG_CLASS_CAPITALISATION → fixable: true, action: rename_shape, name: <correctly capitalised name>
- DUPLICATE_CLASS → fixable: true, action: merge_shapes, name: <class name to keep>
- EMPTY_CLASS_NAME → fixable: true, action: rename_shape, name: <correct name from scenario>
- WRONG_RELATIONSHIP_TYPE → fixable: true, action: change_arrow_type, from_element, to_element, arrow_type: <correct type>
- MISSING_RELATIONSHIP → fixable: true, action: add_arrow, from_element, to_element, arrow_type: <correct type>
- MISSING_MULTIPLICITY → fixable: true, action: add_label, from_element, to_element, multiplicity_from, multiplicity_to
- WRONG_MULTIPLICITY → fixable: true, action: add_label, from_element, to_element, multiplicity_from: <correct>, multiplicity_to: <correct>
- MISSING_ASSOCIATION_LABEL → fixable: true, action: add_label, from_element, to_element, name: <label>
- WRONG_ASSOCIATION_LABEL → fixable: true, action: add_label, from_element, to_element, name: <correct label from scenario>
- WRONG_INHERITANCE_DIRECTION → fixable: false
- CIRCULAR_INHERITANCE → fixable: false
- EXTRA_CLASS → fixable: false
- SELF_ASSOCIATION → fixable: false
- SPELLING_MISTAKE → fixable: true, action: rename_shape, name: <correct spelling from scenario>

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ClassName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "class",
        "name": "MissingClassName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT RULES — MUST FOLLOW:
- Results must be DETERMINISTIC — same diagram + scenario must always give same errors.
- Only report issues you are CONFIDENT about. Do NOT invent errors.

### MISSING_CLASS:
- STRICT: Only report for nouns EXPLICITLY written as class names in the scenario.
- STRICT: Method names like "submitOrder()" are METHODS, never classes.
- STRICT: Attribute names like "orderId", "price" are ATTRIBUTES, never classes.
- STRICT: If a class name in the diagram is a close misspelling (near-identical, 1-2 letters off) of a scenario class → report SPELLING_MISTAKE ONLY, NOT MISSING_CLASS.
- STRICT: For EVERY class explicitly named in the scenario, go through the diagram's class list one by one. If none of the drawn class names is a near-identical spelling match, that scenario class was NOT drawn at all → you MUST report MISSING_CLASS for it. Do NOT skip this just because some other, unrelated class exists in the diagram.
- STRICT: A completely different or unrelated class name already present in the diagram does NOT excuse a scenario class from being reported as MISSING_CLASS.

### MISSING_RELATIONSHIP:
- STRICT: Only report if scenario EXPLICITLY uses trigger words: has, contains, inherits, is a type of, consists of, is composed of, manages, holds, etc.
- STRICT: Do NOT invent relationships just because two classes exist in the same scenario.
- STRICT: If scenario says nothing about the relationship between two classes, no MISSING_RELATIONSHIP error.

### CLASS CAPITALISATION:
- STRICT: If a class "customer" exists and scenario mentions "Customer" → WRONG_CLASS_CAPITALISATION only. Do NOT say remove it. Do NOT say add "Customer" as a new class.

### MISSING_MULTIPLICITY:
- STRICT: Only report if an association/aggregation/composition arrow is drawn AND BOTH `multiplicity_start` AND `multiplicity_end` fields are empty/null/missing.
- STRICT: Check the `"text"` field for pipe format `"startMult|label|endMult"` (e.g. `"1|manages|*"`) — if EITHER the first or last segment is non-empty, multiplicity EXISTS and must NOT be reported as missing.
- STRICT: If `multiplicity_start` OR `multiplicity_end` has any non-empty value → multiplicity IS present → do NOT report MISSING_MULTIPLICITY.
- STRICT: NEVER report MISSING_MULTIPLICITY for an arrow that has multiplicity_start or multiplicity_end set, even if only one side has a value.
- STRICT: Do NOT suggest a specific multiplicity value (like "1 to *") unless the scenario explicitly states it.

### WRONG_MULTIPLICITY (WARNING) ← ALWAYS CHECK:
- Check EVERY association, aggregation, and composition arrow that HAS multiplicity values.
- Look at multiplicity_start and multiplicity_end fields in each arrow shape.
- Compare drawn multiplicity against what the scenario describes.
- Examples of wrong multiplicity:
  * Scenario: "one customer places many orders" → must be "1" on Customer side, "*" on Order side.
    If drawn as "*" on both sides → WRONG_MULTIPLICITY.
  * Scenario: "a student enrolls in many courses, a course has many students" → must be "* " on both sides.
    If drawn as "1" on either side → WRONG_MULTIPLICITY.
- Report element as "ClassA → ClassB" using from/to fields of the arrow.
- description: "Multiplicity 'X' on ClassA end should be 'Y' based on scenario."
- auto_fix: fixable: true, action: update_multiplicity, from_element, to_element, multiplicity_from: "correct", multiplicity_to: "correct"
- STRICT: Only report if scenario CLEARLY states cardinality (one-to-many, many-to-many etc).
- STRICT: Do NOT guess — only report when you are certain.

### MISSING_ATTRIBUTE / MISSING_METHOD:
- STRICT: ONLY report MISSING_ATTRIBUTE if the scenario EXPLICITLY lists specific attributes for a class (e.g. "Customer has attributes: name, email, phone").
- STRICT: ONLY report MISSING_METHOD if the scenario EXPLICITLY lists specific methods for a class (e.g. "Order has methods: placeOrder(), cancelOrder()").
- STRICT: If the scenario does NOT explicitly list attributes/methods for a class → do NOT report any MISSING_ATTRIBUTE or MISSING_METHOD for that class, EVEN IF attributes/methods are commonly expected.
- STRICT: If the scenario lists attributes/methods AND the user has drawn them → NO error.
- STRICT: If the scenario does NOT list attributes/methods → NO error, regardless of what user draws.
- STRICT: Extra attributes/methods that are NOT in the scenario but user adds → NO error (user can add more than scenario requires).

### MISSING_ASSOCIATION_LABEL (WARNING) ← CHECK THIS:
- Check EVERY association, aggregation, and composition arrow in the diagram.
- Look at the relationship_label field AND the text field of each arrow.
- If the scenario uses a VERB to describe the relationship (e.g. "Bank manages Customer", "Teacher teaches Student") AND the drawn arrow has NO label → report MISSING_ASSOCIATION_LABEL as WARNING.
- element: "ClassA → ClassB"
- description: "Association between ClassA and ClassB is missing a label."
- suggestion: "Add label 'verb' to the arrow between ClassA and ClassB."
- auto_fix: fixable: true, action: add_label, from_element: "ClassA", to_element: "ClassB", label: "<verb from scenario>"
- STRICT: Only report when scenario explicitly describes the relationship with a verb.
- STRICT: If scenario does not name the relationship → do NOT report.

### WRONG_ASSOCIATION_LABEL (WARNING) ← CHECK THIS:
- Check EVERY association, aggregation, and composition arrow that HAS a label.
- If the drawn label does NOT match what the scenario describes → report WRONG_ASSOCIATION_LABEL as WARNING.
- Example: scenario says "Bank manages Customer" but arrow label says "owns" → WRONG_ASSOCIATION_LABEL.
- Example: scenario says "Teacher teaches Student" but arrow label says "trains" → WRONG_ASSOCIATION_LABEL.
- description: "Label 'X' on arrow ClassA → ClassB should be 'Y' based on scenario."
- suggestion: ALWAYS state the exact correct word/phrase from the scenario, e.g. "Change label 'X' to 'Y' as described in the scenario."
- auto_fix: fixable: true, action: add_label, from_element: "ClassA", to_element: "ClassB", label: "<correct label>"
- STRICT: Only report when scenario explicitly names the relationship AND drawn label is clearly different.
- STRICT: Minor variations (e.g. "manage" vs "manages") are acceptable — do NOT report.

### General:
- STRICT: When in doubt about ANY error, SKIP it.
- STRICT: Use semantic understanding — a diagram correct in logic is correct even if wording differs.
"""


def _prompt_usecase(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Use Case Diagram validator.

## TASK
This is a USE CASE DIAGRAM. Validate it using ONLY use case diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS:
- Do NOT check for class boxes, attributes, methods, lifelines, or activation bars.
- Do NOT report MISSING_CLASS, MISSING_ATTRIBUTE, MISSING_METHOD, MISSING_LIFELINE.
- Do NOT apply class diagram or sequence diagram rules of any kind.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## RULES TO CHECK (use case diagram ONLY)
YOUR JOB: Only check scenario-based completeness. Connection checking and duplicate/empty name
checking is done by a separate rule-based system — do NOT repeat those checks.

1. MISSING_ACTOR      — An actor explicitly named in the scenario is completely absent from the diagram.
                        Match case-insensitively. If "Customer" is in scenario and "customer" is in diagram → NOT missing.
2. MISSING_USE_CASE   — A use case explicitly named in the scenario is completely absent from the diagram.
                        Match case-insensitively. If "Login" is in scenario and "login" is in diagram → NOT missing.
3. EXTRA_ACTOR        — Actor in diagram not mentioned in scenario at all (WARNING only).
4. EXTRA_USE_CASE     — Use case in diagram not mentioned in scenario at all (INFO only).
5. MISSING_SYSTEM_BOUNDARY — No system boundary rectangle exists in the diagram at all.
6. WRONG_SYSTEM_BOUNDARY_NAME — Boundary exists but label does not match scenario system name.
7. WRONG_RELATIONSHIP — include/extend/generalization used incorrectly per scenario.
8. SPELLING_MISTAKE   — An actor or use case name closely resembles a scenario name but is misspelled (e.g. "Custmer" instead of "Customer"). ONE error only — do NOT also report MISSING_ACTOR/MISSING_USE_CASE or EXTRA_ACTOR/EXTRA_USE_CASE for the same element.

DO NOT CHECK AND DO NOT REPORT:
- DISCONNECTED_ACTOR (handled by rule system)
- ISOLATED_USE_CASE (handled by rule system)
- UNLABELLED_USE_CASE (handled by rule system)
- WRONG_ACTOR_NAME / actor-noun checks (handled by rule system — do NOT report this yourself, it is unreliable from an image/shape read and duplicates the rule engine)
- DUPLICATE_ACTOR / DUPLICATE_USE_CASE (handled by rule system)
- UNLABELLED_ACTOR / UNLABELLED_USE_CASE (handled by rule system)
- Any capitalisation errors — case differences are NEVER errors
- SELF_REFERENTIAL_RELATIONSHIP or any self-referential error for actors — NEVER report these
- INCORRECT_RELATIONSHIP between two actors — actor-to-actor relationships are NEVER an error in use case diagrams
- INCORRECT_RELATIONSHIP or INVALID_RELATIONSHIP of any kind — these error types do not exist in use case diagrams
- WRONG_MULTIPLICITY — multiplicity does not apply to use case diagrams

## SEMANTIC MATCHING — CRITICAL, AVOID FALSE POSITIVES
- Before reporting MISSING_USE_CASE, MISSING_ACTOR, EXTRA_USE_CASE, or EXTRA_ACTOR, compare meaning, not just exact words.
- "Manage Products" ≈ "Add Product" / "Edit Product" / "Browse Products" if the scenario describes that capability in different wording — treat as a MATCH, not missing/extra.
- Synonyms count as matches: "Place Order" ≈ "Order Product", "Make Payment" ≈ "Pay", "Track Deliveries" ≈ "Track Delivery" ≈ "View Delivery Status".
- Singular/plural differences are NEVER errors: "Product" == "Products".
- Only report MISSING_USE_CASE/MISSING_ACTOR if the scenario clearly describes a capability/role that has NO reasonable equivalent anywhere in the diagram.
- Only report EXTRA_USE_CASE/EXTRA_ACTOR if the diagram element has NO reasonable connection to anything described in the scenario.
- If you are not confident an element is truly missing or truly extra, DO NOT report it — under-reporting is better than a false positive.

## CASE-INSENSITIVE MATCHING — ABSOLUTE RULE
- ALL name matching is 100% CASE-INSENSITIVE.
- "Login" == "login" == "LOGIN" == "LogIn" — they are ALL the same.
- NEVER report any element as missing, extra, or wrong because of capitalisation.
- NEVER output any capitalisation-related error type at all.

## SYSTEM BOUNDARY NAME RULES — CRITICAL
- If a system boundary rectangle EXISTS with a label that does not match the scenario system name:
  → Report WRONG_SYSTEM_BOUNDARY_NAME.
  → Description: "System boundary is named 'X' but scenario calls it 'Y'."
  → Suggestion: "Change the system boundary name from 'X' to 'Y'." (never say "add a new boundary")
- If system boundary does NOT exist at all → report MISSING_SYSTEM_BOUNDARY.
- If system boundary exists with correct name → no error.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
For each error, also provide an "auto_fix" object describing exactly how to fix it programmatically.
auto_fix fields:
- "action": one of "add_shape", "delete_shape", "rename_shape", "add_arrow", "add_boundary", "merge_shapes"
- "shape_type": (for add_shape) one of "actor", "use_case_oval", "system_boundary"
- "name": new name or label to set
- "from_element": source element name (for arrows)
- "to_element": target element name (for arrows)
- "arrow_type": one of "association", "include", "extend", "generalization"
- "fixable": true if this can be auto-fixed, false if user must fix manually

AUTO-FIX RULES:
- MISSING_ACTOR → fixable: true, action: add_shape, shape_type: actor, name: <actor name>
- MISSING_USE_CASE → fixable: true, action: add_shape, shape_type: use_case_oval, name: <use case name>
- MISSING_SYSTEM_BOUNDARY → fixable: true, action: add_boundary, shape_type: system_boundary
- WRONG_SYSTEM_BOUNDARY_NAME → fixable: true, action: rename_shape, name: <correct system name from scenario>
- DISCONNECTED_ACTOR → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- ISOLATED_USE_CASE → fixable: true, action: add_arrow, from_element: <actor name>, to_element: <use case name>, arrow_type: association
- MISSING_VERB_IN_USE_CASE → fixable: true, action: rename_shape, name: <corrected name with verb>
- WRONG_CAPITALISATION → fixable: true, action: rename_shape, name: <correctly capitalised name>
- DUPLICATE_ACTOR → fixable: true, action: merge_shapes, name: <actor name to keep>
- DUPLICATE_USE_CASE → fixable: true, action: merge_shapes, name: <use case name to keep>
- EXTRA_ACTOR / EXTRA_USE_CASE → fixable: false (user may have added intentionally)
- WRONG_RELATIONSHIP → fixable: false (user must review relationship semantics)
- ACTOR_NOT_IN_BOUNDARY → fixable: false (requires layout restructuring)
- SPELLING_MISTAKE → fixable: true, action: rename_shape, name: <correct spelling from scenario>

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "ActorOrUseCaseName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "actor",
        "name": "MissingActorName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT VALIDATION RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about. Do NOT invent errors.

### MISSING_ACTOR hallucination prevention:
- STRICT: Only report MISSING_ACTOR for persons/systems that are EXPLICITLY written in the scenario.
- STRICT: Do NOT invent actors that are implied but not written.
- STRICT: A single actor is enough if scenario mentions only one person/system.
- STRICT: If an actor name in the diagram is a close misspelling of a scenario actor → report SPELLING_MISTAKE ONLY, NOT MISSING_ACTOR.

### MISSING_USE_CASE hallucination prevention:
- STRICT: Only report MISSING_USE_CASE for actions that are EXPLICITLY written in the scenario.
- STRICT: Do NOT split one use case into multiple — if scenario says "login", do NOT also require "validate credentials", "check password" etc.
- STRICT: Do NOT invent sub-use-cases that are not written in the scenario.
- STRICT: Matching is CASE-INSENSITIVE — "Login" and "login" are the same use case.
- STRICT: If a use case name in the diagram is a close misspelling of a scenario use case → report SPELLING_MISTAKE ONLY, NOT MISSING_USE_CASE.

### DISCONNECTED_ACTOR hallucination prevention:
- STRICT: A line touching the actor's body, torso, or any limb area = CONNECTED. Do NOT report DISCONNECTED_ACTOR for such actors.
- STRICT: Only report DISCONNECTED_ACTOR if there is genuinely no line anywhere near the actor shape.

### WRONG_SYSTEM_BOUNDARY_NAME rules:
- STRICT: If the boundary EXISTS with a wrong name → say "Change the name from 'X' to 'Y'". Never say "add a new boundary".
- STRICT: Do NOT report both MISSING_SYSTEM_BOUNDARY and WRONG_SYSTEM_BOUNDARY_NAME for the same diagram.

### WRONG_RELATIONSHIP hallucination prevention:
- STRICT: Only report WRONG_RELATIONSHIP if you are 100% certain the relationship type is wrong.
- STRICT: Do NOT flag association arrows as wrong unless scenario explicitly requires include/extend.

### General:
- STRICT: When in doubt about ANY error, SKIP it — do not report it.
- STRICT: Do NOT assume actors or use cases that are implied but not written in scenario.
- STRICT: Results must be DETERMINISTIC — same diagram + scenario must always produce the same errors.
"""


def _prompt_sequence(scenario: str, shapes: List[Dict]) -> str:
    return f"""You are an expert UML Sequence Diagram validator using SEMANTIC analysis.

## TASK
This is a SEQUENCE DIAGRAM. Validate it using ONLY sequence diagram rules. Return ONLY valid JSON.

## ABSOLUTE RESTRICTIONS:
- Do NOT check for class boxes, system boundaries, or use cases.
- Do NOT report MISSING_CLASS, MISSING_SYSTEM_BOUNDARY, MISSING_USE_CASE.
- Do NOT apply class diagram or use case diagram rules of any kind.

## SCENARIO
{scenario}

## DIAGRAM SHAPES
{json.dumps(shapes, indent=2)}

## HOW TO READ SEQUENCE DIAGRAM SHAPES:

### Lifeline / Object shapes:
- Types: `lifeline`, `object_lifeline`, `object` → these are participant boxes at the top.
- `"text"` or `"label"` field = the participant's name.
- An `object` shape is an OBJECT (instance), not an anonymous lifeline. If it has a name, use that name — do NOT report it as unnamed.
- If an object shape has an empty label → report UNLABELLED_OBJECT (not UNLABELLED_LIFELINE).

### Actor shapes:
- Type: `actor` → stick figure at top, represents a human participant.
- `"text"` or `"label"` = actor's name.

### Message arrows:
- Types: `arrow`, `dashed_arrow`, `dotted_arrow` → horizontal arrows between lifelines.
- `"from"` / `"to"` = source and target lifeline names.
- `"label"` or `"text"` = the message/operation name.
- `"type": "self_message_arrow"` or `"selfmessagearrow"` → self-message (same lifeline sends to itself). This is VALID in UML — do NOT report it as an error unless it has no label.

### Deletion / Destroy markers:
- Types: `deletion_marker`, `deletion`, `destroy`, `x`, `cross` → X mark at bottom of a lifeline.
- `"lifeline"`, `"on"`, or `"label"` field links deletion to a lifeline.
- If these shapes exist in the diagram → deletion symbols ARE present for those lifelines.
- Report MISSING_DELETION_SYMBOL only for lifelines that have NO associated deletion shape.

### Activation boxes:
- Types: `activation_box`, `activation` → thin rectangle on a lifeline's dashed line.

### Combined fragments:
- Types: `combined_fragment`, `fragment` → alt/opt/loop/par boxes.

## SEMANTIC ANALYSIS — READ THIS CAREFULLY:
You must use SEMANTIC reasoning. Different students may label messages differently but mean the same thing. A diagram is correct if its overall interaction logic matches the scenario's intent.

Examples:
- "validateUser()" and "validate credentials" both represent the same login validation step → semantically the same.
- "loginResponse" and "authToken returned" both represent the login result → same.
- Message order is correct if the LOGICAL sequence matches the scenario, even if exact wording differs.

## MESSAGE ORDER RULES — CRITICAL:
- Only report WRONG_MESSAGE_ORDER if the sequence in the diagram is CLEARLY and DEFINITIVELY wrong compared to the scenario's described flow.
- Do NOT report WRONG_MESSAGE_ORDER if the order could be interpreted as valid or if the scenario doesn't specify strict ordering.
- When in doubt about message order → DO NOT report it. Skip the error.
- Semantic equivalents count as correct order (a "validate" step before a "response" step is standard flow).

## SELF-MESSAGE RULES:
- `self_message_arrow` shapes are VALID UML — they represent a method call on the same object.
- Do NOT report SELF_MESSAGE as an error unless the scenario specifically says self-messages are wrong.
- Only report INVALID_SELF_MESSAGE if a self-message arrow has no label at all.

## DELETION SYMBOL RULES — CRITICAL:
- Look for shapes of type `deletion_marker`, `deletion`, `destroy`, `x`, `cross`, or any X-shaped marker.
- If deletion shapes are present in the diagram shapes list → deletion symbols EXIST. Do NOT report them as missing.
- Only report MISSING_DELETION_SYMBOL for specific lifelines that have no associated deletion shape.
- If the diagram uses a simple format (no deletion markers) → report NO_DELETION_SYMBOLS as INFO only (not ERROR).

## OBJECT vs LIFELINE NAMING:
- An `object_lifeline` or `object` shape with a name IS a named participant — do NOT report it as having no name.
- If an object shape's label is empty → report: "Object has no name. Add a name to the object box."
- Do NOT say "lifeline has no name" when the shape type is `object` or `object_lifeline` — say "object has no name".

## RULES TO CHECK (sequence diagram ONLY)
1. MISSING_LIFELINE         — Every participant/object in scenario must have a lifeline or object box.
2. EXTRA_LIFELINE           — Lifeline not in scenario (warning).
3. MISSING_MESSAGE          — Important interaction in scenario not shown as a message arrow. Use semantic matching.
4. WRONG_MESSAGE_ORDER      — Messages are in CLEARLY wrong order vs scenario. Only report when certain.
5. MISSING_RETURN           — A call message has no return/response when scenario explicitly expects one.
6. INVALID_MESSAGE_SOURCE   — Message arrow starts from non-existent lifeline.
7. INVALID_MESSAGE_TARGET   — Message arrow ends at non-existent lifeline.
8. ISOLATED_LIFELINE        — Lifeline sends/receives no messages.
9. UNLABELLED_LIFELINE      — Lifeline box has no label.
10. UNLABELLED_OBJECT       — Object box has no label (use "object" terminology, not "lifeline").
11. MISSING_ACTIVATION      — Lifeline that receives messages has no activation box.
12. MISSING_DELETION_SYMBOL — Specific lifeline has no X/destroy marker at its end.
13. UNLABELLED_ARROW        — Message arrow has no label/name.
14. MISSING_ALT_FRAGMENT    — Conditional logic in scenario not shown as alt/opt fragment.
15. SPELLING_MISTAKE        — A lifeline/object/actor name closely resembles a scenario name but is misspelled. ONE error only — do NOT also report MISSING_LIFELINE or EXTRA_LIFELINE for the same element.

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
auto_fix fields:
- "action": one of "add_shape", "rename_shape", "add_arrow", "merge_shapes"
- "shape_type": one of "lifeline", "activation_box", "combined_fragment", "deletion_marker"
- "name": label to set
- "from_element": source lifeline name
- "to_element": target lifeline name
- "message_label": label for the arrow
- "arrow_type": "arrow" (solid call) or "dashed_arrow" (return/response)
- "fixable": true/false

AUTO-FIX RULES:
- MISSING_LIFELINE → fixable: true, action: add_shape, shape_type: lifeline, name: <participant name>
- UNLABELLED_LIFELINE / UNLABELLED_OBJECT → fixable: true, action: rename_shape, name: <correct name>
- UNLABELLED_ARROW → fixable: true, action: rename_shape, name: <message label from scenario>
- MISSING_MESSAGE → fixable: true, action: add_arrow, from_element, to_element, message_label, arrow_type: arrow
- MISSING_RETURN → fixable: true, action: add_arrow, from_element: <receiver>, to_element: <sender>, message_label: <return label>, arrow_type: dashed_arrow
- MISSING_DELETION_SYMBOL → fixable: true, action: add_shape, shape_type: deletion_marker, name: <lifeline name>
- WRONG_MESSAGE_ORDER → fixable: false
- ISOLATED_LIFELINE → fixable: false
- EXTRA_LIFELINE → fixable: false
- INVALID_MESSAGE_SOURCE / INVALID_MESSAGE_TARGET → fixable: false
- MISSING_ACTIVATION → fixable: false
- MISSING_ALT_FRAGMENT → fixable: false
- SPELLING_MISTAKE → fixable: true, action: rename_shape, name: <correct spelling from scenario>

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "LifelineOrMessageName",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "lifeline",
        "name": "MissingLifelineName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "e.g. 2 errors, 1 warning"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT RULES — MUST FOLLOW:
- Results must be DETERMINISTIC — same diagram + scenario must always give the same errors.
- Only report issues you are CONFIDENT about. Do NOT invent errors.

### MISSING_LIFELINE:
- STRICT: Only report for participants EXPLICITLY named in the scenario.
- STRICT: An `object` shape with a name counts as a valid lifeline — do NOT report it missing.
- STRICT: If a lifeline/object name in the diagram is a close misspelling of a scenario participant → report SPELLING_MISTAKE ONLY, NOT MISSING_LIFELINE.

### MISSING_MESSAGE:
- STRICT: Only for interactions EXPLICITLY described in the scenario.
- STRICT: Use semantic matching — "validateUser()" matches "validate credentials" — same interaction.
- STRICT: Do NOT invent intermediate messages not in scenario.

### WRONG_MESSAGE_ORDER:
- STRICT: Only report if the order is CLEARLY wrong (e.g. response comes before request).
- STRICT: If the order is ambiguous or could be valid → DO NOT report it. Skip.
- STRICT: Semantic equivalents count as correct — do not flag stylistic differences as wrong order.

### SELF-MESSAGE:
- STRICT: `self_message_arrow` is VALID UML. Do NOT report it as an error.
- STRICT: Only flag if it has absolutely no label.

### DELETION SYMBOLS:
- STRICT: If deletion_marker/deletion/destroy/x shapes are in the diagram → they ARE present. Do NOT report them missing.
- STRICT: Only report MISSING_DELETION_SYMBOL for lifelines that have genuinely no X marker.

### OBJECT vs LIFELINE:
- STRICT: An `object` or `object_lifeline` shape is an object box, not a "lifeline". Use correct terminology.
- STRICT: If an object has a name, it is named — do NOT say it has no name.

### MISSING_RETURN:
- STRICT: Only report if scenario explicitly says there should be a response.
- STRICT: Many valid sequence diagrams have no return arrows.

### General:
- STRICT: When in doubt about ANY error, SKIP it.
- STRICT: Use semantic understanding — diagrams correct in logic are correct.
"""


def _build_prompt(diagram_type: str, scenario: str, shapes: List[Dict]) -> str:
    dt = diagram_type.lower()
    if "class" in dt:
        return _prompt_class(scenario, shapes)
    elif "usecase" in dt or "use_case" in dt or "use case" in dt:
        return _prompt_usecase(scenario, shapes)
    elif "sequence" in dt:
        return _prompt_sequence(scenario, shapes)
    else:
        return _prompt_class(scenario, shapes)  # default fallback


# ─────────────────────────────────────────────────────────────────────────────
# HTTP call
# ─────────────────────────────────────────────────────────────────────────────

def _call_model(prompt: str, api_key: str, model: str) -> Optional[Dict]:
    url = f"{_OPENAI_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1500,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            _log.warning("Model %s: rate limit — waiting %ss then retry...", model, _RETRY_WAIT)
            time.sleep(_RETRY_WAIT)
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as e2:
                _log.warning("Model %s: retry failed: %s", model, e2)
                return None
        elif e.code == 404:
            _log.warning("Model %s: not found, trying next...", model)
            return None
        else:
            _log.error("Model %s: HTTP %s: %s", model, e.code, err_body[:500])
            return None
    except Exception as e:
        _log.warning("Model %s: failed: %s", model, e)
        return None

    try:
        outer = json.loads(body)
        text  = outer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        _log.warning("Model %s: parse error: %s", model, e)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$",        "", text.rstrip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _log.warning("Model %s: JSON error: %s", model, e)
        return None


def make_error_fingerprint(error_type: str, element: str = "",
                           from_element: str = "", to_element: str = "") -> str:
    """
    Generate a stable fingerprint for an error that can be stored by the Flutter app
    and passed back as ignored_errors on future validate_with_openai() calls.

    Usage (Flutter side):
        When user taps "Ignore" on an error, call:
            fingerprint = make_error_fingerprint(
                error["error_type"],
                error.get("element",""),
                error.get("auto_fix",{}).get("from_element",""),
                error.get("auto_fix",{}).get("to_element",""),
            )
        Store fingerprints in a list and pass as ignored_errors next time you validate.
    """
    def _n_local(s): return str(s).strip().lower()
    return f"{_n_local(error_type)}|{_n_local(element)}|{_n_local(from_element)}|{_n_local(to_element)}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_openai(
    scenario:       str,
    shapes:         List[Dict[str, Any]],
    diagram_type:   str = "class",
    ignored_errors: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate any diagram type using OpenAI.
    diagram_type: 'class' | 'usecase' | 'sequence'
    ignored_errors: list of error fingerprints the user has dismissed
                    (format: "ERROR_TYPE|element|from_element|to_element", all lowercase).
                    These errors will NEVER be reported again in this or future validations.
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping AI validation")
        return None

    _ignored = [_n(x) for x in (ignored_errors or [])]

    # Step 1 — Sanitize shapes
    clean_shapes = _sanitize_shapes(shapes, diagram_type)
    _log.info("Sanitized shapes: %d → %d (diagram: %s)", len(shapes), len(clean_shapes), diagram_type)

    # Step 2 — Run deterministic rule-based checks (connections, duplicates, empty names)
    rule_errors = _run_rule_checks(clean_shapes, diagram_type, scenario)
    _log.info("Rule checks: %d issues found", len(rule_errors))

    # Step 3 — Ask LLM only for scenario-semantic checks
    prompt = _build_prompt(diagram_type, scenario, clean_shapes)

    for model in _MODELS:
        _log.info("Trying model: %s", model)
        result = _call_model(prompt, api_key, model)
        if not result:
            continue
        _log.info("Model %s succeeded", model)

        # Normalize LLM output
        llm_e, llm_w, llm_i = [], [], []
        for e in result.get("errors", []):
            sev = str(e.get("severity", "ERROR")).upper()
            raw_fix = e.get("auto_fix") or {}
            auto_fix = {
                "fixable":           bool(raw_fix.get("fixable", False)),
                "action":            str(raw_fix.get("action",            "")),
                "shape_type":        str(raw_fix.get("shape_type",        "")),
                "name":              str(raw_fix.get("name",              "")),
                "from_element":      str(raw_fix.get("from_element",      "")),
                "to_element":        str(raw_fix.get("to_element",        "")),
                "arrow_type":        str(raw_fix.get("arrow_type",        "")),
                "message_label":     str(raw_fix.get("message_label",     "")),
                "multiplicity_from": str(raw_fix.get("multiplicity_from", "")),
                "multiplicity_to":   str(raw_fix.get("multiplicity_to",   "")),
            }
            if not auto_fix["fixable"]:
                auto_fix = _build_fallback_fix(
                    str(e.get("error_type", "")), str(e.get("element", "")), e, diagram_type, scenario)
            item = {
                "error_type":  str(e.get("error_type", "UNKNOWN")),
                "severity":    sev,
                "element":     str(e.get("element", "")),
                "description": str(e.get("description", "")),
                "suggestion":  str(e.get("suggestion", "")),
                "auto_fix":    auto_fix,
            }
            if sev == "WARNING": llm_w.append(item)
            elif sev == "INFO":  llm_i.append(item)
            else:                llm_e.append(item)

        # Step 4 — Merge: rule errors + filtered LLM errors
        return _merge_results(
            rule_errors, llm_e, llm_w, llm_i,
            clean_shapes,
            int(result.get("score", 50)),
            result.get("summary", ""),
            _ignored,
            scenario,
        )

    # LLM failed — return rule-only results
    _log.error("All models failed — returning rule-only results")
    re_ = [r for r in rule_errors if r.get("severity") == "ERROR"]
    rw_ = [r for r in rule_errors if r.get("severity") == "WARNING"]
    ri_ = [r for r in rule_errors if r.get("severity") == "INFO"]
    all_ = re_ + rw_ + ri_
    def _fmt(r): return {k: r[k] for k in ("error_type","severity","element","description","suggestion","auto_fix") if k in r}
    return {
        "is_valid": len(re_) == 0,
        "score": max(0, 100 - len(re_)*15 - len(rw_)*5),
        "summary": "Validated by rule checks (AI unavailable)",
        "errors": [_fmt(r) for r in re_],
        "warnings": [_fmt(r) for r in rw_],
        "info": [_fmt(r) for r in ri_],
        "total_issues": len(all_),
        "fixable_count": sum(1 for r in all_ if r.get("auto_fix",{}).get("fixable")),
        "source": "rules-only",
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-AWARE PROMPT — jab shapes empty ho aur image available ho
# ─────────────────────────────────────────────────────────────────────────────

def _build_image_prompt(diagram_type: str, scenario: str) -> str:
    """
    Jab shapes empty ho aur image available ho — image ko directly analyze karo.
    """
    dt = diagram_type.lower()

    if "usecase" in dt or "use_case" in dt:
        rules = """1. MISSING_ACTOR — Every person/system in scenario must be an actor (stick figure).
2. EXTRA_ACTOR — Actor not mentioned in scenario (warning).
3. MISSING_USE_CASE — Every action/function in scenario must be a use case oval.
4. EXTRA_USE_CASE — Use case oval not in scenario (info).
5. DISCONNECTED_ACTOR — Actor has NO line to any use case. A line touching ANY part of the actor (head, body, hands, feet) = CONNECTED. Only flag if truly no line exists near the actor.
6. ISOLATED_USE_CASE — Use case has no connection to any actor.
7. MISSING_SYSTEM_BOUNDARY — System boundary box is entirely absent. Only report if you CANNOT SEE any rectangle/box enclosing use cases.
8. WRONG_SYSTEM_BOUNDARY_NAME — Boundary EXISTS but its label doesn't match scenario system name. Say "change name from X to Y" — never say "add a new boundary".
9. WRONG_RELATIONSHIP — include/extend/generalization used incorrectly.
10. MISSING_VERB_IN_USE_CASE — Use case oval label must follow the 'Verb + Noun' format ONLY — a single action verb followed by a single noun/object (e.g. "Manage Order", "Place Order", "Browse Products"). Flag this if the label is missing a verb, missing a noun, or has extra words beyond verb + noun.
11. DUPLICATE_ACTOR — Same actor name appears twice.
12. DUPLICATE_USE_CASE — Same use case name appears twice.
13. SPELLING_MISTAKE   — An actor or use case name closely resembles a scenario name but is misspelled. ONE error only — do NOT also report MISSING_ACTOR/MISSING_USE_CASE or EXTRA_ACTOR/EXTRA_USE_CASE for the same element.
14. WRONG_ACTOR_NAME — Actor names must be nouns/roles (e.g. "Customer", "Admin", "Bank", "System"), never verbs/actions (e.g. "Login", "Register", "Manage", "Browse", "Pay"). Flag any actor whose label is a verb/action.
CASE-INSENSITIVE: "Login" and "login" are the same — do NOT flag capitalisation as missing/extra."""
        dtype_label = "USE CASE"
        extra_rules = """
## ACTOR CONNECTION RULE:
- An actor stick-figure occupies vertical space (head, body, hands, feet).
- Any line touching ANY part of the stick-figure = CONNECTED.
- Only flag DISCONNECTED_ACTOR if there is genuinely no line anywhere near the actor.

## SYSTEM BOUNDARY NAME RULE:
- If boundary EXISTS with wrong name → report WRONG_SYSTEM_BOUNDARY_NAME, suggest renaming.
- If boundary does NOT exist → report MISSING_SYSTEM_BOUNDARY.
- Never report both for the same diagram.

## USE CASE NAMING FORMAT RULE — CRITICAL:
- A valid use case label is EXACTLY "Verb + Noun" — two parts only (e.g. "Manage Order").
- Do NOT require or expect a third word or a trailing verb. "Verb + Noun + Verb" is NOT the rule.
- If the label has only a verb, only a noun, or more than a verb+noun pair, report MISSING_VERB_IN_USE_CASE.
- suggestion: "Rename 'X' to a verb + noun, e.g. 'Manage X' or 'Process X'."

## ACTOR NAME FORMAT RULE — CRITICAL:
- Actor labels must be nouns/roles, never verbs/actions.
- If an actor label is a verb/action → report WRONG_ACTOR_NAME as WARNING.
- description: "Actor names must represent roles or entities, not actions."
- suggestion: "Rename actor to a proper role name like 'Customer' or 'Admin'." """

    elif "sequence" in dt:
        rules = """1. MISSING_LIFELINE — Every participant in scenario must have a lifeline or object box.
2. MISSING_MESSAGE — Important interaction in scenario not shown. Use SEMANTIC matching — equivalent messages count.
3. WRONG_MESSAGE_ORDER — ONLY report if order is CLEARLY and DEFINITIVELY wrong. Skip if any doubt.
4. MISSING_RETURN — Only if scenario explicitly requires a response message.
5. ISOLATED_LIFELINE — Lifeline sends/receives no messages.
6. UNLABELLED_LIFELINE — Lifeline box has no label.
7. UNLABELLED_OBJECT — Object box has no label (say "object" not "lifeline").
8. MISSING_DELETION_SYMBOL — Lifeline has no X/destroy marker. Only report if X symbols are genuinely absent.
9. UNLABELLED_ARROW — Message arrow has no label.
10. SPELLING_MISTAKE — A lifeline/object/actor name closely resembles a scenario name but is misspelled. ONE error only — do NOT also report MISSING_LIFELINE or EXTRA_LIFELINE for the same element."""
        dtype_label = "SEQUENCE"
        extra_rules = """
## SELF-MESSAGE RULE:
- Self-message arrows (looping back to same lifeline) are VALID UML — do NOT report them as errors.

## DELETION SYMBOL RULE:
- If you SEE any X marks at lifeline ends → deletion symbols ARE present. Do NOT report them missing.
- Only report MISSING_DELETION_SYMBOL for lifelines with genuinely no X at the bottom.

## OBJECT vs LIFELINE:
- An object box (rectangle with name) IS a named participant. Do NOT report it as unnamed.
- If the object is truly empty/unlabelled → say "Object has no name", not "lifeline has no name".

## MESSAGE ORDER:
- Only report WRONG_MESSAGE_ORDER if you are 100% certain. When in doubt → SKIP."""

    else:
        rules = """1. MISSING_CLASS              — Important nouns in scenario must be classes. Only explicitly named entities.
2. EXTRA_CLASS                — Class in diagram not mentioned in scenario (warning).
3. WRONG_CLASS_CAPITALISATION — Class name exists but starts with lowercase — suggest capitalising, never suggest removing.
4. WRONG_RELATIONSHIP_TYPE    — Wrong arrow type vs scenario (association/aggregation/composition/generalization).
5. MISSING_RELATIONSHIP       — Relationship EXPLICITLY described in scenario but not drawn.
6. MISSING_MULTIPLICITY       — Association/aggregation/composition arrow drawn but both multiplicity fields are empty.
7. WRONG_MULTIPLICITY         — Multiplicity present but value is wrong vs scenario.
8. MISSING_ASSOCIATION_LABEL  — Scenario names a label for a relationship but it is not on the drawn arrow.
9. WRONG_INHERITANCE_DIRECTION — Inheritance arrow reversed (child should point TO parent).
10. DUPLICATE_CLASS           — Same class name appears twice.
11. CIRCULAR_INHERITANCE      — A inherits B and B inherits A.
12. EMPTY_CLASS_NAME          — Class has no name or placeholder like "Class 1".
13. SELF_ASSOCIATION          — Class connected to itself (warn unless scenario says so).
14. SPELLING_MISTAKE          — A class name closely resembles a scenario class name but is misspelled (e.g. "Custmer" instead of "Customer"). ONE error only — do NOT also report MISSING_CLASS or EXTRA_CLASS for the same element.
15. WRONG_ASSOCIATION_LABEL   — An association/aggregation/composition arrow HAS a visible label, but that label does not match what the scenario describes for that relationship. ALWAYS include the correct word/phrase taken directly from the scenario in the "suggestion" field."""
        dtype_label = "CLASS"
        extra_rules = """
## RELATIONSHIP TYPE VALIDATION:
- If scenario says "is a" / "inherits" / "type of" / "kind of" → expect GENERALIZATION.
- If scenario says "consists of" / "composed of" / "cannot exist without" → expect COMPOSITION.
- If scenario says "contains" / "collection of" / "holds" / "is made up of" → expect AGGREGATION.
- If scenario says "has" / "uses" / "is related to" / "is associated with" → expect ASSOCIATION.
- Wrong type drawn → report WRONG_RELATIONSHIP_TYPE.

## CLASS CAPITALISATION RULE:
- If a class "customer" exists and scenario has "Customer" → report WRONG_CLASS_CAPITALISATION.
- Suggestion: "Capitalise the first letter: rename 'customer' to 'Customer'."
- Do NOT report it as EXTRA_CLASS or say to remove it.

## SPELLING MISTAKE RULE:
- SPELLING_MISTAKE applies ONLY when a class box IS VISIBLE in the image and its name is a NEAR-IDENTICAL misspelling of a scenario class name — differing by only 1-2 letters and clearly the same word (e.g. "Custmer" vs "Customer"). Do NOT match names that are merely related in topic or share a starting letter.
- If a scenario class has no class box in the image with a near-identical name (including when that class box does not exist at all) → you MUST report MISSING_CLASS. Go through each scenario class one by one and check it against every visible class box before deciding. Do NOT report SPELLING_MISTAKE against an unrelated class box just to explain the gap.
- When SPELLING_MISTAKE genuinely applies: report EXACTLY ONE SPELLING_MISTAKE error and do NOT also report MISSING_CLASS for the correctly-spelled version or EXTRA_CLASS for the misspelled version.

## DUPLICATE_CLASS RULE — ALWAYS CHECK:
- Scan every class box name in the image against every other class box name.
- If the SAME class name (case-insensitive) appears in two or more separate boxes → report DUPLICATE_CLASS as an ERROR for that class name.
- element: the repeated class name. description: "Class 'X' appears more than once in the diagram." suggestion: "Remove or merge the duplicate 'X' class."

## MISSING_ASSOCIATION_LABEL RULE — ALWAYS CHECK:
- Look at every association/aggregation/composition line/arrow in the image.
- If the scenario describes that relationship with a verb (e.g. "Bank manages Customer", "Teacher teaches Student") and the drawn line has NO visible text label → report MISSING_ASSOCIATION_LABEL as a WARNING, even if you are also checking other things about that same arrow.
- Do not skip this check just because the arrow itself is present and correctly connects the two classes — connection alone does not satisfy the label requirement.

## WRONG_ASSOCIATION_LABEL RULE — ALWAYS CHECK:
- Look at every association/aggregation/composition line/arrow that DOES have a visible text label.
- Compare the visible label word-for-word (allowing minor wording variants like "manage" vs "manages") against what the scenario says that relationship should be.
- If they clearly differ in meaning → report WRONG_ASSOCIATION_LABEL as a WARNING, and always state the exact correct word/phrase from the scenario in the suggestion.

## RELATIONSHIP RULE:
- Only report MISSING_RELATIONSHIP if scenario EXPLICITLY states a relationship (has, contains, inherits, etc.).
- Do NOT invent relationships just because two classes exist.

## MISSING LABEL RULE:
- Only report MISSING_ASSOCIATION_LABEL if scenario names what the relationship should be called.
- If scenario does not name the relationship → labels are optional.

## WRONG ASSOCIATION LABEL RULE — CRITICAL:
- Check every association/aggregation/composition arrow that HAS a visible label.
- If the drawn label does NOT match what the scenario describes for that relationship → report WRONG_ASSOCIATION_LABEL as a WARNING.
- Example: scenario says "Bank manages Customer" but arrow label reads "owns" → WRONG_ASSOCIATION_LABEL.
- suggestion: ALWAYS name the exact correct word/phrase from the scenario, e.g. "Change label 'owns' to 'manages' as described in the scenario."
- Minor wording variations (e.g. "manage" vs "manages") are acceptable — do NOT report those.

## MULTIPLICITY RULES:
- If a multiplicity value is visible at EITHER end of an association/aggregation/composition arrow → multiplicity EXISTS, do NOT report MISSING_MULTIPLICITY.
- Only report MISSING_MULTIPLICITY if BOTH ends genuinely show no multiplicity value.
- WRONG_MULTIPLICITY: only report if the scenario CLEARLY states cardinality (one-to-many, many-to-many, etc.) and the drawn values contradict it.
- Do NOT suggest a specific multiplicity value unless the scenario explicitly states it.

## INHERITANCE RULES:
- WRONG_INHERITANCE_DIRECTION — the arrow should point FROM the child class TO the parent class; flag if reversed.
- CIRCULAR_INHERITANCE — flag if A inherits from B and B inherits from A.

## DUPLICATE / EMPTY / SELF-ASSOCIATION:
- DUPLICATE_CLASS — same class name appears twice in the image.
- EMPTY_CLASS_NAME — a class box has no visible name or a placeholder like "Class 1".
- SELF_ASSOCIATION — a class connects to itself; only flag as a WARNING unless the scenario explicitly describes this.

## MISSING_ATTRIBUTE / MISSING_METHOD:
- Only report if the scenario EXPLICITLY lists specific attributes or methods for a class.
- If the scenario does not list them, do NOT report anything missing, regardless of what is drawn."""

    return f"""You are an expert UML {dtype_label} Diagram validator using SEMANTIC analysis. You are given an IMAGE of the diagram.

## CRITICAL INSTRUCTION
Look carefully at the ACTUAL IMAGE provided. Validate ONLY what you can SEE.
- Only report something as MISSING if it is genuinely absent from the image.
- Do NOT invent errors for things that are present but styled differently than expected.
- Results must be DETERMINISTIC — same image + scenario must always produce the same errors.

## SCENARIO
{scenario}

## RULES TO CHECK ({dtype_label} diagram ONLY)
{rules}
{extra_rules}

## SEMANTIC ANALYSIS:
- Use semantic reasoning — a diagram correct in overall logic is correct even if exact wording differs.
- Semantically equivalent labels count as correct (e.g. "validateUser()" ≈ "validate credentials").

## SEVERITY
ERROR = must fix | WARNING = should fix | INFO = suggestion

## AUTO-FIX INSTRUCTIONS
For each error, provide an "auto_fix" object. Set "fixable": true only for these:
- MISSING_ACTOR / MISSING_LIFELINE / MISSING_CLASS → action: "add_shape", shape_type, name
- MISSING_USE_CASE → action: "add_shape", shape_type: "use_case_oval", name
- MISSING_SYSTEM_BOUNDARY → action: "add_boundary", shape_type: "system_boundary"
- WRONG_SYSTEM_BOUNDARY_NAME → action: "rename_shape", name: <correct name>
- WRONG_CLASS_CAPITALISATION → action: "rename_shape", name: <capitalised name>
- MISSING_RELATIONSHIP / MISSING_MESSAGE → action: "add_arrow", from_element, to_element, arrow_type, message_label
- MISSING_RETURN → action: "add_arrow", from_element, to_element, arrow_type: "dashed_arrow", message_label
- MISSING_DELETION_SYMBOL → action: "add_shape", shape_type: "deletion_marker", name: <lifeline name>
- EMPTY_CLASS_NAME / UNLABELLED_LIFELINE / UNLABELLED_OBJECT → action: "rename_shape", name
- DUPLICATE_* → action: "merge_shapes", name
- SPELLING_MISTAKE → action: "rename_shape", name: <correct spelling from scenario>, fixable: true
- DISCONNECTED_ACTOR / ISOLATED_USE_CASE → action: "add_arrow", from_element, to_element, arrow_type: "association"
- MISSING_MULTIPLICITY → action: "add_label", from_element, to_element, multiplicity_from, multiplicity_to
- MISSING_VERB_IN_USE_CASE → action: "rename_shape", name: <corrected name with verb + noun only>, fixable: true
- WRONG_ASSOCIATION_LABEL → action: "add_label", from_element, to_element, label: <correct label from scenario>, fixable: true
- WRONG_ACTOR_NAME → fixable: false
All other errors → fixable: false

## RESPONSE FORMAT (JSON only, no markdown)
{{
  "errors": [
    {{
      "error_type": "RULE_CODE",
      "severity": "ERROR|WARNING|INFO",
      "element": "element name",
      "description": "What is wrong",
      "suggestion": "How to fix",
      "auto_fix": {{
        "fixable": true,
        "action": "add_shape",
        "shape_type": "actor",
        "name": "MissingActorName"
      }}
    }}
  ],
  "score": 0-100,
  "summary": "brief summary"
}}

If correct: {{"errors": [], "score": 100, "summary": "Diagram is correct"}}

## STRICT RULES — MUST FOLLOW:
- Only report issues you are CONFIDENT about from what you SEE. Do NOT invent errors.
- STRICT: Same diagram + scenario must always produce same results (deterministic).
- STRICT: If something is NOT explicitly mentioned in the scenario, do NOT report it as missing.
- STRICT: Method names ending with () are METHODS not class names.
- STRICT: Attribute names are ATTRIBUTES not class names.
- STRICT: Do NOT report MISSING_ATTRIBUTE or MISSING_METHOD unless scenario explicitly requires them.
- STRICT: Do NOT invent sub-use-cases, intermediate messages, or implied relationships not in scenario.
- STRICT: Only report what you can clearly SEE is wrong — when in doubt, SKIP the error.
- STRICT: Use semantic matching — equivalent labels count as correct."""


def _call_model_with_image(prompt: str, image_b64: str, mime_type: str, api_key: str, model: str) -> Optional[Dict]:
    """OpenAI Vision — image + text prompt dono bhejo."""
    url = f"{_OPENAI_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            },
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1500,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            _log.warning("Vision model %s: rate limit — waiting %ss", model, _RETRY_WAIT)
            time.sleep(_RETRY_WAIT)
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                    body = resp.read().decode("utf-8")
            except Exception as e2:
                _log.warning("Vision retry failed: %s", e2)
                return None
        else:
            _log.error("Vision model %s: HTTP %s: %s", model, e.code, err_body[:300])
            return None
    except Exception as e:
        _log.warning("Vision model %s failed: %s", model, e)
        return None

    try:
        outer = json.loads(body)
        text  = outer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        _log.warning("Vision parse error: %s", e)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _log.warning("Vision JSON error: %s | text: %s", e, text[:200])
        return None


def validate_with_openai_image(
    scenario:     str,
    image_b64:    str,
    mime_type:    str = "image/png",
    diagram_type: str = "class",
) -> Optional[Dict[str, Any]]:
    """
    Image-based validation — shapes ki jagah actual diagram image bhejo OpenAI ko.
    Yeh tab use hota hai jab user gallery se image upload kare (canvas ke baghair).

    diagram_type: 'class' | 'usecase' | 'sequence'
    Returns structured validation result or None.
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY not set — skipping image validation")
        return None

    # Vision-capable models only
    vision_models = ["gpt-4o-2024-08-06", "gpt-4o-mini"]

    prompt = _build_image_prompt(diagram_type, scenario)

    for model in vision_models:
        _log.info("Trying Vision model: %s (diagram: %s)", model, diagram_type)
        result = _call_model_with_image(prompt, image_b64, mime_type, api_key, model)
        if result:
            _log.info("Vision model %s succeeded!", model)
            raw_errors = result.get("errors", [])
            score      = int(result.get("score", 50))
            summary    = result.get("summary", "Image validation complete")

            errors, warnings, info = [], [], []
            for e in raw_errors:
                sev  = str(e.get("severity", "ERROR")).upper()
                raw_fix = e.get("auto_fix", {})
                auto_fix = {
                    "fixable":          bool(raw_fix.get("fixable", False)),
                    "action":           str(raw_fix.get("action",           "")),
                    "shape_type":       str(raw_fix.get("shape_type",       "")),
                    "name":             str(raw_fix.get("name",             "")),
                    "from_element":     str(raw_fix.get("from_element",     "")),
                    "to_element":       str(raw_fix.get("to_element",       "")),
                    "arrow_type":       str(raw_fix.get("arrow_type",       "")),
                    "message_label":    str(raw_fix.get("message_label",    "")),
                    "multiplicity_from":str(raw_fix.get("multiplicity_from","")),
                    "multiplicity_to":  str(raw_fix.get("multiplicity_to",  "")),
                } if raw_fix else {"fixable": False}

                error_type = str(e.get("error_type", "UNKNOWN"))
                element    = str(e.get("element",    ""))

                if not auto_fix.get("fixable"):
                    auto_fix = _build_fallback_fix(error_type, element, e, diagram_type, scenario)

                item = {
                    "error_type":  error_type,
                    "severity":    sev,
                    "element":     element,
                    "description": str(e.get("description", "")),
                    "suggestion":  str(e.get("suggestion",  "")),
                    "auto_fix":    auto_fix,
                }
                if sev == "WARNING":  warnings.append(item)
                elif sev == "INFO":   info.append(item)
                else:                 errors.append(item)

            all_items = errors + warnings + info

            # Spelling mistake dedup: suppress MISSING_*/EXTRA_* for elements already flagged as SPELLING_MISTAKE
            _img_spelled = set()
            for _i in all_items:
                if _n(str(_i.get("error_type", ""))) == "spelling_mistake":
                    _img_spelled.add(_n(str(_i.get("element", ""))))
            if _img_spelled:
                def _img_keep(i):
                    et   = _n(str(i.get("error_type", "")))
                    elem = _n(str(i.get("element", "")))
                    if et == "spelling_mistake":
                        return True
                    if elem in _img_spelled and ("missing" in et or "extra" in et):
                        return False
                    return True
                errors   = [i for i in errors   if _img_keep(i)]
                warnings = [i for i in warnings if _img_keep(i)]
                info     = [i for i in info     if _img_keep(i)]
                all_items = errors + warnings + info

            fixable_count = sum(1 for i in all_items if i.get("auto_fix", {}).get("fixable"))

            return {
                "is_valid":        len(errors) == 0,
                "score":           score,
                "summary":         summary,
                "errors":          errors,
                "warnings":        warnings,
                "info":            info,
                "total_issues":    len(raw_errors),
                "fixable_count":   fixable_count,
                "source":          "openai-vision",
                "validation_mode": "gemini",
            }

    _log.error("All Vision models failed!")
    return None
