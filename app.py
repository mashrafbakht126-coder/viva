"""
OOAD Diagram Validation Engine - Flask Backend
Validation: Gemini AI (primary)
All 3 diagram types: class, usecase, sequence

FIX: Image upload, diagram_type auto-detect via Gemini Vision.
     if diagram_type is missing/unknown then automatically analyse image that 
     What type of diagram it is (class/usecase/sequence) using OpenAI Vision.
     

UPDATE: gpt-4o -> gpt-4o-mini (rate limit fix), timeout 120s, retry logic added
"""

import os
import re
import json
import base64
import logging
import time
import urllib.request
import urllib.error
from flask import Flask, request, jsonify
from flask_cors import CORS

from nlp_extractor import NLPExtractor
from validators.openai_validator import validate_with_openai, validate_with_openai_image

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app       = Flask(__name__)
CORS(app)
extractor = NLPExtractor()

# OpenAI API config
_OPENAI_API_BASE = "https://api.openai.com/v1"
_VISION_MODELS   = ["gpt-4o-mini"]   # FIX: gpt-4o-mini use
_TIMEOUT         = 120               # FIX:  (worker crash close)
_RETRY_WAIT      = 65                # wait seconds on Rate limit


def _get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key if key else None


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT diagram type from image using OpenAI Vision
# ─────────────────────────────────────────────────────────────────────────────

def _detect_diagram_type_from_image(image_b64: str, mime_type: str = "image/png") -> str:
    """
    Using OpenAI Vision to analyze the image and detect the diagram type.
    Returns: 'class' | 'usecase' | 'sequence'
    Default: 'class' (agar detect na ho sake)
    """
    api_key = _get_api_key()
    if not api_key:
        _log.warning("OPENAI_API_KEY missing - cannot auto-detect diagram type")
        return "class"

    prompt = """Look at this UML diagram image carefully.

Determine which ONE of these three diagram types it is:
1. CLASS diagram     - has rectangles with class names, attributes, methods; arrows for inheritance/association
2. USE CASE diagram  - has stick figures (actors), ovals/ellipses (use cases), system boundary rectangle
3. SEQUENCE diagram  - has vertical dashed lines (lifelines), horizontal arrows between them (messages)

Reply with ONLY one word - exactly one of: class, usecase, sequence
Do not explain. Just the single word."""

    payload = json.dumps({
        "model": "gpt-4o-mini",   # FIX:  use gpt-4o-mini
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 10,
    }).encode("utf-8")

    for model in _VISION_MODELS:
        url = f"{_OPENAI_API_BASE}/chat/completions"
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        # FIX: Retry logic - try on rate limit again 
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"].strip().lower()
                text = re.sub(r"[^a-z]", "", text.split()[0] if text.split() else "")
                if text in ("class", "usecase", "sequence"):
                    _log.info("Auto-detected diagram type: '%s' (model: %s)", text, model)
                    return text
                if "class" in text:   return "class"
                if "use" in text:     return "usecase"
                if "seq" in text:     return "sequence"
                break  # recieve response , close loop 
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit error
                    if attempt < 2:
                        _log.warning("Rate limit hit (attempt %d/3) - waiting %ds...", attempt + 1, _RETRY_WAIT)
                        time.sleep(_RETRY_WAIT)
                    else:
                        _log.error("Rate limit - 3 attempts failed for model %s", model)
                else:
                    _log.warning("Vision model %s HTTP error %d: %s", model, e.code, e)
                    break
            except Exception as e:
                _log.warning("Vision model %s failed: %s", model, e)
                break

    _log.warning("Could not auto-detect diagram type - defaulting to 'class'")
    return "class"


# ─────────────────────────────────────────────────────────────────────────────
#  Normalize diagram_type string
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_dtype(raw: str) -> str:
    """'UseCase', 'use_case', 'CLASS' etc. -> 'class'/'usecase'/'sequence' or ''"""
    s = raw.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if s in ("class", "classdiagram"):          return "class"
    if s in ("usecase", "usecasediagram", "uc"): return "usecase"
    if s in ("sequence", "sequencediagram", "seq"): return "sequence"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OOAD Hybrid Validation Engine",
        "status":  "running",
        "mode":    "OpenAI GPT-4o-mini (primary) + Rule-Based (fallback)",
        "features": {
            "image_auto_detect": "Upload image -> OpenAI Vision auto-detects diagram type",
            "diagram_types":     ["class", "usecase", "sequence"],
        },
        "endpoints": {"/health": "health check", "/validate": "validate diagram", "/extract": "NLP only"}
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":         "ok",
        "message":        "Hybrid Validation Engine is running",
        "openai_enabled": bool(_get_api_key()),
    })


@app.route('/validate', methods=['POST'])
def validate():
    """
    Format A - JSON body:
        { "scenario": "...", "diagram_type": "class|usecase|sequence", "shapes": [...] }

    Format B - multipart form + image file:
        scenario      = text field
        diagram_type  = optional (auto-detected from image if missing)
        image         = image file (PNG/JPG)
        shapes        = optional JSON string
    """
    image_b64      = None
    mime_type      = "image/png"
    shapes         = []
    scenario       = ""
    dtype_raw      = ""
    ignored_errors = []

    if request.content_type and "multipart" in request.content_type:
        # Format B: form data + image
        scenario   = (request.form.get("scenario", "") or "").strip()
        dtype_raw  = (request.form.get("diagram_type", "") or "").strip()
        shapes_str = request.form.get("shapes", "")
        if shapes_str:
            try:    shapes = json.loads(shapes_str)
            except: shapes = []

        ignored_str = request.form.get("ignored_errors", "")
        if ignored_str:
            try:    ignored_errors = json.loads(ignored_str)
            except: ignored_errors = []

        img_file = request.files.get("image")
        if img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type = img_file.mimetype or "image/png"
    else:
        # Format A: JSON body
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400
        scenario       = (data.get("scenario", "") or "").strip()
        dtype_raw      = (data.get("diagram_type", "") or "").strip()
        shapes         = data.get("shapes", [])
        ignored_errors = data.get("ignored_errors", []) or []
        # Also support base64 image inside JSON
        if data.get("image"):
            image_b64 = data["image"]
            mime_type = data.get("mime_type", "image/png")

    if not scenario:
        return jsonify({"error": "scenario field is required"}), 400

    # ── Determine diagram type ─────────────────────────────────────────────
    dtype = _normalize_dtype(dtype_raw)

    if not dtype:
        if image_b64:
            _log.info("diagram_type missing - auto-detecting from image...")
            dtype = _detect_diagram_type_from_image(image_b64, mime_type)
        else:
            return jsonify({
                "error": (
                    f"Unknown or missing diagram_type: '{dtype_raw}'. "
                    "Use 'class', 'usecase', or 'sequence'. "
                    "Or upload an image for auto-detection."
                )
            }), 400

    # ── NLP extraction ─────────────────────────────────────────────────────
    extracted = extractor.extract(scenario)
     # ── TEMP DEBUG: dump incoming payload so we can inspect real shape data ─
    try:
        import json as _json
        with open('/tmp/last_request.json', 'w') as _f:
            _json.dump({"scenario": scenario, "diagram_type": dtype, "shapes": shapes}, _f, indent=2)
        _log.info("DEBUG: dumped request payload to /tmp/last_request.json")
    except Exception as _e:
        _log.warning("DEBUG dump failed: %s", _e)

    # ── OpenAI AI first (PRIMARY) ──────────────────────────────────────────
    # when image available → use OpenAI Vision  (analyzeimage directly )
    # when ther is only shapes  → use text-based OpenAI
    if image_b64:
        _log.info("Image available — using OpenAI Vision for '%s' diagram", dtype)
        gemini_result = validate_with_openai_image(
            scenario=scenario,
            image_b64=image_b64,
            mime_type=mime_type,
            diagram_type=dtype,
        )
    else:
        gemini_result = validate_with_openai(
            scenario,
            shapes,
            diagram_type=dtype,
            ignored_errors=ignored_errors,
            extracted=extracted,
        )

    if gemini_result:
        _log.info("OpenAI validation used for '%s' diagram", dtype)
        gemini_result["validation_mode"] = "openai"
        final_result = gemini_result
    else:
        return jsonify({"error": "OpenAI unavailable. Please try again later."}), 503
    
    return jsonify({
        "diagram_type":        dtype,
        "auto_detected":       bool(image_b64 and not _normalize_dtype(dtype_raw)),
        "extracted_elements":  extracted,
        "validation_result":   final_result,
    })


@app.route('/extract', methods=['POST'])
def extract_only():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    scenario = (data.get("scenario", "") or "").strip()
    if not scenario:
        return jsonify({"error": "scenario required"}), 400
    return jsonify(extractor.extract(scenario))


if __name__ == '__main__':
    key = _get_api_key()
    if not key:
        _log.warning("OPENAI_API_KEY not set - rule-based only. Image auto-detect DISABLED.")
    else:
        _log.info("OpenAI API key found - AI validation + image auto-detect ENABLED")
    app.run(debug=True, host='0.0.0.0', port=5000)
