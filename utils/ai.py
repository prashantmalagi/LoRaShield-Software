"""
LoRaShield – AI Utility Module (Production)
============================================
Real TensorFlow/Keras integration for Morse-to-Text AI decoding.

Model architecture (from training inspection):
    - Tokenizer : Keras Tokenizer  { '.': 1, '-': 2 }
    - Encoder   : sklearn LabelEncoder, 36 classes  (0-9, A-Z)
    - Model     : input_shape=(32, 10)  output_shape=(32, 36)
                  → Character-level classifier: maps one Morse symbol
                    sequence (≤10 dots/dashes) to one predicted character.

Inference strategy:
    1. Split the full Morse string into per-character codes (split on space,
       use ' / ' as word boundary → space character).
    2. For each Morse code, tokenise the dot/dash sequence and pad to len=10.
    3. Batch-predict all characters in a single model.predict() call.
    4. Decode each predicted index via LabelEncoder.classes_.
    5. Aggregate confidence as mean(max(softmax)) across all characters.

Text → Morse:
    Uses the built-in ITU Morse dictionary.
    Unsupported characters are silently skipped (no '?' emission).

All events are logged via utils.logger AND printed to stdout so they appear
in the VS Code terminal even when the GUI is running.
"""

import os
import pickle
import sys
import time
import traceback
import warnings
import zipfile
from pathlib import Path

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# ── Suppress TF C++ noise (set BEFORE importing TF) ──────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Absolute paths relative to project root ───────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_PATH     = BASE_DIR / "models" / "morse_decoder.h5"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
ENCODER_PATH   = BASE_DIR / "models" / "encoder.pkl"

# Model input sequence length (must match training)
MODEL_SEQ_LEN = 10

# Confidence threshold below which prediction is flagged as "low confidence"
LOW_CONFIDENCE_THRESHOLD = 0.60

# ── Module-level state ────────────────────────────────────────────────────────
_model        = None
_tokenizer    = None
_encoder      = None
_model_loaded = False

# ── Standard Morse Code Dictionary (ITU) ─────────────────────────────────────
MORSE_CODE_DICT: dict[str, str] = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..',  '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
    '!': '-.-.--', '/': '-..-.',  '(': '-.--.',  ')': '-.--.-',
    '&': '.-...',  ':': '---...', ';': '-.-.-.',  '=': '-...-',
    '+': '.-.-.',  '-': '-....-', '_': '..--.-',  '"': '.-..-.',
    '$': '...-..-', '@': '.--.-.',
}

MORSE_REVERSE_DICT: dict[str, str] = {v: k for k, v in MORSE_CODE_DICT.items()}
SUPPORTED_CHARS = set(MORSE_CODE_DICT.keys())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _diag_print(msg: str) -> None:
    """Print msg to stdout (VS Code terminal) AND log at INFO level.

    Handles consoles that cannot encode specific Unicode by falling back
    to ASCII-safe output for the print() call.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe, flush=True)
    log.info(msg)


def _inspect_h5_file(path: Path) -> str:
    """
    Inspect the .h5 file to determine its actual format and report findings.
    Returns a human-readable ASCII-safe description string.
    """
    lines: list[str] = []
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
        lines.append(f"  File header (hex): {header.hex()}")

        # HDF5 magic bytes: \x89HDF\r\n\x1a\n
        if header[:8] == b"\x89HDF\r\n\x1a\n":
            lines.append("  Format: HDF5 (legacy .h5 format) - OK")
        elif zipfile.is_zipfile(path):
            lines.append("  Format: ZIP archive (Keras v3 / SavedModel format)")
            with zipfile.ZipFile(path) as z:
                contents = z.namelist()[:15]
                lines.append(f"  ZIP contents: {contents}")
        else:
            lines.append("  Format: UNKNOWN - not HDF5 and not ZIP")
    except Exception as exc:
        lines.append(f"  Inspection error: {exc}")

    # Try h5py
    try:
        import h5py
        with h5py.File(path, "r") as hf:
            lines.append(f"  h5py keys: {list(hf.keys())}")
    except ImportError:
        lines.append("  h5py: not installed (pip install h5py)")
    except Exception as exc:
        lines.append(f"  h5py error: {exc}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_ai_model(
    model_path:     str | Path = MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    encoder_path:   str | Path = ENCODER_PATH,
) -> tuple[bool, str]:
    """
    Load the TensorFlow model, Keras tokenizer, and sklearn LabelEncoder.

    Returns:
        ``(True, "AI Loaded")`` on success.
        ``(False, full_exception_string)`` on any failure – never hidden.

    All steps are printed to stdout AND written to logs/app.log.
    """
    global _model, _tokenizer, _encoder, _model_loaded

    if _model_loaded:
        _diag_print("AI model already loaded – skipping reload.")
        return (True, "AI Loaded")

    model_path     = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    encoder_path   = Path(encoder_path)

    # ── Diagnostic header ────────────────────────────────────────────────────
    sep = "-" * 50
    _diag_print(sep)
    _diag_print("Loading AI Model...")
    _diag_print(f"Model path:       {model_path}")
    _diag_print(f"Tokenizer path:   {tokenizer_path}")
    _diag_print(f"Encoder path:     {encoder_path}")
    _diag_print(f"Current Working Directory: {os.getcwd()}")

    # ── File existence ────────────────────────────────────────────────────────
    model_exists     = model_path.exists()
    tokenizer_exists = tokenizer_path.exists()
    encoder_exists   = encoder_path.exists()
    _diag_print(f"Model exists:     {model_exists}")
    _diag_print(f"Tokenizer exists: {tokenizer_exists}")
    _diag_print(f"Encoder exists:   {encoder_exists}")
    if model_exists:
        _diag_print(f"Model file size:  {model_path.stat().st_size:,} bytes")

    if not model_exists:
        msg = f"Model file not found: {model_path}"
        _diag_print(f"ERROR: {msg}")
        _diag_print(sep)
        return (False, msg)
    if not tokenizer_exists:
        msg = f"Tokenizer not found: {tokenizer_path}"
        _diag_print(f"ERROR: {msg}")
        _diag_print(sep)
        return (False, msg)
    if not encoder_exists:
        msg = f"Encoder not found: {encoder_path}"
        _diag_print(f"ERROR: {msg}")
        _diag_print(sep)
        return (False, msg)

    # ── TensorFlow import ─────────────────────────────────────────────────────
    _diag_print("")
    _diag_print("Importing TensorFlow...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorflow as tf
            from tensorflow.keras.models import load_model as _keras_load
            from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad  # noqa: F401
        _diag_print(f"TensorFlow Version: {tf.__version__}")
        try:
            _diag_print(f"Keras Version:      {tf.keras.__version__}")
        except Exception:
            pass
    except ImportError as exc:
        tb = traceback.format_exc()
        msg = f"TensorFlow not installed.\n\nTraceback:\n{tb}"
        _diag_print("ERROR: TensorFlow not installed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"TensorFlow import error: {exc}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: TensorFlow import failed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    _diag_print("")
    _diag_print("Loading tokenizer...")
    try:
        with open(tokenizer_path, "rb") as fh:
            _tokenizer = pickle.load(fh)
        _diag_print(f"Tokenizer Type:    {type(_tokenizer).__name__}")
        _diag_print(f"Word Index:        {_tokenizer.word_index}")
        _diag_print(f"Vocabulary Size:   {len(_tokenizer.word_index)}")
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"Failed to load tokenizer: {exc}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Tokenizer load failed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    # ── Encoder ───────────────────────────────────────────────────────────────
    _diag_print("")
    _diag_print("Loading label encoder...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(encoder_path, "rb") as fh:
                _encoder = pickle.load(fh)
        _diag_print(f"Encoder Type:      {type(_encoder).__name__}")
        _diag_print(f"Encoder Classes:   {list(_encoder.classes_)}")
        _diag_print(f"Number of Classes: {len(_encoder.classes_)}")
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"Failed to load encoder: {exc}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Encoder load failed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    # ── Keras model ───────────────────────────────────────────────────────────
    _diag_print("")
    _diag_print("Loading Keras model (.h5)...")

    # First inspect the file format
    _diag_print("  Inspecting model file format:")
    _diag_print(_inspect_h5_file(model_path))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = _keras_load(str(model_path))
        _diag_print(f"Model Input Shape:  {_model.input_shape}")
        _diag_print(f"Model Output Shape: {_model.output_shape}")
    except Exception as exc:
        tb = traceback.format_exc()
        full_err = f"Failed to load model: {type(exc).__name__}: {exc}"
        msg = f"{full_err}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Keras model load failed – full traceback:")
        _diag_print(tb)

        # Attempt compile=False fallback
        _diag_print("Retrying with compile=False...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model = _keras_load(str(model_path), compile=False)
            _diag_print(f"compile=False succeeded!  Input={_model.input_shape}  Output={_model.output_shape}")
        except Exception as exc2:
            tb2 = traceback.format_exc()
            _diag_print("compile=False also failed:")
            _diag_print(tb2)
            full_msg = (
                f"{full_err}\n\nTraceback:\n{tb}"
                f"\n\nRetry (compile=False) also failed:\n"
                f"{type(exc2).__name__}: {exc2}\n{tb2}"
            )
            _diag_print(sep)
            return (False, full_msg)

    _model_loaded = True
    _diag_print("")
    _diag_print("SUCCESS: AI model fully loaded and ready.")
    _diag_print(sep)
    return (True, "AI Loaded")


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def decode_morse(morse_input: str) -> tuple[str, float, float, str]:
    """
    Decode Morse code to text.

    When the AI model is loaded → TensorFlow character-level inference.
    When not yet loaded         → dictionary fallback.

    Returns:
        (decoded_text, confidence, elapsed_seconds, method)

        - confidence:     0.0–1.0 (0.0 for dictionary fallback)
        - elapsed_seconds: wall-clock time for the inference call
        - method:         "AI" or "Dictionary"
    """
    t0 = time.perf_counter()
    if not _model_loaded:
        text, conf = _fallback_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("Dictionary decode: '%s' → '%s' (%.3fs)", morse_input[:30], text[:30], elapsed)
        return (text, conf, elapsed, "Dictionary")

    try:
        text, conf = _ai_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("AI decode: '%s' → '%s' conf=%.2f (%.3fs)",
                  morse_input[:30], text[:30], conf, elapsed)
        return (text, conf, elapsed, "AI")
    except Exception as exc:
        log.exception("AI inference error.")
        elapsed = time.perf_counter() - t0
        return ("Prediction failed", 0.0, elapsed, "AI (error)")


def _ai_decode(morse_input: str) -> tuple[str, float]:
    """
    Run the trained character-level Keras model.

    Strategy:
        • The model maps ONE Morse symbol sequence → ONE character.
        • Input shape is (batch, 10) – each row is a padded dot/dash sequence.
        • ' / ' in the Morse string denotes a word boundary (space in output).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    word_index = _tokenizer.word_index   # {'.': 1, '-': 2}

    words = morse_input.strip().split(" / ")
    all_codes:   list[str] = []
    word_breaks: list[int] = []

    for w_idx, word in enumerate(words):
        codes = [c for c in word.strip().split(" ") if c]
        all_codes.extend(codes)
        if w_idx < len(words) - 1:
            word_breaks.append(len(all_codes))

    if not all_codes:
        return ("", 0.0)

    # Tokenise: each dot/dash character → integer index
    sequences = [
        [word_index.get(ch, 0) for ch in code]
        for code in all_codes
    ]

    # Pad to MODEL_SEQ_LEN
    padded = pad_sequences(sequences, maxlen=MODEL_SEQ_LEN,
                           padding="post", truncating="post")

    # Batch predict → shape (n_chars, n_classes)
    raw_preds    = _model.predict(padded, verbose=0)
    pred_indices = np.argmax(raw_preds, axis=-1)
    pred_probs   = raw_preds[np.arange(len(raw_preds)), pred_indices]
    confidence   = float(np.mean(pred_probs))

    classes = _encoder.classes_
    chars = [
        str(classes[idx]) if 0 <= idx < len(classes) else "?"
        for idx in pred_indices
    ]

    # Re-insert word boundaries
    result_parts: list[str] = []
    prev = 0
    for brk in sorted(word_breaks):
        result_parts.append("".join(chars[prev:brk]))
        prev = brk
    result_parts.append("".join(chars[prev:]))
    decoded_text = " ".join(result_parts)

    return (decoded_text, confidence)


# ══════════════════════════════════════════════════════════════════════════════
# Text → Morse Encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_text(text: str) -> str:
    """
    Encode plain text to Morse code using the standard ITU dictionary.

    Supported characters: A–Z, 0–9, space, and .,?/-  (plus several others).
    Unsupported characters are silently skipped.

    Returns:
        Morse string: characters separated by single spaces, words by ' / '.
    """
    words = text.upper().strip().split()
    morse_words: list[str] = []
    for word in words:
        codes = [MORSE_CODE_DICT[ch] for ch in word if ch in MORSE_CODE_DICT]
        if codes:
            morse_words.append(" ".join(codes))
    result = " / ".join(morse_words)
    log.debug("Encoded text (len=%d) → morse (len=%d).", len(text), len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_model_loaded() -> bool:
    """Return ``True`` when the AI model is loaded and ready for inference."""
    return _model_loaded


def get_model_info() -> dict:
    """Return a debug summary of the loaded model state."""
    info: dict = {
        "model_loaded":    _model_loaded,
        "model_path":      str(MODEL_PATH),
        "tokenizer_type":  type(_tokenizer).__name__ if _tokenizer else "None",
        "encoder_type":    type(_encoder).__name__   if _encoder   else "None",
        "encoder_classes": (
            len(_encoder.classes_)
            if _encoder and hasattr(_encoder, "classes_") else 0
        ),
    }
    if _model is not None:
        if hasattr(_model, "input_shape"):
            info["input_shape"]  = str(_model.input_shape)
        if hasattr(_model, "output_shape"):
            info["output_shape"] = str(_model.output_shape)
    return info


def _fallback_decode(morse_input: str) -> tuple[str, float]:
    """
    Dictionary-based Morse decoding used when the AI model is not loaded.

    Returns ``(decoded_text, 0.0)`` – confidence is always 0.0 to indicate
    no AI was involved.
    """
    words = morse_input.strip().split(" / ")
    decoded_words: list[str] = []
    for word in words:
        tokens = word.strip().split(" ")
        decoded_words.append(
            "".join(MORSE_REVERSE_DICT.get(t.strip(), "?") for t in tokens if t)
        )
    return (" ".join(decoded_words), 0.0)
