"""
LoRaShield - AI Utility Module (v2)
=====================================
TensorFlow/Keras integration for Morse-to-Text AI decoding.

New Model Architecture (morse_decoder_v2.keras):
    - Tokenizer : Keras Tokenizer  {dot: 1, dash: 2, space: 3}
    - Model     : Bidirectional LSTM Seq2Seq
                  input_shape  = (None, 108)    -- character-level tokens
                  output_shape = (None, 108, 4) -- softmax over
                                                   {0: pad, 1: dot, 2: dash, 3: space}

Inference Strategy (v2):
    1. Tokenise the full noisy Morse string character-by-character.
    2. Pad/truncate to MODEL_SEQ_LEN (108).
    3. model.predict() -> corrected Morse token sequence.
    4. Convert token indices back to Morse chars.
    5. Decode corrected Morse via ITU dictionary.
    6. Confidence = mean(max_softmax) over non-padded positions.

No LabelEncoder / encoder.pkl required.
All events logged via utils.logger AND printed to stdout.
"""

import os
import pickle
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_PATH     = BASE_DIR / "models" / "morse_decoder_v2.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

# Model max sequence length (must match training)
MODEL_SEQ_LEN = 108

# Output vocabulary: index -> Morse character
# 0 = padding (ignored), 1 = dot, 2 = dash, 3 = space
IDX_TO_MORSE = {0: "", 1: ".", 2: "-", 3: " "}

LOW_CONFIDENCE_THRESHOLD = 0.60

_model        = None
_tokenizer    = None
_model_loaded = False

MORSE_CODE_DICT = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",  "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.",  "(": "-.--.", ")": "-.--.-",
    "&": ".-...",  ":": "---...", ";": "-.-.-.",  "=": "-...-",
    "+": ".-.-.",  "-": "-....-", "_": "..--..",  '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
}

MORSE_REVERSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}
SUPPORTED_CHARS    = set(MORSE_CODE_DICT.keys())


def _diag_print(msg: str) -> None:
    """Print msg to stdout AND log at INFO level."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe, flush=True)
    log.info(msg)


# ---------------------------------------------------------------------------
# Levenshtein fuzzy Morse correction
# ---------------------------------------------------------------------------
# ROOT CAUSE NOTE (2026-08-07 diagnostic):
#   The model performs PURE IDENTITY MAPPING on 100% of test inputs.
#   Every output token equals the input token -- no correction occurs.
#   This is a training failure where the model learned the trivial copy
#   solution.  See module docstring for recommended training fixes.
#
# INFERENCE MITIGATION:
#   1. Identity mapping is detected and logged as a warning per call.
#   2. Unrecognized Morse codes (e.g. '.-.-') are fuzzy-corrected via
#      Levenshtein nearest-neighbour (edit distance <= 1).
#   3. VALID but wrong codes (e.g. '--.' when '---' expected) cannot
#      be fixed in inference; they require model retraining.

def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _fuzzy_decode_symbol(code: str) -> tuple:
    """
    Decode one Morse symbol to a character.

    Strategy:
      1. Exact ITU dictionary lookup (always preferred).
      2. Levenshtein nearest-neighbour over all valid codes
         (accepted if edit distance <= 1).
      3. Returns '?' if no match within threshold.

    Returns:
        (character, method_tag, edit_distance)
        method_tag: 'exact' | 'fuzzy' | 'unknown'
    """
    MAX_EDIT_DIST = 1
    code = code.strip()
    if not code:
        return ("", "empty", 0)

    if code in MORSE_REVERSE_DICT:
        return (MORSE_REVERSE_DICT[code], "exact", 0)

    best_char = None
    best_dist = MAX_EDIT_DIST + 1
    best_code = None
    for valid_morse, char in MORSE_REVERSE_DICT.items():
        d = _levenshtein(code, valid_morse)
        if d < best_dist or (d == best_dist and len(valid_morse) == len(code)):
            best_dist = d
            best_char = char
            best_code = valid_morse

    if best_dist <= MAX_EDIT_DIST:
        return (best_char, "fuzzy", best_dist)
    return ("?", "unknown", best_dist)


def load_ai_model(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
    """
    Load the TensorFlow Seq2Seq Keras model and Keras tokenizer.

    No LabelEncoder / encoder.pkl required by the new architecture.

    Returns:
        (True, "AI Loaded") on success.
        (False, full_exception_string) on failure.
    """
    global _model, _tokenizer, _model_loaded

    if _model_loaded:
        _diag_print("AI model already loaded - skipping reload.")
        return (True, "AI Loaded")

    model_path     = Path(model_path)
    tokenizer_path = Path(tokenizer_path)

    sep = "-" * 50
    _diag_print(sep)
    _diag_print("Loading AI Model (v2 - Seq2Seq Denoising)...")
    _diag_print(f"Model path:       {model_path}")
    _diag_print(f"Tokenizer path:   {tokenizer_path}")
    _diag_print(f"Current Working Directory: {os.getcwd()}")

    model_exists     = model_path.exists()
    tokenizer_exists = tokenizer_path.exists()
    _diag_print(f"Model exists:     {model_exists}")
    _diag_print(f"Tokenizer exists: {tokenizer_exists}")
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

    _diag_print("")
    _diag_print("Importing TensorFlow...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa
        _diag_print(f"TensorFlow Version: {tf.__version__}")
        try:
            _diag_print(f"Keras Version:      {tf.keras.__version__}")
        except Exception:
            pass
    except ImportError:
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

    _diag_print("")
    _diag_print("Loading Keras model (.keras)...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = tf.keras.models.load_model(str(model_path), compile=False)
        _diag_print(f"Model Input Shape:  {_model.input_shape}")
        _diag_print(f"Model Output Shape: {_model.output_shape}")
        _diag_print("Model Type:         Seq2Seq Denoising (BiLSTM)")
    except Exception as exc:
        tb = traceback.format_exc()
        full_err = f"Failed to load model: {type(exc).__name__}: {exc}"
        msg = f"{full_err}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Keras model load failed - full traceback:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    _model_loaded = True
    _diag_print("")
    _diag_print("SUCCESS: AI model (v2) fully loaded and ready.")
    _diag_print("  Architecture : Seq2Seq Bidirectional LSTM")
    _diag_print(f"  Input length : {MODEL_SEQ_LEN} (character tokens)")
    _diag_print("  Output vocab : 4 classes (pad / dot / dash / space)")
    _diag_print(sep)
    return (True, "AI Loaded")


def decode_morse(morse_input: str) -> tuple:
    """
    Decode Morse code to plain text.

    When AI model is loaded  -> Seq2Seq denoising + dictionary decode.
    When not yet loaded      -> dictionary fallback (no noise correction).

    Returns:
        (decoded_text, confidence, elapsed_seconds, method)
    """
    t0 = time.perf_counter()
    if not _model_loaded:
        text, conf = _fallback_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("Dictionary decode: '%s' -> '%s' (%.3fs)",
                  morse_input[:30], text[:30], elapsed)
        return (text, conf, elapsed, "Dictionary")

    try:
        text, conf = _ai_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("AI decode: '%s' -> '%s' conf=%.2f (%.3fs)",
                  morse_input[:30], text[:30], conf, elapsed)
        return (text, conf, elapsed, "AI")
    except Exception:
        log.exception("AI inference error.")
        elapsed = time.perf_counter() - t0
        return ("Prediction failed", 0.0, elapsed, "AI (error)")


def _ai_decode(morse_input: str) -> tuple:
    """
    Run the Seq2Seq Keras model and decode the corrected Morse string.

    Full debug trace is emitted at DEBUG log level for every call.
    Identity mapping (model output == input) is detected and warned.
    Unrecognized Morse codes are fuzzy-corrected via Levenshtein distance.

    NOTE: Diagnostic (2026-08-07) confirmed the model performs PURE IDENTITY
    MAPPING on all inputs -- it copies tokens unchanged.  This is a training
    failure.  See module docstring for training recommendations.
    Fuzzy correction mitigates the '?' symptom for invalid codes only.

    Returns:
        (decoded_text, confidence)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    morse_input = morse_input.strip()
    if not morse_input:
        return ("", 0.0)

    # ── Step 1: Tokenise character-by-character ───────────────────────────────
    # Keras Tokenizer maps: '.' -> 1, '-' -> 2, ' ' -> 3.
    # Unknown chars (e.g. '/') are silently dropped (mapped to nothing).
    sequences       = _tokenizer.texts_to_sequences([morse_input])
    token_seq       = sequences[0]
    input_token_len = len(token_seq)

    log.debug("[AI v2] Step 1 | Input: %r", morse_input)
    log.debug("[AI v2] Step 1 | Tokens (%d): %s", input_token_len, token_seq)

    if input_token_len == 0:
        log.warning("[AI v2] Tokenizer produced 0 tokens for input %r", morse_input)
        return ("", 0.0)

    # ── Step 2: Pad to MODEL_SEQ_LEN ─────────────────────────────────────────
    padded = pad_sequences(
        [token_seq],
        maxlen=MODEL_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    log.debug("[AI v2] Step 2 | Padded shape: %s  first20: %s",
              padded.shape, padded[0][:20].tolist())

    # ── Step 3: Predict -> shape (1, MODEL_SEQ_LEN, 4) ───────────────────────
    raw_preds = _model.predict(padded, verbose=0)
    pred_0    = raw_preds[0]                        # (108, 4)

    # ── Step 4: argmax -> token indices ──────────────────────────────────────
    pred_indices = np.argmax(pred_0, axis=-1)       # (108,)
    pred_probs   = np.max(pred_0, axis=-1)          # (108,)

    # ── Step 5 & 6: Reconstruct corrected Morse, trim to input length ─────────
    active_len     = min(input_token_len, MODEL_SEQ_LEN)
    active_indices = pred_indices[:active_len]
    active_probs   = pred_probs[:active_len]

    # ── Identity-mapping detection ────────────────────────────────────────────
    # Count how many output tokens equal the corresponding input token.
    # 100% identity = model learned to copy input (training failure).
    n_identical    = sum(1 for i in range(active_len)
                        if int(active_indices[i]) == token_seq[i])
    identity_ratio = n_identical / active_len if active_len > 0 else 0.0

    if identity_ratio >= 1.0:
        log.warning(
            "[AI v2] IDENTITY MAPPING: output == input for all %d positions. "
            "Model learned to copy input (training failure). "
            "Fuzzy correction active for invalid codes.",
            active_len,
        )
    elif identity_ratio >= 0.8:
        log.warning(
            "[AI v2] Near-identity mapping: %.0f%% of positions unchanged.",
            identity_ratio * 100,
        )

    log.debug(
        "[AI v2] Step 4 | Identity ratio: %.1f%% (%d/%d positions unchanged)",
        identity_ratio * 100, n_identical, active_len,
    )

    # Per-position trace (DEBUG level only)
    for i in range(active_len):
        in_tok  = token_seq[i]
        out_tok = int(active_indices[i])
        in_chr  = IDX_TO_MORSE.get(in_tok, "?")
        out_chr = IDX_TO_MORSE.get(out_tok, "?")
        prob    = float(active_probs[i])
        p4      = [round(float(pred_0[i][j]), 4) for j in range(4)]
        changed = "" if in_tok == out_tok else "  <-- CHANGED"
        log.debug(
            "[AI v2]  pos=%02d  in=%d(%r)  out=%d(%r)  prob=%.4f  p4=%s%s",
            i, in_tok, in_chr, out_tok, out_chr, prob, p4, changed,
        )

    # ── Build corrected Morse string ──────────────────────────────────────────
    corrected_chars = [IDX_TO_MORSE.get(int(i), "") for i in active_indices]
    corrected_morse = "".join(corrected_chars).strip()
    log.debug("[AI v2] Step 5 | Corrected Morse: %r", corrected_morse)

    # ── Step 7: Decode corrected Morse -> plain text ──────────────────────────
    decoded_text = _decode_corrected_morse(corrected_morse)

    # ── Step 8: Confidence ────────────────────────────────────────────────────
    confidence = float(np.mean(active_probs))

    log.debug(
        "[AI v2] Result | noisy=%r -> corrected=%r -> text=%r "
        "| conf=%.3f | identity=%.0f%%",
        morse_input[:40], corrected_morse[:40], decoded_text[:40],
        confidence, identity_ratio * 100,
    )
    return (decoded_text, confidence)


def _decode_corrected_morse(corrected_morse: str) -> str:
    """
    Convert a corrected character-level Morse string to plain text.

    Decoding strategy per symbol:
      1. Exact ITU dictionary lookup (preferred).
      2. Levenshtein fuzzy correction (edit distance <= 1) for invalid codes.
      3. Returns '?' if no valid match found within threshold.

    Word separator:   ' / '  (space-slash-space)
    Letter separator: single space ' '

    NOTE: This fuzzy layer handles cases where the model passes through
    invalid Morse codes unchanged (identity-mapping failure).  It cannot
    fix valid-but-wrong codes (e.g. '--.' vs '---') -- those require
    a retrained model.
    """
    corrected_morse = corrected_morse.strip()
    if not corrected_morse:
        return ""

    if " / " in corrected_morse:
        word_chunks = corrected_morse.split(" / ")
    else:
        word_chunks = [corrected_morse]

    decoded_words = []
    fuzzy_count   = 0
    unknown_count = 0

    for chunk in word_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        letters = []
        for code in chunk.split(" "):
            code = code.strip()
            if not code:
                continue
            char, method, dist = _fuzzy_decode_symbol(code)
            if method == "fuzzy":
                fuzzy_count += 1
                log.debug("[Fuzzy] %r not in dict -> %r (edit_dist=%d)", code, char, dist)
            elif method == "unknown":
                unknown_count += 1
                log.debug("[Unknown] %r has no valid Morse match (edit_dist=%d)", code, dist)
            letters.append(char)
        if letters:
            decoded_words.append("".join(letters))

    if fuzzy_count > 0:
        log.debug("[AI v2] Fuzzy corrections applied: %d symbols.", fuzzy_count)
    if unknown_count > 0:
        log.debug(
            "[AI v2] %d unrecognized symbols output as '?' "
            "(invalid Morse codes the model passed through unchanged).",
            unknown_count,
        )

    return " ".join(decoded_words)


def encode_text(text: str) -> str:
    """
    Encode plain text to Morse code using the standard ITU dictionary.

    Supported: A-Z, 0-9, space, and punctuation (see MORSE_CODE_DICT).
    Unsupported characters are silently skipped.

    Returns:
        Morse string with letters separated by spaces, words by ' / '.
    """
    words = text.upper().strip().split()
    morse_words = []
    for word in words:
        codes = [MORSE_CODE_DICT[ch] for ch in word if ch in MORSE_CODE_DICT]
        if codes:
            morse_words.append(" ".join(codes))
    result = " / ".join(morse_words)
    log.debug("Encoded text (len=%d) -> morse (len=%d).", len(text), len(result))
    return result


def is_model_loaded() -> bool:
    """Return True when the AI model is loaded and ready for inference."""
    return _model_loaded


def get_model_info() -> dict:
    """Return a debug summary of the loaded model state."""
    info = {
        "model_loaded":   _model_loaded,
        "model_path":     str(MODEL_PATH),
        "tokenizer_type": type(_tokenizer).__name__ if _tokenizer else "None",
        "model_version":  "v2 (Seq2Seq Denoising BiLSTM)",
        "seq_len":        MODEL_SEQ_LEN,
        "output_vocab":   IDX_TO_MORSE,
    }
    if _model is not None:
        if hasattr(_model, "input_shape"):
            info["input_shape"]  = str(_model.input_shape)
        if hasattr(_model, "output_shape"):
            info["output_shape"] = str(_model.output_shape)
    return info


def _fallback_decode(morse_input: str) -> tuple:
    """
    Dictionary-based Morse decoding used when the AI model is not loaded.

    Uses fuzzy Levenshtein correction for unrecognized Morse codes.
    Returns (decoded_text, 0.0) -- confidence is always 0.0.
    """
    words = morse_input.strip().split(" / ")
    decoded_words = []
    for word in words:
        tokens = word.strip().split(" ")
        letters = []
        for t in tokens:
            t = t.strip()
            if t:
                char, method, dist = _fuzzy_decode_symbol(t)
                letters.append(char)
        decoded_words.append("".join(letters))
    return (" ".join(decoded_words), 0.0)
