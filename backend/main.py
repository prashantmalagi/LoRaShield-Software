"""
LoRaShield – FastAPI Backend
==============================
Exposes all LoRaShield Python logic as a REST + WebSocket API
consumed by the React frontend.

Run with:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import serial_comm
from utils.encryption import encrypt_message, decrypt_message, generate_key, is_crypto_available
from utils.history import (
    load_history, save_history, delete_history_record,
    clear_history, export_history, search_history, record_count,
)
from utils.ai import encode_text, decode_morse, load_ai_model, is_model_loaded, get_model_info
from utils.morse_correction import (
    correct_morse_character,
    correct_morse_message,
    suggest_morse_corrections,
    apply_correction,
    set_default_threshold,
    _DEFAULT_THRESHOLD as CORRECTION_DEFAULT_THRESHOLD,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="LoRaShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared app state (mirrors Tkinter app_state) ──────────────────────────────
_app_state: dict[str, Any] = {
    "com_port":     "",
    "enc_key":      "LoRaShield2026",
    "ai_loaded":    False,
    "connected":    False,
    "baud_rate":    9600,
    "auto_connect": False,
}

_SETTINGS_FILE = ROOT / "data" / "settings.json"

# Load settings on startup
def _load_settings_on_boot():
    if _SETTINGS_FILE.exists():
        try:
            with _SETTINGS_FILE.open() as fh:
                data = json.load(fh)
            _app_state["com_port"]     = data.get("com_port", "")
            _app_state["enc_key"]      = data.get("enc_key", "LoRaShield2026")
            _app_state["baud_rate"]    = int(data.get("baud_rate", 9600))
            _app_state["auto_connect"] = bool(data.get("auto_connect", False))
        except Exception:
            pass

_load_settings_on_boot()

# ── Active WebSocket clients ───────────────────────────────────────────────────
_ws_clients: list[WebSocket] = []


async def _broadcast(payload: dict):
    """Send a JSON message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _serial_rx_callback(data: str):
    """Called on background thread when serial data arrives; schedule WS broadcast."""
    key = _app_state.get("enc_key", "")
    if key and ":" in data:
        plaintext, err = decrypt_message(data, key)
        if err:
            payload = {"type": "rx", "cipher": data, "plain": data, "enc_status": "Decrypt Error"}
        else:
            payload = {"type": "rx", "cipher": data, "plain": plaintext, "enc_status": "AES-256 CBC"}
    else:
        payload = {"type": "rx", "cipher": "", "plain": data, "enc_status": "None"}

    # Schedule the coroutine-based broadcast from a sync thread
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)


# ═══════════════════════════════════════════════════════════════════════════════
# Serial Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/serial/ports")
def get_ports():
    return {"ports": serial_comm.list_com_ports()}


class ConnectRequest(BaseModel):
    port: str
    baud_rate: int = 9600


@app.post("/api/serial/connect")
def connect_serial(req: ConnectRequest):
    ok, msg = serial_comm.connect_esp32(req.port, req.baud_rate)
    if ok:
        _app_state["com_port"] = req.port
        _app_state["baud_rate"] = req.baud_rate
        _app_state["connected"] = True
        serial_comm.start_receive_loop(
            callback=_serial_rx_callback,
            error_callback=lambda err: asyncio.run_coroutine_threadsafe(
                _broadcast({"type": "error", "message": err}),
                asyncio.get_event_loop()
            )
        )
    return {"ok": ok, "message": msg}


@app.post("/api/serial/disconnect")
def disconnect_serial():
    serial_comm.stop_receive_loop()
    ok, msg = serial_comm.disconnect_esp32()
    if ok:
        _app_state["connected"] = False
    return {"ok": ok, "message": msg}


@app.get("/api/serial/status")
def serial_status():
    return {
        "connected": serial_comm.is_connected(),
        "port": _app_state.get("com_port", ""),
        **serial_comm.get_stats(),
    }


class SendRequest(BaseModel):
    message: str
    use_encryption: bool = True


@app.post("/api/serial/send")
async def send_message(req: SendRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Message cannot be empty")

    key = _app_state.get("enc_key", "") if req.use_encryption else ""

    if key:
        encrypted, err = encrypt_message(msg, key)
        if err:
            raise HTTPException(500, f"Encryption failed: {err}")
        payload = encrypted
        enc_status = "AES-256 CBC"
    else:
        payload = msg
        encrypted = ""
        enc_status = "None"

    ok, result = serial_comm.send_lora(payload)

    if ok:
        # Broadcast TX to terminal
        await _broadcast({
            "type": "tx",
            "plain": msg,
            "cipher": encrypted,
            "enc_status": enc_status,
        })
        # Save to history
        save_history({
            "direction": "TX",
            "plain_text": msg,
            "morse_code": "",
            "encrypted_data": encrypted,
            "sender": "LOCAL",
            "receiver": _app_state.get("com_port", "Remote"),
            "encryption_status": enc_status,
        })

    return {"ok": ok, "message": result}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket – Live Serial Terminal
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/serial")
async def ws_serial(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    # Send current connection state
    await websocket.send_json({
        "type": "status",
        "connected": serial_comm.is_connected(),
        "port": _app_state.get("com_port", ""),
    })
    try:
        while True:
            # Keep the socket alive; actual data is pushed via _serial_rx_callback
            await asyncio.sleep(2)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# Morse Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class EncodeRequest(BaseModel):
    text: str


class DecodeRequest(BaseModel):
    morse: str


@app.post("/api/morse/encode")
def morse_encode(req: EncodeRequest):
    if not req.text:
        raise HTTPException(400, "Text cannot be empty")
    result = encode_text(req.text)
    return {"morse": result, "input_length": len(req.text)}


@app.post("/api/morse/decode")
def morse_decode(req: DecodeRequest):
    if not req.morse:
        raise HTTPException(400, "Morse input cannot be empty")
    text, confidence, elapsed, method = decode_morse(req.morse)
    return {
        "text": text,
        "confidence": round(confidence * 100, 2),
        "elapsed_ms": round(elapsed * 1000, 1),
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Morse Correction Endpoints  (manual typing error correction)
# These endpoints are completely separate from the AI decoder.
# They use offline Levenshtein edit-distance matching against the ITU
# Morse dictionary and require no internet connection or loaded model.
# ═══════════════════════════════════════════════════════════════════════════════

class CorrectCharRequest(BaseModel):
    morse: str
    max_suggestions: int = 3
    threshold: float = CORRECTION_DEFAULT_THRESHOLD


class CorrectMessageRequest(BaseModel):
    message: str
    threshold: float = CORRECTION_DEFAULT_THRESHOLD


class ApplyCorrectionRequest(BaseModel):
    message: str
    word_index: int
    token_index: int
    replacement_morse: str


@app.post("/api/morse/correct-char")
def morse_correct_char(req: CorrectCharRequest):
    """
    Validate a single Morse token and return correction suggestions.

    Returns:
        is_valid:    True when the input is an exact ITU Morse code.
        char:        Decoded character for a valid input, else null.
        suggestions: Up to max_suggestions closest valid patterns.
        reliable:    False when no suggestions meet the similarity threshold.
    """
    if not req.morse:
        raise HTTPException(400, "Morse input cannot be empty")
    result = correct_morse_character(req.morse, threshold=req.threshold)
    return result


@app.post("/api/morse/correct-message")
def morse_correct_message(req: CorrectMessageRequest):
    """
    Validate every token in a full Morse message and return per-token results.

    Word separator is ' / '; letter separator is a single space.
    Each token is checked independently; valid tokens are returned as-is.

    Returns:
        tokens:      List of per-token result objects.
        has_errors:  True when at least one token is not a valid Morse code.
        all_valid:   Inverse of has_errors.
        word_count:  Number of slash-separated word groups.
        token_count: Total number of Morse tokens processed.
    """
    if not req.message:
        raise HTTPException(400, "Message cannot be empty")
    result = correct_morse_message(req.message, threshold=req.threshold)
    return result


@app.post("/api/morse/apply-correction")
def morse_apply_correction(req: ApplyCorrectionRequest):
    """
    Replace one Morse token in a full message and return the corrected string.

    Called after the user clicks a suggestion chip in the GUI.
    """
    if not req.message:
        raise HTTPException(400, "Message cannot be empty")
    corrected = apply_correction(
        req.message,
        req.word_index,
        req.token_index,
        req.replacement_morse,
    )
    return {"corrected_message": corrected}


@app.get("/api/morse/correction-threshold")
def get_correction_threshold():
    """Return the current default similarity threshold for the correction engine."""
    return {"threshold": CORRECTION_DEFAULT_THRESHOLD}


# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ai/load")
def ai_load():
    ok, msg = load_ai_model()
    _app_state["ai_loaded"] = ok
    return {"ok": ok, "message": msg}


@app.get("/api/ai/status")
def ai_status():
    return {
        "loaded": is_model_loaded(),
        **get_model_info(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Encryption Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class EncryptRequest(BaseModel):
    message: str
    key: str


class DecryptRequest(BaseModel):
    encrypted: str
    key: str


@app.post("/api/encrypt/encrypt")
def api_encrypt(req: EncryptRequest):
    if not is_crypto_available():
        raise HTTPException(503, "pycryptodome not installed")
    enc, err = encrypt_message(req.message, req.key)
    if err:
        raise HTTPException(400, err)
    return {"encrypted": enc}


@app.post("/api/encrypt/decrypt")
def api_decrypt(req: DecryptRequest):
    if not is_crypto_available():
        raise HTTPException(503, "pycryptodome not installed")
    plain, err = decrypt_message(req.encrypted, req.key)
    if err:
        raise HTTPException(400, err)
    return {"plaintext": plain}


@app.post("/api/encrypt/generate-key")
def api_generate_key():
    key = generate_key(32)
    return {"key": key}


@app.get("/api/encrypt/status")
def api_crypto_status():
    return {"available": is_crypto_available()}


# ═══════════════════════════════════════════════════════════════════════════════
# History Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/history")
def api_get_history(q: str = ""):
    if q:
        return {"records": search_history(q)}
    return {"records": load_history()}


@app.post("/api/history")
def api_save_history(record: dict):
    ok, msg = save_history(record)
    if not ok:
        raise HTTPException(500, msg)
    return {"ok": True, "message": msg}


@app.delete("/api/history/clear")
def api_clear_history():
    ok, msg = clear_history()
    return {"ok": ok, "message": msg}


@app.get("/api/history/export")
def api_export_history():
    ok, msg = export_history()
    return {"ok": ok, "message": msg}


@app.delete("/api/history/{index}")
def api_delete_history(index: int):
    ok, msg = delete_history_record(index)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# ═══════════════════════════════════════════════════════════════════════════════
# Settings Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/settings")
def api_get_settings():
    if _SETTINGS_FILE.exists():
        try:
            with _SETTINGS_FILE.open() as fh:
                return json.load(fh)
        except Exception:
            pass
    return _app_state


class SettingsPayload(BaseModel):
    com_port: str = ""
    baud_rate: str = "9600"
    enc_key: str = ""
    auto_connect: bool = False
    ai_threshold: float = 0.75
    theme: str = "Dark"


@app.post("/api/settings")
def api_save_settings(settings: SettingsPayload):
    data = settings.model_dump()
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_FILE.open("w") as fh:
        json.dump(data, fh, indent=2)
    # Update app state
    _app_state["com_port"]     = data.get("com_port", "")
    _app_state["enc_key"]      = data.get("enc_key", "")
    _app_state["baud_rate"]    = int(data.get("baud_rate", 9600))
    _app_state["auto_connect"] = bool(data.get("auto_connect", False))
    return {"ok": True, "message": "Settings saved."}


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard / Stats Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard")
def api_dashboard():
    stats = serial_comm.get_stats()
    return {
        "connected":       serial_comm.is_connected(),
        "port":            _app_state.get("com_port", ""),
        "msgs_sent":       stats.get("msgs_sent", 0),
        "msgs_received":   stats.get("msgs_received", 0),
        "last_comm_time":  stats.get("last_comm_time", "Never"),
        "ai_loaded":       is_model_loaded(),
        "crypto_available": is_crypto_available(),
        "record_count":    record_count(),
    }
