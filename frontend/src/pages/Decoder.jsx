/**
 * LoRaShield – Morse to Text (AI) Decoder Page
 * =============================================
 * Extended with an interactive Morse Code Typing Error Correction panel.
 *
 * Architecture:
 *   MANUAL INPUT  →  /api/morse/correct-message  (offline similarity matching)
 *                 →  User selects correction
 *                 →  /api/morse/decode             (existing AI / dictionary)
 *
 * The AI model and existing decode flow are NOT changed in any way.
 * Correction is an optional pre-decode step the user can invoke manually.
 */

import { useState, useCallback } from 'react'

const API = 'http://127.0.0.1:8000'

// ─── Colour tokens (mirrors index.css variables) ───────────────────────────
const CLR = {
  success:  'var(--success)',
  warning:  'var(--warning)',
  error:    'var(--error)',
  accent:   'var(--accent)',
  text2:    'var(--text-secondary)',
  card2:    'var(--card-2)',
  border:   'var(--border)',
}

// ─── Small inline components ───────────────────────────────────────────────

/**
 * A single suggestion chip rendered for one invalid Morse token.
 * Clicking it calls onSelect(char, morse, wordIndex, tokenIndex).
 */
function SuggestionChip({ char, morse, score, wordIndex, tokenIndex, onSelect }) {
  const pct = Math.round(score * 100)
  return (
    <button
      id={`suggestion-${wordIndex}-${tokenIndex}-${char}`}
      onClick={() => onSelect(char, morse, wordIndex, tokenIndex)}
      title={`Select ${char} (${morse}) – ${pct}% similar`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '5px 10px',
        border: `1px solid ${CLR.accent}`,
        borderRadius: 6,
        background: 'transparent',
        color: CLR.accent,
        fontFamily: 'var(--font-mono)',
        fontSize: 13,
        cursor: 'pointer',
        transition: 'background 0.15s, color 0.15s',
        marginRight: 6,
        marginBottom: 4,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = CLR.accent
        e.currentTarget.style.color = '#fff'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = CLR.accent
      }}
    >
      <strong>{char}</strong>
      <span style={{ opacity: 0.75 }}>({morse})</span>
      <span style={{ fontSize: 10, opacity: 0.6 }}>{pct}%</span>
    </button>
  )
}

/**
 * Row for one invalid Morse token showing the token, a "Did you mean?" label,
 * and up to 3 suggestion chips.
 */
function CorrectionRow({ token, onSelect }) {
  if (token.is_valid) return null   // Valid tokens are never shown in the panel

  const { input, suggestions, reliable, word_index, token_index } = token

  return (
    <div
      id={`correction-row-${word_index}-${token_index}`}
      style={{
        padding: '10px 12px',
        marginBottom: 8,
        borderRadius: 8,
        border: `1px solid ${reliable ? CLR.warning : CLR.error}`,
        background: reliable
          ? 'rgba(210,153,34,0.06)'
          : 'rgba(248,81,73,0.06)',
      }}
    >
      {/* Invalid token display */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 15,
            fontWeight: 700,
            color: reliable ? CLR.warning : CLR.error,
            letterSpacing: 2,
          }}
        >
          {input}
        </span>
        <span style={{ fontSize: 11, color: CLR.text2 }}>
          {reliable ? '– Unknown Morse code.  Did you mean?' : '– No reliable correction found.'}
        </span>
      </div>

      {/* Suggestion chips */}
      {reliable && suggestions.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {suggestions.map(s => (
            <SuggestionChip
              key={`${s.char}-${s.morse}`}
              char={s.char}
              morse={s.morse}
              score={s.score}
              wordIndex={word_index}
              tokenIndex={token_index}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}

      {/* Not reliable → hint to re-enter */}
      {!reliable && (
        <p style={{ margin: 0, fontSize: 11, color: CLR.error, opacity: 0.8 }}>
          The pattern is too different from any known Morse character.
          Please re-enter this character.
        </p>
      )}
    </div>
  )
}

/**
 * The full correction panel shown below the input textarea when errors exist.
 * Lists every invalid token from the last check result.
 */
function CorrectionPanel({ corrections, currentMorse, onSelect, onDismiss }) {
  if (!corrections || corrections.tokens.length === 0) return null

  const invalidTokens = corrections.tokens.filter(t => !t.is_valid)

  if (invalidTokens.length === 0) {
    // All valid – show a success banner
    return (
      <div
        id="correction-panel-success"
        style={{
          padding: '10px 16px',
          borderRadius: 8,
          border: `1px solid ${CLR.success}`,
          background: 'rgba(63,185,80,0.07)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: 10,
        }}
      >
        <span style={{ color: CLR.success, fontSize: 13, fontWeight: 600 }}>
          ✓  All Morse patterns are valid — ready to decode.
        </span>
        <button
          id="correction-dismiss-btn"
          onClick={onDismiss}
          style={{ background: 'none', border: 'none', color: CLR.text2, cursor: 'pointer', fontSize: 16 }}
        >
          ×
        </button>
      </div>
    )
  }

  return (
    <div
      id="correction-panel"
      style={{
        marginTop: 10,
        padding: 14,
        borderRadius: 10,
        border: `1px solid ${CLR.border}`,
        background: CLR.card2,
      }}
    >
      {/* Panel header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: CLR.text2, letterSpacing: 1 }}>
          MORSE CORRECTION SUGGESTIONS
          <span style={{ marginLeft: 8, color: CLR.warning }}>
            {invalidTokens.length} invalid token{invalidTokens.length > 1 ? 's' : ''}
          </span>
        </span>
        <button
          id="correction-panel-close"
          onClick={onDismiss}
          title="Close correction panel"
          style={{ background: 'none', border: 'none', color: CLR.text2, cursor: 'pointer', fontSize: 16 }}
        >
          ×
        </button>
      </div>

      {/* One CorrectionRow per invalid token */}
      {invalidTokens.map((tok, i) => (
        <CorrectionRow
          key={`${tok.word_index}-${tok.token_index}-${i}`}
          token={tok}
          onSelect={onSelect}
        />
      ))}

      <p style={{ margin: '6px 0 0', fontSize: 10, color: CLR.text2, opacity: 0.7 }}>
        Click a suggestion to replace the token in the input field, then click Decode.
      </p>
    </div>
  )
}

// ─── Main Decoder page ─────────────────────────────────────────────────────

export default function Decoder() {
  // Existing decode state (unchanged)
  const [morse, setMorse]         = useState('')
  const [output, setOutput]       = useState('')
  const [conf, setConf]           = useState(0)
  const [method, setMethod]       = useState('')
  const [elapsed, setElapsed]     = useState(0)
  const [aiLoaded, setAiLoaded]   = useState(false)
  const [aiLoading, setAiLoading] = useState(false)
  const [status, setStatus]       = useState('Ready.')
  const [loading, setLoading]     = useState(false)

  // ── New correction state ────────────────────────────────────────────────
  const [corrections, setCorrections]       = useState(null)   // result of correct-message
  const [correcting, setCorrecting]         = useState(false)  // loading spinner
  const [threshold, setThreshold]           = useState(0.65)   // similarity threshold
  const [showThresholdCtrl, setShowThreshold] = useState(false)

  // ─── Existing handlers (unchanged) ─────────────────────────────────────

  const loadModel = async () => {
    setAiLoading(true)
    setStatus('Loading AI model…')
    try {
      const r = await fetch(`${API}/api/ai/load`, { method: 'POST' })
      const data = await r.json()
      setAiLoaded(data.ok)
      setStatus(data.ok ? 'AI model loaded successfully.' : `Load failed: ${data.message}`)
    } catch (e) {
      setStatus(`Error: ${e.message}`)
    }
    setAiLoading(false)
  }

  const decode = async () => {
    if (!morse.trim()) { setStatus('Error: Morse input is empty.'); return }
    setLoading(true)
    try {
      const r = await fetch(`${API}/api/morse/decode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ morse }),
      })
      const data = await r.json()
      setOutput(data.text)
      setConf(data.confidence)
      setMethod(data.method)
      setElapsed(data.elapsed_ms)
      setStatus(`Decoded ${morse.length} characters.`)
    } catch (e) {
      setStatus(`Error: ${e.message}`)
    }
    setLoading(false)
  }

  const copy = () => {
    if (!output) return
    navigator.clipboard.writeText(output)
    setStatus('Decoded text copied to clipboard.')
  }

  const clear = () => {
    setMorse('')
    setOutput('')
    setConf(0)
    setMethod('')
    setStatus('Cleared.')
    setCorrections(null)
  }

  // ─── New correction handlers ────────────────────────────────────────────

  /**
   * Check the current Morse input for typing errors.
   * Calls POST /api/morse/correct-message and stores the result in state.
   */
  const checkAndCorrect = useCallback(async () => {
    if (!morse.trim()) { setStatus('Error: Morse input is empty.'); return }
    setCorrecting(true)
    setStatus('Checking Morse patterns…')
    setCorrections(null)
    try {
      const r = await fetch(`${API}/api/morse/correct-message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: morse, threshold }),
      })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      setCorrections(data)
      if (data.all_valid) {
        setStatus(`✓ All ${data.token_count} Morse token(s) are valid.`)
      } else {
        const n = data.tokens.filter(t => !t.is_valid).length
        setStatus(`Found ${n} invalid Morse token${n > 1 ? 's' : ''}. Select a correction below.`)
      }
    } catch (e) {
      setStatus(`Correction check failed: ${e.message}`)
    }
    setCorrecting(false)
  }, [morse, threshold])

  /**
   * User clicked a suggestion chip.
   * Calls POST /api/morse/apply-correction and updates the input textarea.
   */
  const handleSuggestionSelect = useCallback(async (char, suggMorse, wordIndex, tokenIndex) => {
    try {
      const r = await fetch(`${API}/api/morse/apply-correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: morse,
          word_index: wordIndex,
          token_index: tokenIndex,
          replacement_morse: suggMorse,
        }),
      })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      const correctedMsg = data.corrected_message

      // Update input field with corrected Morse
      setMorse(correctedMsg)
      setStatus(`Replaced token with ${char} (${suggMorse}). Click Decode to continue.`)

      // Re-run the check on the corrected message so the panel refreshes
      const r2 = await fetch(`${API}/api/morse/correct-message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: correctedMsg, threshold }),
      })
      if (r2.ok) {
        const data2 = await r2.json()
        setCorrections(data2)
        if (data2.all_valid) {
          setStatus(`✓ All tokens corrected – ready to decode (last: ${char} = ${suggMorse}).`)
        }
      }
    } catch (e) {
      setStatus(`Failed to apply correction: ${e.message}`)
    }
  }, [morse, threshold])

  // ─── Derived colours ───────────────────────────────────────────────────
  const confColor = conf >= 80 ? CLR.success : conf >= 60 ? CLR.warning : CLR.error

  // ─── Render ────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">Morse Code → Text (AI)</span>
      </div>
      <div className="page-divider" />
      <p className="page-subtitle">
        Decode Morse code to text using the trained AI model.
        Use <em>Check &amp; Correct</em> to fix typing errors before decoding.
      </p>

      {/* AI model bar – unchanged */}
      <div className="config-bar mb-4">
        <span className="config-label">AI Model</span>
        <div className={`led ${aiLoaded ? 'connected' : 'error'}`} />
        <span style={{ fontSize: 12, color: aiLoaded ? CLR.success : CLR.error }}>
          {aiLoaded ? 'AI Loaded' : 'Not Loaded'}
        </span>
        <button
          id="btn-load-ai"
          className="btn btn-purple btn-sm"
          onClick={loadModel}
          disabled={aiLoading || aiLoaded}
        >
          {aiLoading ? 'Loading…' : aiLoaded ? 'Loaded ✓' : 'Load AI Model'}
        </button>
        <span className="text-muted text-xs" style={{ marginLeft: 'auto' }}>
          models/morse_decoder_v4.keras
        </span>
      </div>

      {/* ── Correction Settings (collapsible) ─────────────────────────── */}
      <div
        style={{
          marginBottom: 10,
          borderRadius: 8,
          border: `1px solid ${CLR.border}`,
          background: CLR.card2,
          overflow: 'hidden',
        }}
      >
        <button
          id="btn-toggle-correction-settings"
          onClick={() => setShowThreshold(v => !v)}
          style={{
            width: '100%',
            background: 'none',
            border: 'none',
            padding: '8px 14px',
            textAlign: 'left',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: CLR.text2,
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: 1,
          }}
        >
          <span>{showThresholdCtrl ? '▾' : '▸'}</span>
          CORRECTION SETTINGS
          <span style={{ marginLeft: 'auto', fontWeight: 400, letterSpacing: 0 }}>
            Threshold: {Math.round(threshold * 100)}%
          </span>
        </button>

        {showThresholdCtrl && (
          <div style={{ padding: '4px 14px 12px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 11, color: CLR.text2, minWidth: 140 }}>
              Similarity threshold
            </span>
            <input
              id="correction-threshold-slider"
              type="range"
              min={50}
              max={95}
              step={5}
              value={Math.round(threshold * 100)}
              onChange={e => setThreshold(Number(e.target.value) / 100)}
              style={{ flex: 1 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: CLR.accent, minWidth: 38 }}>
              {Math.round(threshold * 100)}%
            </span>
            <span style={{ fontSize: 10, color: CLR.text2 }}>
              ↑ stricter  ↓ more permissive
            </span>
          </div>
        )}
      </div>

      {/* ── Input / Output panels – layout unchanged ──────────────────── */}
      <div className="split-pane" style={{ height: 280, marginBottom: 10 }}>
        <div className="flex-col gap-2">
          <p className="input-label">Morse Code Input</p>
          <textarea
            id="morse-input-textarea"
            className="textarea mono grow"
            style={{ resize: 'none', height: '100%' }}
            placeholder=". - . - / . . . / ."
            value={morse}
            onChange={e => {
              setMorse(e.target.value)
              // Clear correction panel when user edits the input
              setCorrections(null)
            }}
          />
        </div>
        <div className="flex-col gap-2">
          <p className="input-label">Decoded Text Output</p>
          <textarea
            id="morse-output-textarea"
            className="textarea grow"
            style={{ resize: 'none', height: '100%' }}
            readOnly
            value={output}
            placeholder="Decoded text appears here…"
          />
        </div>
      </div>

      {/* ── Correction panel (rendered between input and controls) ─────── */}
      <CorrectionPanel
        corrections={corrections}
        currentMorse={morse}
        onSelect={handleSuggestionSelect}
        onDismiss={() => setCorrections(null)}
      />

      {/* ── Controls ──────────────────────────────────────────────────── */}
      <div className="flex gap-2 items-center mb-3" style={{ marginTop: 10 }}>
        <button
          id="btn-decode"
          className="btn btn-primary"
          onClick={decode}
          disabled={loading}
        >
          {loading ? 'Decoding…' : 'Decode'}
        </button>

        {/* New: Check & Correct button */}
        <button
          id="btn-check-correct"
          className="btn"
          onClick={checkAndCorrect}
          disabled={correcting}
          style={{
            background: 'transparent',
            border: `1px solid ${CLR.warning}`,
            color: CLR.warning,
            fontWeight: 700,
            padding: '7px 14px',
            borderRadius: 6,
            cursor: 'pointer',
            fontSize: 13,
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(210,153,34,0.12)' }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          title="Check for Morse typing errors and suggest corrections"
        >
          {correcting ? 'Checking…' : '⚡ Check & Correct'}
        </button>

        <button id="btn-copy" className="btn btn-ghost" onClick={copy}>Copy</button>
        <button id="btn-clear" className="btn btn-danger" onClick={clear}>Clear</button>
        <span className="text-muted text-sm" style={{ marginLeft: 'auto' }}>{status}</span>
      </div>

      {/* ── Confidence bar – unchanged ─────────────────────────────────── */}
      <div className="card2">
        <div className="flex items-center gap-4">
          <span className="config-label" style={{ margin: 0 }}>AI Confidence</span>
          <span style={{ fontSize: 20, fontWeight: 800, fontFamily: 'var(--font-mono)', color: confColor }}>
            {conf.toFixed(1)}%
          </span>
          <span className="text-muted text-xs">
            {method ? `${method}  ·  ${elapsed.toFixed(1)} ms` : 'No prediction yet.'}
          </span>
        </div>
        <div className="confidence-bar-track">
          <div className="confidence-bar-fill" style={{ width: `${conf}%`, background: confColor }} />
        </div>
      </div>
    </div>
  )
}
