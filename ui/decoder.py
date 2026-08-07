"""
LoRaShield – Morse to Text (AI) Decoder Page
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from utils.ai import load_ai_model, decode_morse, is_model_loaded


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'warning': '#D29922', 'error': '#F85149',
    'purple': '#8B5CF6', 'input_bg': '#0D1117',
}


def _btn(parent, text, command, color=None, fg=None, **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  font=('Segoe UI', 10, 'bold'), bd=0, padx=16, pady=8,
                  cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


class DecoderPage(tk.Frame):
    """Morse to Text AI Decoder page."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='MORSE CODE  →  TEXT (AI)', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        tk.Label(self,
                 text='Decode Morse code to text using the trained AI model.',
                 bg=C['bg'], fg=C['text2'], font=('Segoe UI', 10)
                 ).pack(anchor='w', padx=28, pady=(10, 0))

        # ── AI Model Status bar ───────────────────────────────────────────────
        model_bar = tk.Frame(self, bg=C['card2'],
                             highlightbackground=C['border'], highlightthickness=1)
        model_bar.pack(fill='x', padx=28, pady=(14, 0))

        ml_inner = tk.Frame(model_bar, bg=C['card2'])
        ml_inner.pack(fill='x', padx=16, pady=10)

        tk.Label(ml_inner, text='AI MODEL', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        self.model_status_canvas = tk.Canvas(ml_inner, width=10, height=10,
                                             bg=C['card2'], bd=0, highlightthickness=0)
        self.model_status_canvas.pack(side='left', padx=(12, 6))
        self._led = self.model_status_canvas.create_oval(1, 1, 9, 9,
                                                         fill=C['error'], outline='')

        self.model_status_var = tk.StringVar(value='Not Loaded')
        tk.Label(ml_inner, textvariable=self.model_status_var,
                 bg=C['card2'], fg=C['error'], font=('Segoe UI', 9)
                 ).pack(side='left', padx=(0, 20))

        _btn(ml_inner, '  LOAD AI MODEL  ', self._load_model,
             C['purple']).pack(side='left')

        tk.Label(ml_inner, text='Model path: models/morse_decoder_v2.keras',
                 bg=C['card2'], fg=C['text3'], font=('Segoe UI', 8)).pack(side='right')

        # ── Main panels ───────────────────────────────────────────────────────
        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=20, pady=14)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # Input
        tk.Label(body, text='MORSE CODE INPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')
                 ).grid(row=0, column=0, sticky='w', padx=8, pady=(0, 4))

        in_frame = tk.Frame(body, bg=C['card'],
                            highlightbackground=C['border'], highlightthickness=1)
        in_frame.grid(row=1, column=0, sticky='nsew', padx=8)

        self.txt_input = tk.Text(in_frame, bg=C['card'], fg=C['success'],
                                 font=('Consolas', 13), bd=0, padx=14, pady=12,
                                 insertbackground=C['accent'], wrap='word',
                                 selectbackground=C['accent'],
                                 highlightthickness=0)
        sb_in = ttk.Scrollbar(in_frame, orient='vertical', command=self.txt_input.yview)
        self.txt_input.configure(yscrollcommand=sb_in.set)
        self.txt_input.pack(side='left', fill='both', expand=True)
        sb_in.pack(side='right', fill='y')

        # Output
        tk.Label(body, text='DECODED TEXT OUTPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')
                 ).grid(row=0, column=1, sticky='w', padx=8, pady=(0, 4))

        out_frame = tk.Frame(body, bg=C['input_bg'],
                             highlightbackground=C['border'], highlightthickness=1)
        out_frame.grid(row=1, column=1, sticky='nsew', padx=8)

        self.txt_output = tk.Text(out_frame, bg=C['input_bg'], fg=C['text1'],
                                  font=('Segoe UI', 12), bd=0, padx=14, pady=12,
                                  state='disabled', wrap='word',
                                  selectbackground=C['accent'],
                                  highlightthickness=0)
        sb_out = ttk.Scrollbar(out_frame, orient='vertical', command=self.txt_output.yview)
        self.txt_output.configure(yscrollcommand=sb_out.set)
        self.txt_output.pack(side='left', fill='both', expand=True)
        sb_out.pack(side='right', fill='y')

        # ── Controls ─────────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=C['bg'])
        ctrl.pack(fill='x', padx=28, pady=(0, 12))

        _btn(ctrl, '   DECODE   ', self._decode, C['accent']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   COPY   ', self._copy, '#21262D', C['text1']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   CLEAR   ', self._clear, C['error']).pack(side='left')

        # Confidence
        conf_frame = tk.Frame(self, bg=C['card2'],
                              highlightbackground=C['border'], highlightthickness=1)
        conf_frame.pack(fill='x', padx=28, pady=(0, 20))

        ci = tk.Frame(conf_frame, bg=C['card2'])
        ci.pack(fill='x', padx=16, pady=10)

        tk.Label(ci, text='AI CONFIDENCE', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        self.conf_var = tk.StringVar(value='0.00%')
        self.conf_label = tk.Label(ci, textvariable=self.conf_var, bg=C['card2'],
                                   fg=C['text2'], font=('Consolas', 13, 'bold'))
        self.conf_label.pack(side='left', padx=16)

        self.pred_var = tk.StringVar(value='No prediction yet.')
        self.pred_label = tk.Label(ci, textvariable=self.pred_var, bg=C['card2'],
                                   fg=C['text3'], font=('Segoe UI', 9))
        self.pred_label.pack(side='left')

        self.status_var = tk.StringVar(value='Ready.')
        self.status_label = tk.Label(conf_frame, textvariable=self.status_var,
                                     bg=C['card2'], fg=C['text3'], font=('Segoe UI', 9))
        self.status_label.pack(anchor='e', padx=16, pady=(0, 6))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load the AI model in a background thread and update UI with the real result."""
        self.model_status_var.set('Loading...')
        self.model_status_canvas.itemconfig(self._led, fill=C['warning'])
        self.status_var.set('Loading AI model – please wait...')

        def _do():
            ok, msg = load_ai_model()

            def _update():
                if ok:
                    # ── SUCCESS ────────────────────────────────────────────
                    self.model_status_var.set('AI Loaded')
                    self.model_status_canvas.itemconfig(self._led, fill=C['success'])
                    # Update the label colour to green on success
                    for w in self.model_status_canvas.master.winfo_children():
                        if isinstance(w, tk.Label) and w.cget('textvariable') != '':
                            try:
                                if str(w.cget('textvariable')) == str(self.model_status_var):
                                    w.config(fg=C['success'])
                            except Exception:
                                pass
                    self.status_var.set('AI model loaded successfully.')
                else:
                    # ── FAILURE: show the REAL reason ───────────────────────────
                    # First line of the error for the compact label
                    first_line = msg.split('\n')[0] if msg else 'Unknown error'
                    short = first_line[:60] + ('...' if len(first_line) > 60 else '')

                    self.model_status_var.set(f'Failed: {short}')
                    self.model_status_canvas.itemconfig(self._led, fill=C['error'])
                    # Make the status-label red
                    for w in self.model_status_canvas.master.winfo_children():
                        if isinstance(w, tk.Label):
                            try:
                                if str(w.cget('textvariable')) == str(self.model_status_var):
                                    w.config(fg=C['error'])
                            except Exception:
                                pass

                    # Full error in the bottom status bar
                    self.status_var.set(f'Load failed: {first_line}')

                    # ── Popup showing the FULL exception ────────────────────────
                    messagebox.showerror(
                        'Failed to Load AI Model',
                        f'Failed to load model:\n\n{msg}',
                        parent=self,
                    )

            self.after(0, _update)

        threading.Thread(target=_do, daemon=True).start()

    def _decode(self):
        morse = self.txt_input.get('1.0', 'end').strip()
        if not morse:
            self.status_var.set('Error: Morse input is empty.')
            return

        # decode_morse returns (text, confidence, elapsed_seconds, method)
        result = decode_morse(morse)
        text, confidence, elapsed, method = result

        # Write decoded output
        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.insert('1.0', text)
        self.txt_output.configure(state='disabled')

        pct = confidence * 100
        self.conf_var.set(f'{pct:.2f}%')

        elapsed_ms = elapsed * 1000

        if is_model_loaded():
            # ── Color code confidence ──────────────────────────────────────
            if text == 'Prediction failed':
                self.conf_label.config(fg=C['error'])
                self.pred_label.config(fg=C['error'])
                self.pred_var.set(f'Prediction failed  |  {method}  |  {elapsed_ms:.1f} ms')
                self.status_var.set('AI inference failed. Check console for details.')
            elif pct < 60.0:
                self.conf_label.config(fg=C['warning'])
                self.pred_label.config(fg=C['warning'])
                self.pred_var.set(f'Low confidence  |  {method}  |  {elapsed_ms:.1f} ms')
                self.status_var.set(f'Decoded {len(morse)} chars — low confidence ({pct:.1f}%).')
            else:
                self.conf_label.config(fg=C['success'])
                self.pred_label.config(fg=C['text3'])
                self.pred_var.set(f'AI prediction  |  {pct:.1f}% confidence  |  {elapsed_ms:.1f} ms')
                self.status_var.set(f'Decoded {len(morse)} characters successfully.')
        else:
            self.conf_label.config(fg=C['text2'])
            self.pred_label.config(fg=C['text3'])
            self.pred_var.set(
                f'Dictionary fallback  |  {elapsed_ms:.1f} ms  '
                f'(Load AI Model for real inference).'
            )
            self.status_var.set(f'Decoded {len(morse)} characters using dictionary.')

    def _copy(self):
        out = self.txt_output.get('1.0', 'end').strip()
        if out:
            self.clipboard_clear()
            self.clipboard_append(out)
            self.status_var.set('Decoded text copied to clipboard.')

    def _clear(self):
        self.txt_input.delete('1.0', 'end')
        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.configure(state='disabled')
        self.conf_var.set('0.00%')
        self.pred_var.set('No prediction yet.')
        self.status_var.set('Cleared.')
