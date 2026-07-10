"""
LoRaShield – One-shot AI diagnostic script.
Run: .venv\Scripts\python.exe diag_ai.py
"""
import sys, os, traceback, pickle, warnings
sys.path.insert(0, '.')
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
MODEL_PATH     = BASE_DIR / 'models' / 'morse_decoder.h5'
TOKENIZER_PATH = BASE_DIR / 'models' / 'tokenizer.pkl'
ENCODER_PATH   = BASE_DIR / 'models' / 'encoder.pkl'

sep = '-' * 60
print(sep)
print('Loading AI Model...')
print(f'Model path:       {MODEL_PATH}')
print(f'Tokenizer path:   {TOKENIZER_PATH}')
print(f'Encoder path:     {ENCODER_PATH}')
print(f'CWD:              {os.getcwd()}')
print()
print(f'Model exists:     {MODEL_PATH.exists()}')
print(f'Tokenizer exists: {TOKENIZER_PATH.exists()}')
print(f'Encoder exists:   {ENCODER_PATH.exists()}')
if MODEL_PATH.exists():
    print(f'Model file size:  {MODEL_PATH.stat().st_size} bytes')
print(sep)

# ── TensorFlow ────────────────────────────────────────────────────────────────
print('\nImporting TensorFlow...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'   # show ALL TF logs during diagnostic
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    import tensorflow as tf
    print(f'TensorFlow Version: {tf.__version__}')
    print(f'Keras version:      {tf.keras.__version__}')
except Exception:
    print('FAILED to import TensorFlow:')
    traceback.print_exc()
    sys.exit(1)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print('\nLoading tokenizer...')
try:
    with open(TOKENIZER_PATH, 'rb') as fh:
        tok = pickle.load(fh)
    print(f'Tokenizer Type:    {type(tok).__name__}')
    print(f'Word Index:        {tok.word_index}')
    print(f'Vocabulary Size:   {len(tok.word_index)}')
except Exception:
    print('FAILED to load tokenizer:')
    traceback.print_exc()
    sys.exit(1)

# ── Encoder ───────────────────────────────────────────────────────────────────
print('\nLoading encoder...')
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with open(ENCODER_PATH, 'rb') as fh:
            enc = pickle.load(fh)
    print(f'Encoder Type:      {type(enc).__name__}')
    print(f'Encoder Classes:   {list(enc.classes_)}')
    print(f'Number of Classes: {len(enc.classes_)}')
except Exception:
    print('FAILED to load encoder:')
    traceback.print_exc()
    sys.exit(1)

# ── Keras model ───────────────────────────────────────────────────────────────
print('\nLoading Keras model (.h5)...')
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f'Model Input Shape:  {model.input_shape}')
    print(f'Model Output Shape: {model.output_shape}')
    print('\nSUCCESS: Model loaded correctly.')
except Exception:
    print('\nFAILED to load model – full traceback:')
    traceback.print_exc()

    # Try loading as .keras format fallback
    print('\nAttempting native .keras format load...')
    try:
        model2 = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print(f'Native load succeeded!  Input={model2.input_shape}  Output={model2.output_shape}')
    except Exception:
        print('Native format also failed:')
        traceback.print_exc()

    # Inspect what the file actually is
    print('\nInspecting file header (first 20 bytes):')
    with open(MODEL_PATH, 'rb') as fh:
        header = fh.read(20)
    print('  Hex:', header.hex())
    print('  Raw:', header)

    # Check if it is actually a zip/SavedModel
    import zipfile
    if zipfile.is_zipfile(MODEL_PATH):
        print('  -> File IS a zip archive (keras SavedModel format?)')
        with zipfile.ZipFile(MODEL_PATH) as z:
            print('  Contents:', z.namelist()[:10])
    else:
        print('  -> File is NOT a zip archive (legacy HDF5 expected)')

    # Try h5py directly
    print('\nTrying h5py direct open...')
    try:
        import h5py
        with h5py.File(MODEL_PATH, 'r') as hf:
            print(f'  h5py opened OK.  Keys: {list(hf.keys())}')
    except ImportError:
        print('  h5py not installed – install with: pip install h5py')
    except Exception:
        print('  h5py open failed:')
        traceback.print_exc()
