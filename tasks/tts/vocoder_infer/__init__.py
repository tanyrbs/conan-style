from . import hifigan

try:
    from . import hifigan_nsf
except Exception:
    hifigan_nsf = None
