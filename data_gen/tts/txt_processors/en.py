import re
import unicodedata

try:
    from g2p_en import G2p
    from g2p_en.expand import normalize_numbers
    from nltk import pos_tag
    from nltk.tokenize import TweetTokenizer
    _EN_TXT_PROCESSOR_IMPORT_ERROR = None
except Exception as exc:  # optional dependency: only required for English text processing
    G2p = None
    normalize_numbers = None
    pos_tag = None
    TweetTokenizer = None
    _EN_TXT_PROCESSOR_IMPORT_ERROR = exc

from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from utils.text.text_encoder import PUNCS, is_sil_phoneme


def _require_en_text_processor_dependencies():
    if (
        G2p is None
        or normalize_numbers is None
        or pos_tag is None
        or TweetTokenizer is None
    ):
        raise ImportError(
            "English text processing requires optional dependencies "
            "'g2p_en' and 'nltk' (including the tagger / cmudict resources)."
        ) from _EN_TXT_PROCESSOR_IMPORT_ERROR


class EnG2p(G2p if G2p is not None else object):
    word_tokenize = TweetTokenizer().tokenize if TweetTokenizer is not None else None

    def __init__(self, *args, **kwargs):
        _require_en_text_processor_dependencies()
        super().__init__(*args, **kwargs)
        if self.word_tokenize is None:
            type(self).word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        _require_en_text_processor_dependencies()
        # preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]


@register_txt_processors('en')
class TxtProcessor(BaseTxtProcessor):
    g2p = None

    @classmethod
    def _get_g2p(cls):
        if cls.g2p is None:
            try:
                cls.g2p = EnG2p()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the English G2P pipeline. "
                    "Install 'g2p_en' and the required NLTK resources before "
                    "running English text preprocessing."
                ) from exc
        return cls.g2p

    @staticmethod
    def preprocess_text(text):
        _require_en_text_processor_dependencies()
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls._get_g2p()(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt
