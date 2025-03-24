import sentencepiece as spm


def load_tokenizer(model_path: str = '../bpe_tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
