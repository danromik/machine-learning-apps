from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from paths import TOKENIZER_DIR
from paths import raw_path

END_OF_TEXT = "<|endoftext|>"
DEFAULT_VOCAB_SIZE = 8192

def tokenizer_path(vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    return TOKENIZER_DIR / f"bpe-{vocab_size}.json"

def train_tokenizer(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    split: str = "valid",
    force: bool = False
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on one corpus split and save it"""
    dest = tokenizer_path(vocab_size)

    if dest.exists() and not force:
        print(f"{dest.name} already present")
        return Tokenizer.from_file(str(dest))

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train([str(raw_path(split))], trainer)
    tok.save(str(dest))
    print(f"trained {vocab_size}-token BPE -> {dest.name}")
    return tok

def load_tokenizer(vocab_size: int = DEFAULT_VOCAB_SIZE) -> Tokenizer:
    path = tokenizer_path(vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"no tokenizer at {path}; run tokenizer.py first")
    return Tokenizer.from_file(str(path))


if __name__ == "__main__":
    tok = train_tokenizer()

    sample = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(sample).ids

    print(f"\nvocab size: {tok.get_vocab_size()}")
    print(f"eot id:     {tok.token_to_id(END_OF_TEXT)}")
    print(f"\ntext:   {sample}")
    print(f"ids:     {ids}")
    print(f"tokens: {tok.encode(sample).tokens}")
    print(f"decoded: {tok.decode(ids)}")
