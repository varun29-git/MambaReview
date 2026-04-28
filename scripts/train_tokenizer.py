import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB_SIZE = 4096
BATCH_SIZE = 1000

# Special tokens matching what we expect in train.py
SPECIAL_TOKENS = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]

def get_training_corpus():
    # Load the training split in streaming mode
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Take the first 100,000 examples to train the tokenizer.
    # TinyStories is simple enough that this should be plenty to build a 4k vocab.
    iterator = iter(dataset)
    for _ in range(0, 100000, BATCH_SIZE):
        batch = []
        for _ in range(BATCH_SIZE):
            try:
                example = next(iterator)
                batch.append(example["text"])
            except StopIteration:
                break
        if not batch:
            break
        yield batch

def main():
    print(f"Training BPE Tokenizer with vocab size {VOCAB_SIZE} on TinyStories...")
    
    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize the trainer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Save the tokenizer
    save_path = os.path.join(REPO_ROOT, "tokenizer_4k.json")
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    print(f"To use this in your training scripts, update VOCAB_SIZE={VOCAB_SIZE} and tokenizer.from_file('{save_path}')")

if __name__ == "__main__":
    main()
