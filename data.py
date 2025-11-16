import jax.numpy as jnp


class CharTokenizer:
    def __init__(self):
        self.pad_token = 0
        self.vocab_size = 128
        self.mask_token = 127

    def encode(self, input, max_len=512):
        tokens = [ord(c) % self.vocab_size for c in input]

        if len(tokens) < max_len:
            tokens = tokens + [self.pad_token] * (max_len - len(tokens))
        return tokens[:max_len]


def create_dataloader(*, dataset, batch_size, seq_len, tokenizer):
    def generator():
        batch = []
        for example in dataset:
            # print(example)
            text = example["text"]
            tokens = tokenizer.encode(text, seq_len)
            batch.append(tokens)

            if len(batch) == batch_size:
                yield batch

        if batch:
            yield batch

    return generator
