import jax.numpy as jnp


class CharTokenizer:
    def __init__(self):
        self.pad_token = 0
        self.vocab_size = 128
        self.mask_token = 127

    def encode(self, input, max_len=512, eval=False):
        tokens = [ord(c) % self.vocab_size for c in input]
        total_input_tokens = len(tokens)
        if total_input_tokens < max_len:
            if eval:
                tokens = tokens + [self.mask_token] * (max_len - len(tokens))
            tokens = tokens + [self.pad_token] * (max_len - len(tokens))
        return tokens[:max_len], total_input_tokens

    def decode(self, token_ids):
        tokens_text = []
        # print("Debug token ids : ", token_ids)
        for tid in token_ids:
            if tid == 0:
                tokens_text.append("[PAD_TOKEN]")
            elif tid == 127:
                tokens_text.append("[MASK_TOKEN]")
            else:
                tokens_text.append(chr(tid))
        return "".join(tt for tt in tokens_text)


def create_dataloader(*, dataset, batch_size, seq_len, tokenizer):
    def generator():
        batch = []
        for example in dataset:
            # print(example)
            text = example["text"]
            tokens, _ = tokenizer.encode(text, seq_len)
            batch.append(tokens)

            if len(batch) == batch_size:
                yield jnp.array(batch)
                batch = []

        if batch:
            yield jnp.array(batch, dtype=jnp.float32)

    return generator
