import jax
import jax.numpy as jnp
from flax import nnx


class LearnedPositionalEmbedding(nnx.Module):
    def __init__(self, vocab_size, d_model, rngs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nnx.Embed(self.vocab_size, self.d_model, rngs=rngs)
        self.drop_out = nnx.Dropout(rate=0.3, rngs=rngs)

    def __call__(self, x):
        position_ids = jnp.arange(x.shape[1])
        position_ids = jnp.broadcast_to(position_ids, (x.shape[0], x.shape[1]))
        pos_x = self.embed(position_ids)
        pos_embeddings = self.drop_out(x + pos_x)

        return pos_embeddings


class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, rngs):
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_prod = nnx.MultiHeadAttention(
            num_heads=self.n_heads,
            in_features=self.d_model,
            qkv_features=self.d_model // 2,
            decode=False,
            rngs=rngs,
        )
        self.ln1 = nnx.LayerNorm(num_features=self.d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=self.d_model, rngs=rngs)
        self.mlp1 = nnx.Linear(
            in_features=self.d_model, out_features=self.d_model * 4, rngs=rngs
        )
        self.mlp2 = nnx.Linear(
            in_features=self.d_model * 4, out_features=self.d_model, rngs=rngs
        )

    def __call__(self, x):
        attn_out = self.qkv_prod(
            x
        )  # check this again, whether, I will have to manually calculate hte q,k,v then pass it in as the input to this layer
        x = self.ln1(x + attn_out)
        ffn_out = self.mlp1(x)
        ffn_out = nnx.gelu(ffn_out)
        ffn_out = self.mlp2(ffn_out)
        out_norm = self.ln2(ffn_out + x)
        return out_norm


class DiffusionLM(nnx.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, rngs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos_emb = LearnedPositionalEmbedding(vocab_size, d_model, rngs)
        self.transformer_blocks = nnx.List(
            [TransformerBlock(d_model, n_heads, rngs) for _ in range(n_layers)]
        )
        self.tok_emb = nnx.Embed(self.vocab_size, self.d_model, rngs=rngs)
        self.output = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, x, t, training=False):
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        t_emb = self.get_timestep_embedding(t)
        t_emb = t_emb[:, None, :]
        h = pos_emb + t_emb + token_emb

        for block in self.transformer_blocks:
            h = block(h)

        logits = self.output(h)

        return logits

    def get_timestep_embedding(self, t, max_period=100000):
        half_dim = self.d_model // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

        return embedding


if __name__ == "__main__":
    key = jax.random.key(0)
    test_array = jax.random.randint(key, (8, 64), 0, 127)
    time_steps = jax.random.randint(key, (8,), 0, 10000)
    print(test_array.shape)

    model = DiffusionLM(128, 256, 8, 6, 512, rngs=nnx.Rngs(0))
    pred = model(test_array, time_steps)
    # print(pred)

    # dlm = DiffusionLM()
    # output = dlm(x)
