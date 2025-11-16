import jax
import jax.numpy as jnp
from flax import linen as nn


class LearnedPositionalEmbedding(nn.Module):
    vocab_size = 128
    d_model = 256

    @nn.compact
    def __call__(self, x):
        position_ids = jnp.arange(x.shape[1])
        position_ids = jnp.broadcast_to(position_ids, x.shape)
        pos_x = nn.Embed(self.vocab_size, self.d_model)(position_ids)
        pos_embeddings = nn.Dropout(rate=0.3)(x + pos_x)

        return pos_embeddings


class TransformerBlock(nn.Module):
    d_model = 256
    n_heads = 8

    @nn.compact
    def __call__(self, x):
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.d_model
        )(x, x)
        x = nn.LayerNorm()(x + attn_out)

        ffn_out = nn.Dense(self.d_model * 4)(x)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dense(self.d_model)(ffn_out)
        x = nn.LayerNorm()(x + attn_out)

        return x


class DiffusionLM(nn.Module):
    vocab_size = 128
    d_model = 128
    n_heads = 8
    n_layers = 6
    max_seq_len = 512

    @nn.compact
    def __call__(self, x, t, training=False):
        token_emb = nn.Embed(self.vocab_size, self.d_model)(x)
        pos_emb = LearnedPositionalEmbedding()(token_emb)
        t_emb = self.get_timestep_embedding(t)
        t_emb = t_emb[:, None, :]

        h = pos_emb + t_emb

        for _ in range(self.n_layers):
            h = TransformerBlock()(h)

        logits = nn.Dense(self.vocab_size)(h)

        return logits

    def get_timestep_embedding(self, t, max_period=100000):
        half_dim = self.d_model // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

        return embedding


if __name__ == "__main__":
    key = jax.random.key(0)
    test_array = jax.random.randint(key, (8, 64, 256), 0, 127)
    print(test_array)

    # dlm = DiffusionLM()
    # output = dlm(x)
