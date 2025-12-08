import jax
import jax.numpy as jnp
from flax import nnx


def precomute_frequencies(dim, max_seq_len, theta=10000, dtype=jnp.float32):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))

    # Create position indices
    t = jnp.arange(max_seq_len, dtype=dtype)

    # Outer product: (max_seq_len, 1) @ (1, dim//2) -> (max_seq_len, dim//2)
    freqs = jnp.outer(t, freqs)

    # Convert to complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = jnp.exp(1j * freqs)

    return freqs_cis
