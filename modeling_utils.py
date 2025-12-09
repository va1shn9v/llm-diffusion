from typing import Tuple

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


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)
        dtype: Output dtype

    Returns:
        Tuple of rotated (query, key) tensors with same shapes as input
    """
    # Reshape x into complex numbers by pairing adjacent dimensions
    # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_heads, head_dim//2)
    xq_complex = xq.reshape(*xq.shape[:-1], -1, 2)
    xq_complex = jax.lax.complex(xq_complex[..., 0], xq_complex[..., 1])

    xk_complex = xk.reshape(*xk.shape[:-1], -1, 2)
    xk_complex = jax.lax.complex(xk_complex[..., 0], xk_complex[..., 1])

    # Reshape freqs_cis to broadcast correctly
    # (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis[None, :, None, :]

    # Apply rotation by complex multiplication
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    # Convert back to real numbers by interleaving real and imaginary parts
    xq_out = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1)
    xq_out = xq_out.reshape(*xq.shape)

    xk_out = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1)
    xk_out = xk_out.reshape(*xk.shape)

    return xq_out.astype(dtype), xk_out.astype(dtype)
