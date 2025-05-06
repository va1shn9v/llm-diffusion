from datasets import load_dataset
import jax
import jax.numpy as jnp

ds = load_dataset("roneneldan/TinyStories")
ds = ds.with_format("jax")

print(ds["train"][0])





