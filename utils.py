import time

import jax
import jax.numpy as jnp
from flax import nnx


def count_parameters(model):
    params = nnx.state(model, nnx.Param)
    total_params = sum(
        jnp.prod(jnp.array([*x.shape])) for x in jax.tree_util.tree_leaves(params)
    )
    return total_params


def generate_text(model, prompt, length, tokenizer, scheduler):
    # print("debug user prompt : ", prompt)
    # print(
    #     "debug tokenizer output : ", tokenizer.encode(prompt, max_len=length, eval=True)
    # )
    prompt_tokens, text_tokens_len = tokenizer.encode(prompt, max_len=length, eval=True)
    prompt_array = jnp.array(prompt_tokens)[None, ...]
    # print("debug prompt array shape : ", prompt_array.shape)

    for t in reversed(range(scheduler.T)):
        t_batch = jnp.array([t])
        preds = model(prompt_array, t_batch)
        # print("Debug preds shape : ", preds.shape)
        # print(preds)
        pred_tokens = jnp.argmax(preds, axis=-1)
        # print(pred_tokens.shape)
        pred_text = tokenizer.decode(pred_tokens[0])
        print("\r" + pred_text, end="", flush=True)
        time.sleep(0.5)
        prompt_array.at[:, text_tokens_len].set(pred_tokens[:, text_tokens_len])
