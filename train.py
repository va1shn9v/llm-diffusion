import time

from datasets import load_dataset

from data import CharTokenizer, create_dataloader
from loss import loss_func
from model import DiffusionLM



def train_one_batch()

tsd = load_dataset("roneneldan/TinyStories", split="train")
tokenizer = CharTokenizer()
dataloader = create_dataloader(
    dataset=tsd, batch_size=64, seq_len=512, tokenizer=tokenizer
)
