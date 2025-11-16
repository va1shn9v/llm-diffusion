import time

from datasets import load_dataset

from data import CharTokenizer, create_dataloader

tsd = load_dataset("roneneldan/TinyStories", split="train")
tokenizer = CharTokenizer()
dataloader = create_dataloader(
    dataset=tsd, batch_size=64, seq_len=512, tokenizer=tokenizer
)

i = 10
start_time = time.time()
while i > 0:
    batch = next(iter(dataloader()))
    i = -1
end_time = time.time()
print(end_time - start_time)
