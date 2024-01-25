import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

def load_clean_dataset(split: str, tokenized: bool = False) -> list[str]:
    # checking just startswith, because you can include slice like "train[:1000]"
    assert split.startswith("train") or split.startswith("validation")
    hf_ds = load_dataset(f"jbrinkma/tinystories-v2-clean{'-tokenized' if tokenized else ''}", split=split)
    dataset = []
    for sample in tqdm(hf_ds["tokens" if tokenized else "story"]):
        dataset.append(sample)
    return dataset

def token_map(tokenized_dataset: list[str]) -> dict[int, list[tuple[int, int]]]:
    unique_tokens = np.unique(tokenized_dataset)
    mapping = {}
    for token in tqdm(unique_tokens):
        indices = np.where(tokenized_dataset == token)
        mapping[token] = list(zip(*indices))

    return mapping
