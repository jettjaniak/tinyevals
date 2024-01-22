from datasets import load_dataset
from tqdm.auto import tqdm

def load_txt_dataset(split: str) -> list[str]:
    # checking just startswith, because you can include slice like "train[:1000]"
    assert split.startswith("train") or split.startswith("validation")
    hf_ds = load_dataset(f"jbrinkma/tinystories-v2-clean", split=split)
    dataset = []
    for sample_txt in tqdm(hf_ds["text"]):
        dataset.append(sample_txt)
    return dataset