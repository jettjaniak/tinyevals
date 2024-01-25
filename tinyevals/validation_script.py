
#%%
# The symbols "#%%" make a new notebook-like cell in VSCode

#%% load dataset from tinyevals\dataset.py
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

#%% load the validation dataset
val_ds = load_clean_dataset("validation")  # 27516 samples

#%%  Load a model
import torch; torch.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#%%  Toxenize the validation set (15 secs)
def tokenize(tokenizer, sample_txt: str) -> list[int]:
    # supposedly this can be different than prepending the bos token id
    return tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0]

val_tok = [tokenize(tokenizer, txt) for txt in tqdm(val_ds)]


#%% (from utils.py) logits and correct probs for one examples

def get_logits(model, sample_tok):
    sample_tok = sample_tok.unsqueeze(0)  # LMs need a batch dimension
    return model(sample_tok).logits[0]    # Just get output for first batch

# example 
sample_tok = val_tok[0]   # validation example 1:  size 136
logits = get_logits(model, sample_tok)  
print(logits.shape)     # (136, 50257)

def get_correct_probs(model, sample_tok):
    # logits: pos, d_vocab
    logits = get_logits(model, sample_tok)
    # pos, d_vocab
    probs = torch.softmax(logits, dim=-1)
    # drop the value for the last position, as we don't know
    # what is the correct next token there
    probs = probs[:-1]
    # out of d_vocab values, take the one that corresponds to the correct next token
    return probs[range(len(probs)), sample_tok[1:]]

sample_tok = val_tok[0]   # validation example 1:  size 136
correct_probs = get_correct_probs(model, sample_tok)  
print(correct_probs.shape)     # torch.Size([135])


# %%  IN SEQUENCE: get correct probs for every token for every sample
# in sequence (loop over each example)
# 1000 examples -> 90s (so the whole dataset, 27000 samples -> 40mins)

all_correct_probs = []  # List to store all probabilities

for sample_tok in tqdm(val_tok[:1000]):
    correct_probs = get_correct_probs(model, sample_tok)
    all_correct_probs.append(correct_probs)
# %%  [work in progress] PARALLEL: get correct probs for every token for every sample
# The idea is to process several samples each time 


# %%  collect several tokenized samples in a matrix
    # convert list of vectors, into matrix (filling the size difference with zeros)
def pad_sequences(vectors, padding_value=0):
    """
    Pads a list of vectors with zeros to make them all the same length.
    
    :param vectors: A list of 1D tensors or lists of varying lengths.
    :param padding_value: The value used for padding shorter sequences.
    :return: A 2D tensor where each row is a zero-padded vector.
    """
    # Find the length of the longest vector
    max_len = max(len(vec) for vec in vectors)

    # Create a padded matrix
    padded_matrix = torch.zeros(len(vectors), max_len, dtype=torch.long)

    # Copy each vector into the matrix, padding with zeros if necessary
    for i, vec in enumerate(vectors):
        length = len(vec)
        padded_matrix[i, :length] = torch.tensor(vec, dtype=torch.long)

    return padded_matrix
# %% get their logits in parallel
def get_logits(model, samples_tok: list[str]) -> torch.Tensor:
    padded_matrix_samples = pad_sequences(samples_tok)
    logits = model(padded_matrix_samples).logits
    return logits   # (num_seqs, max_seq_len, vocab_size)

logits = get_logits(model, val_tok[:10])
print(logits.shape)

# %%  get the correct probs (this can be done with a for loop)
def get_correct_probs(model, samples_tok):
    # logits: seq, pos, d_vocab
    logits = get_logits(model, samples_tok)
    # probs: seq, pos, d_vocab
    probs = torch.softmax(logits, dim=-1)

    # make probs a list of lists of correct token probabilities.
    list_prob =[]
    for i, sample in enumerate(samples_tok):
        valid_length = len(sample) - 1  # Last token doesn't have a next token
        sample_probs = probs[i, :valid_length, :]  # [valid_length, vocab_size]

        # Extract the probabilities of the actual next tokens
        next_tokens = sample[1:valid_length+1]  # Tokens that follow each token in the sequence
        correct_probs = sample_probs[torch.arange(valid_length), next_tokens]

        list_prob.append(correct_probs)
    return list_prob

correct_probs = get_correct_probs(model, val_tok[:10])
#  Víctor's computer crashed with 1000 samples in one time
# correct_probs = get_correct_probs(model, val_tok[:1000])

# %% Next steps:
# Set up a pipeline to process several batches at a time
# How to srote data: in a file or in a variable?