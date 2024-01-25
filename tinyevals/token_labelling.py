"""
This script creates labels for tokens in a sentence. 
It takes the context of the token into account. 
Additionally, it can visualize the sentences and their poart-of-speech (POS) tags.
"""

from typing import List

import spacy
from spacy.tokens import Doc
from spacy import displacy


# make sure the english language model capabilities are installed by the equivalent of:
# python -m spacy download en_core_web_sm
# Should be run once, initially. Download only starts if not already installed.
spacy.cli.download("en_core_web_sm", False, False, '-q')


CATEGORIES = {
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (lambda token: token.text[0].isupper()),  # bool
    "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    "Is Noun": (lambda token: token.pos_ == "NOUN"),  # redundant
    "Is Verb": (lambda token: "VB" in token.tag_),  # redundant
    "Is Adjective": (lambda token: token.pos_ == "ADJ"),  # redundant
    "Is Adverb": (lambda token: token.pos_ == "ADV"),  # redundant
    "Named Entity Type": (lambda token: token.ent_type_ if token.ent_type_ != '' else token.ent_type_),  # False, 'PERSON', 'ORG', 'GPE', ..
}


def label_tokens(sentences: List, verbose: bool = False) -> List[List]:
    """
    Labels tokens in a sentence. Takes the context of the token into account.

    Parameters
    ----------
    sentences : List
        A batch/list of sentences, each being a list of tokens.
    verbose : bool, optional
        Whether to print the tokens and their labels to the console, by default False.  

    Returns
    -------
    List[List]
        Returns a list of sentences. Each sentence contains a list of its 
        corresponding token length where each entry provides the labels/categories 
        for the token. Sentence -> Token -> Labels
    """
    assert isinstance(sentences, list)
    # Load english language model
    nlp = spacy.load("en_core_web_sm")
    # labelled tokens, List holding sentences which hold tokens which hold corresponding token labels
    labelled_sentences = list()

    for sentence in sentences:
        # Create a Doc from the list of tokens
        doc = Doc(nlp.vocab, words=sentence)

        # Apply the spaCy pipeline, except for the tokenizer
        for name, proc in nlp.pipeline:
            if name != "tokenizer":
                doc = proc(doc)

        labelled_tokens = list()  # List holding labels for all tokens of sentence

        for token in doc:
            labels = list()  #  The list holding labels of a single token
            for cname, category_check in CATEGORIES.items():
                label = category_check(token)
                labels.append(label)
            # add current token's to the list
            labelled_tokens.append(labels)
            
            # print the token and its labels to console
            if verbose is True:
                print(f"Token: {token.text}")
                print(" | ".join(list(CATEGORIES.keys())))
                printable = [str(l).ljust(len(cname)) for l, cname in zip(labels, CATEGORIES.keys())]
                printable = " | ".join(printable)
                print(printable)
                print("---")
        # add current sentence's tokens' labels to the list
        labelled_sentences.append(labelled_tokens)
        
        if verbose is True:
            print("\n")

        return labelled_sentences


if __name__ == "__main__":
    result = label_tokens(
        ["Hi, my name is Quan. This is a great example, Peter.".split(" ")], verbose=True
    )
    print(result)
