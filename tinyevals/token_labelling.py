"""
This script creates labels for tokens in a sentence. 
It takes the context of the token into account. 
Additionally, it can visualize the sentences and their poart-of-speech (POS) tags.
"""

from typing import List

import spacy
from spacy.tokens import Doc
from spacy import displacy


CATEGORIES = {
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (lambda token: token.text[0].isupper()),  # bool
    "POS Tag": (lambda token: token.pos_),  # 'NOUN', 'VB', ..
    "Is Noun": (lambda token: token.pos_ == "NOUN"),  # redundant
    "Is Verb": (lambda token: "VB" in token.tag_),  # redundant
    "Is Adjective": (lambda token: token.pos_ == "ADJ"),  # redundant
    "Is Adverb": (lambda token: token.pos_ == "ADV"),  # redundant
}


def label_tokens(sentences: List, verbose: bool = False) -> List:
    assert isinstance(input, list)
    # Load english
    nlp = spacy.load("en_core_web_sm")
    # labelled tokens
    labelled_sentences = (
        list()
    )  # List holding sentences which hold tokens which hold corresponding token labels

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
            for category_check in CATEGORIES.values():
                label = category_check(token)
                labels.append(label)
                print(label, end="\t")
            print()
            labelled_tokens.append(labels)

            if verbose is True:
                print(f"Token: {token.text}")
                print(
                    f"Starts with Space: {'Yes' if token.text.startswith(' ') else 'No'}"
                )
                print(f"Capitalized: {'Yes' if token.text[0].isupper() else 'No'}")
                print(f"POS Tag: {token.pos_}")
                print(f"Is Noun: {'Yes' if token.pos_ == 'NOUN' else 'No'}")
                print(
                    f"Is Singular: {'Yes' if token.tag_ == 'NN' else 'No'}, {token.tag_}"
                )
                print(f"Is Plural: {'Yes' if token.tag_ == 'NNS' else 'No'}")
                print(f"Is Verb: {'Yes' if 'VB' in token.tag_ else 'No'}")
                print(f"Is Adjective: {'Yes' if token.pos_ == 'ADJ' else 'No'}")
                print(f"Is Adverb: {'Yes' if token.pos_ == 'ADV' else 'No'}")
                # print(f"Is Part of a Word: {'Yes' if not token.is_alpha else 'No'}")
                # Named Entity Recognition
                if token.ent_type_:
                    print(f"Named Entity Type: {token.ent_type_}")
                print("---")

            # Additional checks for subjects or other complex categories can be added here

        labelled_sentences.append(labelled_tokens)
        print("\n")

        return labelled_sentences


if __name__ == "__main__":
    label_tokens(
        ["Hi, my name is Quan. This is a great example.".split(" ")], verbose=True
    )
