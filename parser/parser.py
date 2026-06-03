import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> Det N | Det AP N | N | NP PP
VP -> V | V NP | V PP | V NP PP | Adv VP | VP Adv | VP Conj VP
AP -> Adj | Adj AP
PP -> P NP
"""
# S:  a sentence is a noun phrase + verb phrase, or two sentences joined by a conjunction
# NP: a noun phrase is a noun (with optional determiner/adjectives), optionally followed by a prepositional phrase
# VP: a verb phrase is a verb optionally followed by objects/prepositional phrases, or modified by an adverb
# AP: an adjective phrase is one or more adjectives chained together
# PP: a prepositional phrase is a preposition followed by a noun phrase

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """Convert sentence to a lowercase list of words, excluding non-alphabetic tokens."""
    words = nltk.word_tokenize(sentence.lower())
    # Drop punctuation and any token that has no letters (e.g. ".", ",", "123")
    return [word for word in words if any(c.isalpha() for c in word)]


def np_chunk(tree):
    """Return all NP subtrees that contain no nested NP subtrees."""
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            # Skip this NP if it contains a smaller NP inside it
            has_sub_np = any(
                sub.label() == "NP"
                for sub in subtree.subtrees()
                if sub is not subtree
            )
            if not has_sub_np:
                chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
