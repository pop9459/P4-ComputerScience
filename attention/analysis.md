# Analysis

## Layer 2, Head 1

This head consistently attends from each token forward to the immediately following token, forming a near-diagonal stripe one step to the right of the main diagonal. This captures local sequential structure — the model is effectively "peeking ahead" one position, which helps represent the flow of a sentence left-to-right.

Example Sentences:
- "The cat [MASK] on the mat." — Each token's attention is concentrated on the next token, so "cat" strongly attends to [MASK], [MASK] to "on", etc.
- "She quickly [MASK] the door and left." — The same forward-shift pattern appears; "quickly" attends strongly to [MASK], and [MASK] to "the".

## Layer 7, Head 6

This head shows strong attention from almost every token back to the [CLS] special token (position 0). [CLS] acts as a global summary vector in BERT, and this head appears dedicated to routing per-token information into that summary. The pattern is visible as a bright first column regardless of sentence content.

Example Sentences:
- "The scientist [MASK] the experiment carefully." — All content tokens show high attention weight on [CLS], with only weak attention elsewhere.
- "We [MASK] the problem together last night." — Again, regardless of which word is masked, every token's attention is dominated by the [CLS] position, indicating this head gathers global sentence context rather than tracking local relationships.
