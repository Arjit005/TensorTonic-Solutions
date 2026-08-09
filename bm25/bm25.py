import numpy as np
from collections import Counter
import math


def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """

    # ============================================================
    #                  BM25 — THE GOAL
    # ============================================================

    """
    GOAL:
        Given a query, rank documents by relevance.

    Example query:
        "machine learning"

    BM25 tries to answer:

        Which documents are most relevant to this query?


    But "relevance" is tricky.

    BM25 solves 3 problems:

        ┌─────────────────────────────────────────────────────────┐
        │                 GOAL: Rank by Relevance                 │
        └─────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼

        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  PROBLEM 1   │ │  PROBLEM 2   │ │  PROBLEM 3   │
        │ Common words │ │ More mentions│ │ Long docs    │
        │ shouldn't    │ │ shouldn't    │ │ shouldn't    │
        │ dominate     │ │ grow forever │ │ auto-win     │
        └──────────────┘ └──────────────┘ └──────────────┘
                │             │             │
                ▼             ▼             ▼
             SOLUTION:     SOLUTION:      SOLUTION:
                IDF       TF Saturation   Length Norm


    ============================================================
                    THE "WHY" IN ONE TABLE
    ============================================================

    | Calculation             | Why It Exists                         |
    |-------------------------|---------------------------------------|
    | IDF                     | Stop common words like "the" from     |
    |                         | dominating                            |
    | TF Saturation           | Stop docs with 1000 mentions from     |
    |                         | automatically winning                 |
    | Length Norm             | Stop long docs from winning just      |
    | (|D| / avgDL)           | because they contain more words       |
    | Multiply & Sum          | Combine everything into one           |
    |                         | relevance score                       |


    ============================================================
                    OVERALL BM25 PIPELINE
    ============================================================

        ┌─────────────┐
        │   Corpus    │
        │  (all docs) │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Indexing   │
        │(count terms)│
        └──────┬──────┘
               │
               ▼
        ┌─────────────────┐
        │  Query arrives  │
        │  "machine       │
        │   learning"     │
        └──────┬──────────┘
               │
               ▼
        ┌─────────────┐
        │   Ranking   │
        │ (BM25 score │
        │  per doc)   │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ Sort &      │
        │ Return      │
        └─────────────┘


    ============================================================
                    STEP 1 — IDF
              "HOW RARE IS THIS WORD?"
    ============================================================

    Before looking at any individual document, BM25 asks:

        "Does this word even matter?"


    Mathematical idea:

        Rare word
            ↓
        More informative
            ↓
        Higher IDF

        Common word
            ↓
        Less informative
            ↓
        Lower IDF


    Example:

        IDF("machine")
        = log(1 + (10 − 6 + 0.5) / (6 + 0.5))
        = 0.526


        IDF("learning")
        = log(1 + (10 − 5 + 0.5) / (5 + 0.5))
        = 0.693


    Why?

        "learning"
        appears in only 5 docs
              ↓
        rarer
              ↓
        higher IDF
              ↓
        contributes more to score


        "machine"
        appears in 6 docs
              ↓
        more common
              ↓
        lower IDF
              ↓
        contributes less


    If a word appears in ALL 10 documents:

        appears everywhere
              ↓
        IDF becomes very small
              ↓
        word is almost ignored


    Analogy:

        Search:
            "the algorithm"

        "the"
            ↓
        appears everywhere
            ↓
        IDF ≈ 0
            ↓
        not useful


        "algorithm"
            ↓
        rarer / more informative
            ↓
        higher IDF
            ↓
        valuable


    ============================================================
                    STEP 2 — DOCUMENT LENGTH
              "HOW LONG IS THIS DOCUMENT?"
    ============================================================

    Before we can normalize a document, we need:

        |D| = length of document D


    Example:

        D4 = "AI Encyclopedia"

        |D| = 400 words


    We also need:

        avgDL = average document length


    Example:

        avgDL = 192


    avgDL is simply:

        arithmetic mean of all document lengths.


    Why do we need |D|?

    Because long documents naturally contain more words.

    Example:

        D4 = 400 words
        mentions "machine" 4 times

        D3 = 120 words
        mentions "machine" 4 times


    Which document is MORE focused on machines?

        D3 — the short document.


    Why?

        Same number of mentions
        but much smaller document
              ↓
        higher concentration
              ↓
        probably more focused


    Therefore BM25 must account for document length.


    ============================================================
                    STEP 3 — LENGTH NORMALIZATION
                  "IS THIS DOC TOO LONG?"
    ============================================================

    Solution:

        Length Normalization


    Formula:

        norm = (1 − b) + b × (|D| / avgDL)


    Parameters:

        b = 0.75
        avgDL = 192


    Interpretation:

        |D| < avgDL
            ↓
        norm < 1
            ↓
        shorter than average
            ↓
        receives a relative bonus


        |D| ≈ avgDL
            ↓
        norm ≈ 1
            ↓
        approximately neutral


        |D| > avgDL
            ↓
        norm > 1
            ↓
        longer than average
            ↓
        receives a penalty


    Table:

    | Doc | Length | norm  | Meaning                       |
    |-----|-------:|------:|-------------------------------|
    | D3  |    120 | 0.719 | Short doc → bonus (norm < 1)  |
    | D1  |    150 | 0.836 | Short doc → bonus             |
    | D5  |    200 | 1.031 | ≈ avg length → neutral        |
    | D4  |    400 | 1.813 | Long doc → heavy penalty      |


    ============================================================
                    HOW |D| FLOWS FORWARD
    ============================================================

        Raw document
              │
              ▼
        |D| = document length
              │
              ▼
        avgDL = average document length
              │
              ▼
        Length Normalization
              │
              ▼
        norm
              │
              ▼
        norm becomes an input
        to TF Saturation
              │
              ▼
        TF_score


    Concrete example:

        D4 = 400 words

        norm
        = 0.25 + 0.75 × (400 / 192)
        = 1.813


    Meaning:

        D4 is approximately 2× longer than average.


    ============================================================
                    STEP 4 — TERM FREQUENCY
                  "HOW MANY TIMES DID IT APPEAR?"
    ============================================================

    Now we look at a specific query term
    inside a specific document.

    Let:

        f = raw frequency of the query term
            inside the document.


    Example:

        Document:
            ["machine", "learning", "machine"]

        Query term:
            "machine"

        f = 2


    Important:

        More occurrences should increase relevance,

        BUT

        1000 occurrences should NOT make a document
        automatically win.


    This creates the need for TF Saturation.


    ============================================================
                    STEP 5 — TF SATURATION
                "DIMINISHING RETURNS"
    ============================================================

    Formula:

        TF_score =
            f × (k₁ + 1)
            ----------------
            f + k₁ × norm


    Key idea:

        First few mentions
            ↓
        useful information


        More and more mentions
            ↓
        diminishing additional value


        Very large f
            ↓
        score approaches a maximum
            ↓
        but does not grow forever


    With k₁ = 1.5:

        TF_score approaches 2.5
        but never exceeds it.


    Example values:

    | Doc | Term     | raw f | norm  | TF_score | % of max |
    |-----|----------|------:|------:|---------:|---------:|
    | D5  | machine  |     4 | 1.031 |    1.803 |      72% |
    | D5  | learning |     5 | 1.031 |    1.909 |      76% |
    | D1  | machine  |     3 | 0.836 |    1.763 |      71% |
    | D1  | learning |     4 | 0.836 |    1.903 |      76% |
    | D4  | machine  |     2 | 1.813 |    1.060 |      42% |
    | D4  | learning |     2 | 1.813 |    1.060 |      42% |


    Notice:

    • D5 has the highest raw counts (4 and 5),
      so its TF scores are highest.

    • D4 has the same raw counts as D1/D2 but is crushed
      by norm = 1.813 → TF scores are much lower.

    • D3 has machine = 4 but learning = 0
      → it only gets credit for "machine".


    Analogy:

        1st mention
            → tells you a lot

        2nd / 3rd mention
            → still useful

        50th mention
            → tells you very little new


    BM25 models this with a saturation curve.


    ============================================================
                    STEP 6 — FINAL SCORE
                    "MULTIPLY & SUM"
    ============================================================

    Now we have two important quantities:

        IDF(term)
            ↓
        How informative is the term?


        TF_score(term, document)
            ↓
        How strongly does this document contain the term?


    Combine them:

        contribution =
            IDF(term) × TF_score(term, document)


    For every query term:

        score =
            Σ IDF(term) × TF_score(term, document)


    Example:

        Query:
            "machine learning"


        For "machine":

            IDF("machine") × TF_score("machine", D)


        For "learning":

            IDF("learning") × TF_score("learning", D)


        Final:

            machine contribution
                    +
            learning contribution
                    =
            BM25 score for document D


    Higher score
        ↓
    more relevant document
        ↓
    higher ranking


    ============================================================
                    COMPLETE CALCULATION FLOW
    ============================================================

        Query term
            │
            ▼
        How rare is it?
            │
            ▼
           IDF
            │
            │
            ├──────────────┐
            │              │
            ▼              ▼
        Document       Term frequency
        length |D|          f
            │              │
            ▼              │
        Length Norm         │
            │              │
            └──────┬───────┘
                   ▼
             TF Saturation
                   │
                   ▼
              TF_score
                   │
                   ▼
            IDF × TF_score
                   │
                   ▼
             Sum query terms
                   │
                   ▼
             BM25 score
                   │
                   ▼
                Ranking


    ============================================================
                    IMPORTANT PARAMETERS
    ============================================================

        k₁ = 1.5
        b  = 0.75


    These are hyperparameters that can be tuned.


    avgDL = 192

    avgDL means:

        Average Document Length

    = arithmetic mean of all document lengths.


    ============================================================
                    WHY EACH PART EXISTS
    ============================================================

        Problem:
        Common words dominate
              ↓
        Solution:
        IDF


        Problem:
        Repeating a word 1000 times
        should not give unlimited score
              ↓
        Solution:
        TF Saturation


        Problem:
        Long documents naturally contain
        more words and would have an unfair advantage
              ↓
        Solution:
        Length Normalization


        Finally:

        IDF × TF_score
              ↓
        sum across query terms
              ↓
        final relevance score


    ============================================================
                    WHY LENGTH NORMALIZATION MATTERS
    ============================================================

    Without |D| and length normalization:

        D4
        400 words
        machine = 2
        learning = 2


    could score higher than:

        D1
        150 words
        machine = 3
        learning = 4


    That would be undesirable because D1 is clearly
    more focused on "machine learning."


    Therefore:

        |D|
         ↓
        Length Normalization
         ↓
        norm
         ↓
        TF Saturation
         ↓
        TF_score
         ↓
        IDF × TF_score
         ↓
        Final BM25 Score
         ↓
        Ranking


    ============================================================
                    FINAL INTUITION
    ============================================================

    BM25 asks three questions:

    1. Is this word rare and informative?
       → IDF

    2. How strongly does the document contain it?
       → TF Saturation

    3. Is the document unusually long?
       → Length Normalization


    Then:

        IDF × TF_score

    is calculated for each query term,

    and all query-term contributions are summed.


    ============================================================
                    ONE-LINE MENTAL MODEL
    ============================================================

        BM25 =

        "How rare is the word?"
              ×
        "How strongly does this document contain it?"
              ×
        "Adjust for document length"

              ↓

        Then sum across query terms.


    ============================================================
                     CHEAT SHEET
                  
    ============================================================

        ┌──────────────────────────────────────────────────────┐
        │       BM25 = IDF × SATURATED_TF                      │
        ├──────────────────────────────────────────────────────┤
        │ IDF(q) = log((N − n + 0.5) / (n + 0.5))              │
        │          // n = docs containing q                    │
        │          // Rare = high                              │
        ├──────────────────────────────────────────────────────┤
        │ norm = (1 − b) + b × (|D| / avgDL)                   │
        │        // |D| = document length                      │
        │        // Short = relative bonus                     │
        ├──────────────────────────────────────────────────────┤
        │ TF_score = f × (k₁ + 1) / (f + k₁ × norm)            │
        │            // f = raw count                          │
        │            // Saturation curve                       │
        ├──────────────────────────────────────────────────────┤
        │ score = Σ IDF(qᵢ) × TF_score(qᵢ, D)                  │
        │         // Sum across ALL query terms                │
        │         // Highest score wins                        │
        └──────────────────────────────────────────────────────┘


    FINAL MENTAL MODEL:

    "BM25 finds documents that are focused (short),
     relevant (match query terms), and not spammy
     (reasonable term density) — then ranks them by
     combining word rarity, saturated term frequency,
     and document-length fairness."


    ============================================================
                    CODING ROADMAP
              CONCEPT → CODE → NEXT DEPENDENCY
    ============================================================

        1. Documents
             ↓
        2. Document lengths |D|
             ↓
        3. Average document length avgDL
             ↓
        4. Length normalization norm
             ↓
        5. Query terms
             ↓
        6. Term frequency f
             ↓
        7. Document frequency n
             ↓
        8. IDF
             ↓
        9. TF Saturation
             ↓
        10. IDF × TF_score
             ↓
        11. Sum across query terms
             ↓
        12. BM25 score for every document
             ↓
        13. Return NumPy array
             ↓
        14. Rank documents


    
    """

    # ============================================================
    #                    CODE STARTS HERE
    # ============================================================
# Write BM25 implementation here

    N = len(docs)
    
    doc_lengths = []
    
    for doc in docs:
        doc_length = len(doc)
        doc_lengths.append(doc_length)
    
    # convert doc_lengths array into numpy array so that we can calculate average
    doc_lengths_array = np.asarray(doc_lengths)
    
    avgDL = np.sum(doc_lengths_array) / N
    
    
    # calculating norms ==> so that we can make sure that large document
    # doesn't always win using penalty and bonus
    norms = (1 - b) + b * (doc_lengths_array / avgDL)
    
    
    # Count How Many Documents Contain Each Query Term — n(q)
    term_doc_counter = Counter()
    
    for doc in docs:                # Loop 1: process each document
        unique_terms = set(doc)     # Remove duplicates within this doc
                                    # ["machine","learning","machine"]
                                    # → {"machine","learning"}
    
        for term in unique_terms:   # Loop 2: for each unique term in this doc
            term_doc_counter[term] += 1   # Count: this doc contains this term
    
    
    # compute idf
    idf_scores = {}
    
    for term in query_tokens:
        nq = term_doc_counter[term]   # How many docs contain this term
    
        idf = math.log(
            1 + (N - nq + 0.5) / (nq + 0.5)
        )
    
        idf_scores[term] = idf
    
    
    # Compute Raw Term Frequency f(q, D) for Each (Query Term, Document) Pair
    
    # Create a matrix to store TF scores
    # rows    → documents
    # columns → query terms
    tf_scores = np.zeros((N, len(query_tokens)))
    
    for doc_idx, doc in enumerate(docs):        # Outer loop: each document
    
        for term_idx, term in enumerate(query_tokens):  # Inner loop: each query term
    
            f = doc.count(term)                  # Count!
    
            # compute TF saturation formula
            tf_score = (
                f * (k1 + 1)
                / (f + k1 * norms[doc_idx])
            )
    
            # Store the TF score for this document and query term
            tf_scores[doc_idx, term_idx] = tf_score
    
    
    # Convert idf_scores dict into a NumPy array in the same order as query_tokens
    idf_array = np.array(
        [idf_scores[term] for term in query_tokens]
    )   # shape: (num_terms,)
    
    
    # Multiply: each doc's TF scores by the corresponding IDF weights
    # tf_scores has shape: (N, num_terms), idf_array has shape: (num_terms,)
    # NumPy broadcasting: idf_array is applied to every row
    weighted_scores = tf_scores * idf_array   # shape: (N, num_terms)
    
    
    # Sum across query terms (axis=1) to get one score per document
    bm25_scores = np.sum(weighted_scores, axis=1)   # shape: (N,)
    
    return bm25_scores