import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):

    # Tokenize
    tokenize = [doc.split() for doc in documents]

    # Vocabulary
    vocab = sorted(set(word
                       for doc in tokenize
                       for word in doc))

    # TF
    TF = []

    for doc in tokenize:

        counts = Counter(doc)
        total = len(doc)

        tf_doc = {}

        for word in vocab:
            tf_doc[word] = counts[word] / total

        TF.append(tf_doc)

    # IDF
    N = len(documents)
    idf = {}

    for word in vocab:

        df = sum(1 for doc in tokenize if word in doc)

        idf[word] = math.log(N / df)

    # TF-IDF Matrix
    matrix = []

    for doc_tf in TF:

        row = []

        for word in vocab:
            row.append(doc_tf[word] * idf[word])

        matrix.append(row)

    matrix = np.array(matrix)

    return matrix, vocab