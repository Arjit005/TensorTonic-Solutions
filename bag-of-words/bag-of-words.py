from collections import defaultdict
import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    # counts words that are present in document ,wrt vocab
    # represent document as vector


    #  collect all words uniquely from all documents


    # vocab = the questions we must answer.
    # tokens = the document we search

    """
    unique_words=np.unique1d(tokens,vocab)# gives an array
    n=len(unique_words)
    ans=[]
    for i in range(n):
        count=0
        if unique_words[i]  in tokens:
            count+=1
            ans.append(count)
        else:
            ans.append(0)
    return ans        
    wrong """


    """
    Your job is:

    For each word in vocab:
           ↓
    count it inside tokens
           ↓
    put count into answer
    """

    # method 1
    # ans=[]
    # for word in range(len(vocab)):
    #     count=tokens.count(vocab[word])
    #     ans.append(count)
           
    # final_ans=np.asarray(ans)
    # return final_ans

    # method 2
    """
    TOKENS
      ↓
    count each word
      ↓
    dictionary
    
    "i"    → 1
    "love" → 1
    "cats" → 2
        """
    count=defaultdict(int)
    for word in tokens:
        count[word]+=1
    ans=[]
    for word in vocab:
        ans.append(count[word])
    final_ans=np.asarray(ans,dtype=int)
    return final_ans
    
    