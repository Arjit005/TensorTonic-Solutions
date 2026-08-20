def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    #A stopword is a word that has been designated as relatively uninformative for a particular NLP application.

    #Notice the important word:  
    # designated   
    # There is no universal mathematical rule saying:  
    # "the" = stopword    
    # It's a preprocessing decision.
    
    """
    Raw text
       ↓
    Lowercase
       ↓
    Tokenization
       ↓
    Remove punctuation
       ↓
    Remove stopwords
        """
    result=[]
    for word in tokens:
        if word not in stopwords:
            result.append(word)
    return result    