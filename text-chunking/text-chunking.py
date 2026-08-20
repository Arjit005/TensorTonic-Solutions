def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    #Chunking allows us to retrieve only the relevant portion of a large document.

    #Large document → smaller meaningful pieces → easier retrieval → better RAG answers.


    """
    2. Tokens vs characters ⭐⭐⭐

    This is very important.
    
    A chunk can be measured in:
    
    characters
    words
    tokens
    
    For example:
    
    "Python is easy to learn"
    
    might be:
    
    Characters → ~23
    Words      → 5
    Tokens     → roughly 5–7
    
    LLMs ultimately operate around tokens, not simply characters
    """
    """
    Understand the RAG pipeline ⭐⭐⭐

        Before implementing chunking, understand where it sits:
        
        PDF / Website / Document
                  ↓
             Load document
                  ↓
             Extract text
                  ↓
              CHUNKING  ← you're learning this
                  ↓
              Embeddings
                  ↓
           Vector Database
                  ↓
               Retrieval
                  ↓
                 LLM
                  ↓
               Answer
        
        This is important because chunking isn't an isolated technique.
        
        Its purpose is to make retrieval work better.
    
    """
    
    step = chunk_size - overlap
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    res = []
    
    for i in range(0, len(tokens), step):
    
        if i + chunk_size > len(tokens) and i != 0:
            break
    
        res.append(tokens[i:i+chunk_size])
    
    return res