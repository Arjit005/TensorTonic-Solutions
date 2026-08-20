def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Write code here

    """
        Edit Distance — What is the Concept? 🧩

        Edit Distance measures how different two strings are.
        
        The most common version is Levenshtein Distance.
        
        It asks:
        
        What is the minimum number of edits required to transform string A into string B?
        
        There are 3 basic edits:
        
            Insert a character
            Delete a character
            Replace a character
    """
    """
    
    Why is this useful?

    This concept becomes extremely useful whenever two strings might be similar but not exactly identical.
    
    🔎 Spell checking
    
    User types:
    
    "pyhton"
    
    Dictionary contains:
    
    "python"
    
    Compare:
    
    pyhton
    python
    
    They are very similar.
    
    Edit distance can help identify:
    
    "python" is probably what the user intended.


    🔎 Search engines

    Suppose someone searches:
    
    "recieve"
    
    but the actual word is:
    
    "receive"
    
    A search system can use edit distance to find nearby words.
    """


    """
    🔎 Fuzzy string matching

    Suppose your database contains:
    
    Apple Inc.
    Microsoft
    Google
    Amazon
    
    User enters:
    
    "Googel"
    
    Exact matching fails:
    
    "Googel" == "Google"
    # False
    
    But edit distance can tell us:
    
    Googel
    Google
       ↑
    swap characters
    
    They are very close.
    
    """
    """
    The DP table idea

        For:
        
        cat
        cut
        
        we compare prefixes:
        
                ""   c   u   t
             ----------------
        ""   |  0   1   2   3
        c    |  1   ?   ?   ?
        a    |  2   ?   ?   ?
        t    |  3   ?   ?   ?
        
        The table eventually gives:
        
                ""   c   u   t
             ----------------
        ""   |  0   1   2   3
        c    |  1   0   1   2
        a    |  2   1   1   2
        t    |  3   2   2   1
        
        Bottom-right:
        
        1
        
        Therefore:
        
        Edit Distance("cat", "cut") = 1
    
    """
    m = len(s1) + 1
    n = len(s2) + 1
    
    dp = [[0] * n for _ in range(m)]
    
    # first column
    for i in range(m):
        dp[i][0] = i
    
    # first row
    for j in range(n):
        dp[0][j] = j
    
    for i in range(1, m):
        for j in range(1, n):
    
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
    
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # delete
                    dp[i][j-1],      # insert
                    dp[i-1][j-1]     # replace
                )
    
    return dp[m-1][n-1]