def word_count_dict(sentences):
    count_dict = {}

    for sentence in sentences:
        for word in sentence:
            count_dict[word] = count_dict.get(word, 0) + 1

    return count_dict