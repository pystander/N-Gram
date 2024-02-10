def get_vocab(lines: list[str], threshold: int) -> set:
    """
    Build vocabulary from corpus.
    """

    counter = {}

    for line in lines:
        tokens = line.strip().split()

        for token in tokens:
            if token in counter:
                counter[token] += 1
            else:
                counter[token] = 1

    vocab = set()

    for token, count in counter.items():
        if count >= threshold:
            vocab.add(token)

    return vocab
