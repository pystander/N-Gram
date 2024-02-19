import math


class NGram:
    def __init__(self, n: int, vocab: set) -> None:
        # Check if n is a positive integer
        assert n >= 1

        self.n = n
        self.vocab = vocab

        # Training parameters
        self.counter = None

    def tokenize(self, sentence: str) -> list[str]:
        start = ["<START>"] * (self.n - 1) if self.n > 1 else ["<START>"]
        end = ["<END>"] * (self.n - 1) if self.n > 1 else ["<END>"]
        tokens = start + sentence.strip().split() + end

        # Handle OOV
        for i in range(len(tokens)):
            if tokens[i] not in self.vocab:
                tokens[i] = "<UNK>"

        return tokens

    def fit(self, train_lines: list[str]) -> None:
        # Count n-gram and (n-1)-gram
        n = self.n
        counter = {}

        for line in train_lines:
            tokens = self.tokenize(line)

            for i in range(len(tokens) - n + 1):
                token_set = tuple(tokens[i : i + n])
                subset = tuple(token_set[:-1])

                if token_set in counter:
                    counter[token_set] += 1
                else:
                    counter[token_set] = 1

                if subset in counter:
                    counter[subset] += 1
                else:
                    counter[subset] = 1

        self.counter = counter

    def get_log_prob(self, token_set: tuple) -> float:
        # Compute probability
        counter = self.counter
        subset = tuple(token_set[:-1])

        return math.log2(counter[token_set] / counter[subset])

    def get_perplexity(self, test_lines: list[str]) -> float:
        # Check if trained
        assert self.counter != None

        # Compute product (log sum) of probability
        n = self.n
        counter = self.counter
        log_sum = 0
        N = 0

        for line in test_lines:
            tokens = self.tokenize(line)
            N += len(tokens)

            for i in range(len(tokens) - n + 1):
                token_set = tuple(tokens[i : i + n])

                if token_set in counter:
                    log_sum += self.get_log_prob(token_set)
                else:
                    return float("inf")

        return pow(2, (-1 / N) * log_sum)


class SmoothedNGram(NGram):
    def __init__(self, n: int, vocab: set) -> None:
        super().__init__(n, vocab)

        self.V = len(vocab)

    # Override
    def fit(self, train_lines: list[str]) -> None:
        # Count n-gram and (n-1)-gram
        n = self.n
        counter = {}

        for line in train_lines:
            tokens = self.tokenize(line)

            for i in range(len(tokens) - n + 1):
                token_set = tuple(tokens[i : i + n])
                subset = tuple(token_set[:-1])

                if token_set in counter:
                    counter[token_set] += 1
                else:
                    counter[token_set] = 1

                if subset in counter:
                    counter[subset] += 1
                else:
                    counter[subset] = 1

        self.counter = counter

    # Override
    def get_log_prob(self, token_set: tuple, k: float = 0) -> float:
        # Check if trained
        assert self.counter != None

        # Compute probability
        counter = self.counter
        V = self.V
        subset = tuple(token_set[:-1])

        if token_set in counter:
            return math.log2((counter[token_set] + k) / (counter[subset] + k * V))
        else:
            subset_count = 0

            if subset in counter:
                subset_count = counter[subset]

            return math.log2(k / (subset_count + k * V))

    # Override
    def get_perplexity(self, test_lines: list[str], k: float) -> float:
        # Check if trained
        assert self.counter != None

        # Compute product (log sum) of probability
        n = self.n
        log_sum = 0
        N = 0

        for line in test_lines:
            tokens = self.tokenize(line)
            N += len(tokens)

            for i in range(len(tokens) - n + 1):
                token_set = tuple(tokens[i : i + n])
                log_sum += self.get_log_prob(token_set, k)

        return pow(2, (-1 / N) * log_sum)


class InterpolatedTrigram(NGram):
    def __init__(self, vocab: set) -> None:
        super().__init__(3, vocab)

        # Training parameters
        self.word_count = 0
        self.counter = None

    # Override
    def fit(self, train_lines: list[str]) -> None:
        # Count trigram, bigram, and unigram
        n = self.n
        word_count = 0
        counter = {}

        for line in train_lines:
            tokens = self.tokenize(line)
            word_count += len(tokens)

            for i in range(len(tokens) - n + 1):
                trigram = tuple(tokens[i : i + n])
                bigram = tuple(trigram[:-1])
                unigram = tuple(bigram[:-1])

                if trigram in counter:
                    counter[trigram] += 1
                else:
                    counter[trigram] = 1

                if bigram in counter:
                    counter[bigram] += 1
                else:
                    counter[bigram] = 1

                if unigram in counter:
                    counter[unigram] += 1
                else:
                    counter[unigram] = 1

        self.word_count = word_count
        self.counter = counter

    # Override
    def get_perplexity(self, test_lines: list[str], l1: float, l2: float, l3: float) -> float:
        # Check if trained
        assert self.counter != None

        # Check lambda
        assert 0.0 <= l1 <= 1.0 and 0.0 <= l2 <= 1.0 and 0.0 <= l3 <= 1.0
        assert l1 + l2 + l3 == 1.0

        # Compute product (log sum) of probability
        n = self.n
        word_count = self.word_count
        counter = self.counter
        log_sum = 0
        N = 0

        for line in test_lines:
            tokens = self.tokenize(line)
            N += len(tokens)

            for i in range(len(tokens) - n + 1):
                trigram = tuple(tokens[i : i + n])
                bigram = tuple(trigram[:-1])
                unigram = tuple(bigram[:-1])

                trigram_prob = 0
                bigram_prob = 0
                unigram_prob = 0

                if trigram in counter:
                    trigram_prob = counter[trigram] / counter[bigram]

                if bigram in counter:
                    bigram_prob = counter[bigram] / counter[unigram]

                if unigram in counter:
                    unigram_prob = counter[unigram] / word_count

                log_sum += math.log2(l3 * trigram_prob + l2 * bigram_prob + l1 * unigram_prob)

        return pow(2, (-1 / N) * log_sum)
