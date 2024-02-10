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
        # Check if trained
        assert self.counter != None

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
                    subset = tuple(token_set[:-1])
                    log_sum += self.get_log_prob(token_set)
                else:
                    return float("inf")

        return pow(2, (-1 / N) * log_sum)
