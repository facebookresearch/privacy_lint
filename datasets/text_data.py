import torch

class TextIterator:
    def __init__(self, sequence, batch_size, seq_len):
        assert sequence.ndim == 1
        self.batch_size = batch_size
        self.sequence = sequence.view(seq_len, -1)
        self.i_batch = 0

    def __iter__(self):
        self.i_batch = 0

        return self

    def __next__(self):
        if (self.i_batch + 1) * self.batch_size < self.sequence.size(1):
            start = self.i_batch * self.batch_size
            end = (self.i_batch + 1) * self.batch_size
            self.i_batch += 1

            return torch.arange(start, end), self.sequence[:, start:end]
        else:
            raise StopIteration
