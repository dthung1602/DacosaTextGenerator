from typing import List, Tuple

import torch
from torch import nn, Tensor

from constants import *
from .AbstractModel import AbstractModel


class CharacterLevelRNN(AbstractModel):
    def __init__(self, book_sets: List[BookSet], seq_len: int, batch_size: int,
                 learning_rate: float, weight_decay: float,
                 embedding_dim: int, rnn_hidden_size: int, rnn_num_layers: int, rnn_dropout: float,
                 rnn_variant: str, dropout: float):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.lm_level = LanguageModelLevel.CHAR_LEVEL

        self.embedding = nn.Embedding(CHAR_VOCAB_SIZE, embedding_dim=embedding_dim)
        self.rnn = getattr(nn, rnn_variant)(
            batch_first=True,
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(rnn_hidden_size, CHAR_VOCAB_SIZE)

    def generate(self, seed_sentence: str, required_length: int) -> str:
        seed_sentence = [CHAR_VOCAB_MAPPING[c] for c in seed_sentence]
        seed_sentence = torch.tensor(seed_sentence).long().unsqueeze(dim=0).cuda()

        self.freeze()
        #  print("\n\n\n=============================\n\n")
        _, rnn_state = self(seed_sentence)
        last_char = seed_sentence[:, -1].unsqueeze(dim=0)
        out_sentence = ""

        for _ in range(required_length):
            logits, rnn_state = self(last_char, rnn_state)
            last_char = torch.argmax(logits.flatten())
            out_sentence += CHAR_VOCABULARY[last_char]
            last_char = last_char.unsqueeze(dim=0).unsqueeze(dim=0)

        self.unfreeze()
        return out_sentence

    def forward(self, inputs: Tensor, init_rnn_state: Tensor = None) -> Tuple:
        x = self.embedding(inputs)
        x, next_forward_rnn_state = self.rnn(x, init_rnn_state)
        x = self.dropout(x)

        logits = []
        for s in range(x.shape[1]):  # loop through every char in seq
            logits.append(self.linear(x[:, s, :]))
        logits = torch.stack(logits)
        logits = logits.permute([1, 2, 0])

        return logits, next_forward_rnn_state
