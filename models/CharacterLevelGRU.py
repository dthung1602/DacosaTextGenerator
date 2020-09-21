from typing import List

import torch
from torch import nn, Tensor

from constants import *
from .AbstractModel import AbstractModel


class CharacterLevelGRU(AbstractModel):

    def __init__(self, book_sets: List[BookSet], seq_len: int, batch_size: int,
                 learning_rate: float, weight_decay: float,
                 embedding_dim: int, gru_hidden_size: int, gru_num_layers: int, gru_dropout: float,
                 context_size: int, linear_size: int, linear_dropout: float, linear_activation: str):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.lm_level = LanguageModelLevel.CHAR_LEVEL

        self.embedding = nn.Embedding(CHAR_VOCAB_SIZE, embedding_dim=embedding_dim)
        self.gru = nn.GRU(
            batch_first=True,
            input_size=embedding_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
        )
        self.linear = nn.Linear(gru_hidden_size * context_size, linear_size)
        self.linear_output = nn.Linear(linear_size, CHAR_VOCAB_SIZE)
        self.dropout = nn.Dropout(linear_dropout)
        self.activation = getattr(nn, linear_activation)()

    def generate(self, seed_sentence: str, length: int) -> str:
        return "Unimplemented"

    def forward(self, inputs: Tensor) -> Tensor:
        context_size = self.hparams.context_size
        gru_hidden_size = self.hparams.gru_hidden_size

        embedding = self.embedding(inputs)  # shape: batch, seq, embedding_size
        gru = self.gru(embedding)[0]  # shape: batch, seq, gru_hidden_size

        logits = []
        for i in range(self.hparams.seq_len - self.hparams.context_size):  # loop through seq_len
            context = gru[:, max(i + 1 - context_size, 0):i + 1, :]  # shape batch, context, gru_hidden_size
            context = context.flatten(start_dim=1)  # shape: batch, context * gru_hidden_size
            actual_batch_size, actual_flatten_context_size = context.shape

            pad_size = context_size * gru_hidden_size - actual_flatten_context_size
            pad = torch.zeros([actual_batch_size, pad_size])
            context = torch.cat([pad, context], dim=1)

            x = self.dropout(context)  # shape: batch, context * gru_hidden_size
            x = self.linear(x)  # shape: batch, linear_size
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_output(x)  # shape: batch, vocab_size
            logits.append(x)

        logits = torch.stack(logits)  # shape: seq_len, batch, vocab_size
        logits = logits.permute([1, 2, 0])  # shape: batch, vocab_size, seq_len
        return logits
