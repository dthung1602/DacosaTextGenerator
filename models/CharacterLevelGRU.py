from typing import List, Tuple

import torch
from torch import nn, Tensor
from tqdm.auto import tqdm

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

    def generate(self, seed_sentence: str, required_length: int) -> str:
        seed_sentence = [CHAR_VOCAB_MAPPING[c] for c in seed_sentence]
        seed_sentence = torch.tensor(seed_sentence).long().unsqueeze(dim=0).cuda()

        self.freeze()
        #  print("\n\n\n=============================\n\n")
        _, gru_state, context_state = self(seed_sentence)
        last_char = seed_sentence[:, -1].unsqueeze(dim=0)
        out_sentence = ""

        for _ in tqdm(range(required_length)):
            logits, gru_state, context_state = self(last_char, gru_state, context_state)
            last_char = torch.argmax(logits.flatten())
            out_sentence += CHAR_VOCABULARY[last_char]
            last_char = last_char.unsqueeze(dim=0).unsqueeze(dim=0)

        self.unfreeze()
        return out_sentence

    def forward(self, inputs: Tensor, init_gru_state: Tensor = None, init_context_state: Tensor = None) -> Tuple:
        context_size = self.hparams.context_size
        gru_hidden_size = self.hparams.gru_hidden_size
        actual_batch_size = inputs.shape[0]
        #  print("-------------------------------------------")
        if init_context_state is None:
            init_context_state = torch.zeros([actual_batch_size, (context_size - 1) * gru_hidden_size]).cuda()
        else:
            required_padding = (context_size - 1) * gru_hidden_size - init_context_state.shape[1]
            if required_padding > 0:
                padding = torch.zeros([actual_batch_size, required_padding]).cuda()
                init_context_state = torch.cat([padding, init_context_state], dim=1)

        #  print("input ", inputs.shape)
        embedding = self.embedding(inputs)  # shape: batch, seq, embedding_size
        #  print("embedding ", embedding.shape)
        if init_gru_state is not None:
            gru, h_n = self.gru(embedding, init_gru_state)  # gru shape: batch, actual_seq_len, gru_hidden_size
        else:
            gru, h_n = self.gru(embedding)
        #  print("gru ", gru.shape)
        #  print("hn ", h_n.shape)
        next_forward_context_state = gru[:, -(context_size - 1):, :].flatten(start_dim=1)
        next_forward_gru_state = h_n
        actual_seq_len = gru.shape[1]

        logits = []

        for i in range(actual_seq_len):  # loop through seq_len
            #  print('>>')
            context = gru[:, max(0, i + 1 - context_size):i + 1, :]  # shape batch, context, gru_hidden_size
            #  print("  context ", context.shape)
            context = context.flatten(start_dim=1)  # shape: batch, context * gru_hidden_size
            #  print("  context ", context.shape)
            if i + 1 - context_size < 0:
                context = torch.cat([init_context_state, context], dim=1)
                context = context[:, -context_size * gru_hidden_size:]
            #  print("  context ", context.shape)

            x = self.dropout(context)  # shape: batch, context * gru_hidden_size
            #  print("  dropout ", x.shape)
            x = self.linear(x)  # shape: batch, linear_size
            #  print("  linear ", x.shape)
            x = self.activation(x)
            x = self.dropout(x)
            #  print("  dropout ", x.shape)
            x = self.linear_output(x)  # shape: batch, vocab_size
            #  print("  linear output ", x.shape)
            logits.append(x)

        logits = torch.stack(logits)  # shape: seq_len, batch, vocab_size
        #  print("logits ", logits.shape)
        logits = logits.permute([1, 2, 0])  # shape: batch, vocab_size, seq_len
        #  print("logits ", logits.shape)
        #  print("next_gru ", next_forward_gru_state.shape)
        #  print("next_context ", next_forward_context_state.shape)

        return logits, next_forward_gru_state, next_forward_context_state
