import math
import os
from abc import ABC, abstractmethod
from random import shuffle
from typing import List, Iterable
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from constants import BookSet, LanguageModelLevel, PROCESSED_DATA_DIR


class ConfigurableDataset(IterableDataset):
    def __init__(self, book_sets: List[BookSet], lm_level: LanguageModelLevel, seq_len: int, batch_size: int):
        self.seq_len = seq_len
        self.batch_size = batch_size

        file_extension = f".{lm_level.value}.npy"
        files = []
        for file in os.listdir(PROCESSED_DATA_DIR):
            for bs in book_sets:
                if file.upper().startswith(bs.value) and file.endswith(file_extension):
                    files.append(os.path.join(PROCESSED_DATA_DIR, file))
        shuffle(files)

        self.data = np.concatenate([np.load(file) for file in files])

        self.num_samples = math.floor((len(self.data) - 1) / seq_len)
        self.length = math.ceil(self.num_samples / batch_size)

    def __len__(self):
        return self.length

    def __iter__(self) -> Iterable:
        seq_len = self.seq_len
        for i in range(self.num_samples):
            yield (
                torch.tensor(self.data[i * seq_len: i * seq_len + seq_len]).long(),
                torch.tensor(self.data[i * seq_len + 1: i * seq_len + seq_len + 1]).long(),
            )


SAMPLE_SEED_SENTENCES = [
    "Cách mệnh An Nam cũng là một bộ phận trong cách mệnh thế giới. ",
    "Trong khi đó, dù đạt được những kết quả rất đáng tự hào, đất nước ta vẫn đứng trước nhiều khó khăn, thách thức.",
    "Đại hội đại biểu toàn quốc lần thứ XIII của Đảng lần này diễn ra trong bối cảnh tình hình thế giới và khu vực có những diễn biến rất nhanh, phức tạp, khó dự báo.",
    "Báo cáo chính trị là văn kiện trung tâm của Đại hội, cùng với Báo cáo tổng kết thực hiện Chiến lược phát triển kinh tế - xã hội 10 năm 2011 - 2020, xây dựng Chiến lược phát triển kinh tế - xã hội 10 năm 2021 - 2030"
]


class AbstractModel(LightningModule, ABC):
    @staticmethod
    def loss(outputs: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(outputs, labels)

    @abstractmethod
    def generate(self, seed_sentence: str, length: int) -> str:
        pass

    def configure_optimizers(self):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        batch_size = self.hparams.batch_size
        dataset = ConfigurableDataset(
            book_sets=self.hparams.book_sets,
            lm_level=self.hparams.lm_level,
            seq_len=self.hparams.seq_len,
            batch_size=self.hparams.batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_num: int) -> dict:
        inputs, labels = batch
        logits = self(inputs)[0]
        loss = self.loss(logits, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        for sentence in SAMPLE_SEED_SENTENCES:
            generated_sentence = self.generate(sentence, len(sentence))
            print("---------")
            print(f'"{generated_sentence}"')
            print("---------")
            self.logger.experiment.log_text("generated_sentence", generated_sentence)
        return {}
