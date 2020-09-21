import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger

from constants import *
from models import CharacterLevelGRU


def main():
    seed_everything(SEED)
    logger = NeptuneLogger(
        api_key=NEPTUNE_API_TOKEN,
        project_name=NEPTUNE_PROJECT_NAME,
        close_after_fit=False,
    )
    trainer = LightningTrainer(
        gpus=torch.cuda.device_count(),
        min_epochs=1,
        max_epochs=1,
        reload_dataloaders_every_epoch=True,
        logger=logger
    )
    model = CharacterLevelGRU(
        book_sets=[BookSet.HCM],
        seq_len=64,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.01,
        embedding_dim=128,
        gru_hidden_size=128,
        gru_num_layers=2,
        gru_dropout=0.1,
        context_size=3,
        linear_size=512,
        linear_dropout=0.1,
        linear_activation='PReLU'
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
