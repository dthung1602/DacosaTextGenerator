import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger

from constants import *
from models import CharacterLevelGRU


def train():
    seed_everything(SEED)
    logger = NeptuneLogger(
        api_key=NEPTUNE_API_TOKEN,
        project_name=NEPTUNE_PROJECT_NAME,
        close_after_fit=False,
    )
    trainer = LightningTrainer(
        gpus=torch.cuda.device_count(),
        min_epochs=2,
        max_epochs=2,
        reload_dataloaders_every_epoch=True,
        logger=logger
    )
    model = CharacterLevelGRU(
        book_sets=[BookSet.LENIN, BookSet.HCM, BookSet.VK_DANG],
        seq_len=128 + 32,
        batch_size=32,
        learning_rate=0.0001,
        weight_decay=0.02,
        embedding_dim=128,
        gru_hidden_size=128,
        gru_num_layers=2,
        gru_dropout=0.2,
        context_size=64,
        linear_size=1024 + 512,
        linear_dropout=0.2,
        linear_activation='PReLU'
    )
    try:
        trainer.fit(model)
    except Exception as e:
        logger.experiment.stop(str(e))
        raise e
    else:
        logger.experiment.stop()


if __name__ == '__main__':
    train()
