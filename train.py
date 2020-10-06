import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger

from constants import *
from models import CharacterLevelRNN


def train():
    seed_everything(SEED)
    logger = NeptuneLogger(
        api_key=NEPTUNE_API_TOKEN,
        project_name=NEPTUNE_PROJECT_NAME,
        close_after_fit=False,
    )
    trainer = LightningTrainer(
        gpus=torch.cuda.device_count(),
        min_epochs=4,
        max_epochs=4,
        reload_dataloaders_every_epoch=True,
        logger=logger
    )
    model = CharacterLevelRNN(
        book_sets=[BookSet.HCM],
        seq_len=32,
        batch_size=16,
        learning_rate=0.0002,
        weight_decay=0.01,
        embedding_dim=512,
        rnn_variant='GRU',
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0.2,
        dropout=0.2,
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
