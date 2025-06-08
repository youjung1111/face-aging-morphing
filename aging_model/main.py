from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from gan_module import AgingGAN

parser = ArgumentParser()
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')


def main():
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    model = AgingGAN(config)

    best_checkpoint = ModelCheckpoint( #best 모델 저
        save_top_k=1,
        monitor="val_loss", 
        mode="min",
        dirpath="/content/drive/MyDrive/checkpoints",
        filename='best-model',
    )

    epoch_checkpoint = ModelCheckpoint( #매 에폭마다 저장
        save_top_k=-1,
        monitor="val_loss",
        every_n_epochs=1,
        dirpath="/content/drive/MyDrive/checkpoints",
        filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
    )
    
    
    trainer = Trainer(
        max_epochs=config['epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        callbacks=[best_checkpoint, epoch_checkpoint],
        val_check_interval=1.0,
        limit_val_batches=1.0
    )
    trainer.fit(model)

if __name__ == '__main__': 
    main()
