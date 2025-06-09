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

    if 'ckpt_path' in config and config['ckpt_path']:
        print(f"Loading from checkpoint: {config['ckpt_path']}")
        model = AgingGAN.load_from_checkpoint(config['ckpt_path'], config=config)
    else:
        model = AgingGAN(config)

    best_checkpoint = ModelCheckpoint( #best 모델 저
        save_top_k=1,
        monitor="val_loss", 
        mode="min",
        dirpath="/content/drive/MyDrive/checkpoints",
        filename='best-model',
    )
    
    trainer = Trainer(
        max_epochs=config['epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        callbacks=[best_checkpoint],
        val_check_interval=1.0,
        limit_val_batches=1.0
    )
    trainer.fit(model)

if __name__ == '__main__': 
    main()
