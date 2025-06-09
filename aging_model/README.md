#학습 배경
fast-AgingGAN 모델을 활용하여 domainA (0-20대 얼굴) 약 800장 domainB (40–50대 얼굴) 약 800장으로 학습을 진행함. 학습 목적은 젊은 얼굴을 노화시키는 방향이다.

#학습 결과 test 결과 어린 아이의 얼굴이 어른 얼굴로 노화되는 변화가 미미함 이 부분에 한계가 있다고 생각하여 0-20대 얼굴 약 800장, 40-50대 얼굴 약 800장을 기존 체크포인트를 기반으로 파인튜닝을 진행함.

#향후 고려할 부분
어린이 데이터 비율을 더 높여서 학습시키거나 입력에 나이 조건을 추가해 얼굴 변화 퀄리티를 높이는 것

#관련 링크
1차 학습 dataset 구글 드라이브 링크: https://drive.google.com/drive/folders/1z80F5t07LMHBiVkmxcicaK3b5YFTH5YT?usp=sharing
2차 학습 dataset 구글 드라이브 링크: https://drive.google.com/drive/folders/13rTb38RJ49Cdh6qM2QtjB82OxpjHNHkA?usp=sharing
1차 학습 체크포인트 링크: https://drive.google.com/drive/folders/19JuMV51l5Hks8OrKG13FMKC2Q04C4AT5?usp=sharing [best-model.ckpt]
2차 학습 체크포인트 링크: https://drive.google.com/drive/folders/19JuMV51l5Hks8OrKG13FMKC2Q04C4AT5?usp=sharing [best-model2.ckpt (프로그램 제작 시 이 체크포인트 사용)]

# Fast-AgingGAN
This repository holds code for a face aging deep learning model. It is based on the CycleGAN, where we translate young faces to old and vice versa.

# Samples
Top row is input image, bottom row is aged output from the GAN.
![Sample](https://user-images.githubusercontent.com/4294680/86517626-b4d54100-be2a-11ea-8cf1-7e4e088f96a3.png)
![Second-Sample](https://user-images.githubusercontent.com/4294680/86517663-f5cd5580-be2a-11ea-9e39-51ddf8be2084.png)
# Timing
The model executes at 66fps on a GTX1080 with an image size of 512x512. Because of the way it is trained, a face detection pipeline is not needed. As long as the image of spatial dims 512x512 contains a face of size 256x256, this will work fine.

# Demo
To try out the pretrained model on your images, use the following command:
```bash
python infer.py --image_dir 'path/to/your/image/directory'
```

# Training
To train your own model on CACD or UTK faces datasets, you can use the provided preprocessing scripts in the preprocessing directory to prepare the dataset.
If you are going to use CACD, use the following command:
```bash
python preprocessing/preprocess_cacd.py --image_dir '/path/to/cacd/images' --metadata '/path/to/the/cacd/metadata/file' --output_dir 'path/to/save/processed/data'
```
If using UTK faces, use the following:
```bash
python preprocessing/preprocess_utk.py --data_dir '/path/to/cacd/images' --output_dir 'path/to/save/processed/data'
```

Once the dataset is processed, you should go into ``` configs/aging_gan.yaml``` and modify the paths to point to the processed dataset you just created. Change any other hyperparameters if you wish, then run training with:
```bash
python main.py
```

# Tensorboard
While training is running, you can observe the losses and the gan generated images in tensorboard, just point it to the 'lightning_logs' directory like so:
```bash
tensorboard --logdir=lightning_logs --bind_all
```
