from PIL import Image
import numpy as np
import os

# 설정
input_folder = 'processed_images224'    # 원본 이미지들이 있는 폴더 이름
output_folder = 'processed_images224_new'  # 변환한 이미지를 저장할 폴더 이름
target_size = (256, 256)   # 원하는 사이즈


# 폴더 안 모든 파일에 대해 반복
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 이미지 파일만
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')  # RGB 변환
        img = img.resize(target_size)              # 크기 조정
        
        # 변환된 이미지 저장
        save_path = os.path.join(output_folder, filename)
        img.save(save_path)
        
       
