from PIL import Image
import numpy as np
import os

input_folder = 'processed_images'  # 변환된 이미지들이 저장된 폴더

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        
        print(f"{filename} shape: {img_array.shape}")
