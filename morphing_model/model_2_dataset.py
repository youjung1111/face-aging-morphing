import os, shutil

# 남자, 여자 이미지가 분리된 상태라고 가정
male_dir = '30_50 남_new'
female_dir = '30_50_녀_new'
output_dir = 'morph_mf'
os.makedirs(output_dir, exist_ok=True)

males = sorted([f for f in os.listdir(male_dir) if f.endswith('.jpg')])
females = sorted([f for f in os.listdir(female_dir) if f.endswith('.jpg')])
pair_count = min(len(males), len(females))

for i in range(pair_count):
    shutil.copy(os.path.join(male_dir, males[i]), os.path.join(output_dir, f'{i+1:04d}_A.jpg'))
    shutil.copy(os.path.join(female_dir, females[i]), os.path.join(output_dir, f'{i+1:04d}_B.jpg'))

print(f'{pair_count}쌍 생성 완료 → morph_mf 폴더에 저장됨')
