import os, shutil

# 원본 이미지 폴더 (남자 이미지만 있어야 함)
male_dir = '30_50_녀_new'

# 출력 폴더
output_dir = 'morph_ff'
os.makedirs(output_dir, exist_ok=True)

# 남자 이미지 파일 리스트 (정렬)
males = sorted([f for f in os.listdir(male_dir) if f.lower().endswith(('.jpg', '.png'))])

# 가능한 쌍 개수 계산 (짝수 기준)
pair_count = len(males) // 2

# 쌍 생성
for i in range(pair_count):
    a_img = males[i * 2]
    b_img = males[i * 2 + 1]
    
    shutil.copy(os.path.join(male_dir, a_img), os.path.join(output_dir, f'{i+1:04d}_A.jpg'))
    shutil.copy(os.path.join(male_dir, b_img), os.path.join(output_dir, f'{i+1:04d}_B.jpg'))

print(f'✅ 남남 morph_mm 폴더에 {pair_count}쌍 생성 완료!')
