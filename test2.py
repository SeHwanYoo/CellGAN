from PIL import Image
import numpy as np

img_path = '/home/yoos-bii/Desktop/ALL/Normal/P9770_A12_Scan1_Normal_1_28001_39501.tif'

try:
    with Image.open(img_path) as img:
        img = img.convert("RGB")  # [R, G, B] 형식으로 변환
        img = np.array(img)  # 이미지 데이터를 numpy 배열로 변환
        # img를 OpenCV 형식으로 변환하려면
        img = img[:, :, ::-1]  # [R, G, B] -> [B, G, R]
except Exception as e:
    print(f"Error: {e}")
