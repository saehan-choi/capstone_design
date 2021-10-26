import Augmentor

## 증강 시킬 이미지 폴더 경로
img = Augmentor.Pipeline("./train/2xxx/")

## 왜곡
# img.random_distortion(probability=0.1, grid_width=10, grid_height=10, magnitude=8)
# img.rotate(0.8, max_left_rotation=2, max_right_rotation=2)
# img.zoom(probability=0.6, min_factor=1.1, max_factor=1.2)
## 증강 이미지 수
img.random_erasing(1,0.2)
img.sample(200)