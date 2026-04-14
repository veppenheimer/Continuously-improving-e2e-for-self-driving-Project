import os
import cv2
import albumentations as A

# ==== 配置参数 ====
input_dir = 'data/raw'       # 原始图像目录
output_dir = 'data/aug3'     # 增强后图像目录
os.makedirs(output_dir, exist_ok=True)

# ==== 运动模糊强度配置 ====
blur_strengths = [5, 9, 13]  # 适中的多级模糊参数

# ==== 遍历图像并增强 ====
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for blur_size in blur_strengths:
            transform = A.Compose([
                A.MotionBlur(blur_limit=(blur_size, blur_size), p=1.0),  # 强度固定
            ])
            augmented = transform(image=img_rgb)
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

            # 生成保存路径，标记模糊强度
            name, ext = os.path.splitext(filename)
            save_name = f"{name}_motion{blur_size}{ext}"
            save_path = os.path.join(output_dir, save_name)

            cv2.imwrite(save_path, aug_img)

print(f"增强完成，共生成 {len(blur_strengths)} 倍于原图数量的增强图，保存在 {output_dir}")
