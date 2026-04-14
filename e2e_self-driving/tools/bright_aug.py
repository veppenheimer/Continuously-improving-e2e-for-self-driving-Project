import os
import cv2
import albumentations as A

# ==== 配置参数 ====
input_dir = 'data/raw'       # 原始图像目录
output_dir = 'data/aug3'     # 增强后图像目录
os.makedirs(output_dir, exist_ok=True)

# ==== 运动模糊强度配置 ====
blur_strengths = [5, 9, 13]  # 多种模糊强度

# ==== 递增索引 ====
new_index = 0

# ==== 遍历图像并增强 ====
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === 从文件名中提取转向角 ===
        name, _ = os.path.splitext(filename)
        try:
            # 假设原始文件名格式为 xxx_angle.jpg
            steering_angle_str = name.split('_')[-1]
        except Exception:
            steering_angle_str = '0.0'  # 提取失败默认值

        for blur_size in blur_strengths:
            transform = A.Compose([
                A.MotionBlur(blur_limit=(blur_size, blur_size), p=1.0),
            ])
            augmented = transform(image=img_rgb)
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

            # === 保存文件 ===
            save_name = f"{new_index}_{steering_angle_str}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, aug_img)
            new_index += 1

print(f"增强完成，共生成 {new_index} 张图像，保存在 {output_dir}")
