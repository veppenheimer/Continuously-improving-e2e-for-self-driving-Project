import os
import shutil
import concurrent.futures

# --------- 函数：复制单个文件 ---------
def copy_file(task):
    src, dst = task
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"❌ 复制失败: {src} -> {dst}, 错误: {e}")

# --------- 函数：获取当前最大索引 ---------
def get_max_index(folder):
    max_idx = -1
    if not os.path.exists(folder):
        return max_idx
    for fname in os.listdir(folder):
        if fname.endswith(".jpg") and "_" in fname:
            try:
                idx = int(fname.split("_")[0])
                max_idx = max(max_idx, idx)
            except:
                continue
    return max_idx

# --------- 主函数 ---------
def main():
    # 配置路径
    target_folder = "../dataset/data"
    source_folders = [
        "final/1",
        "final/1 (2)",
        "final/1 (3)",
        "final/1 (5)",
        "final/1 (6)",
        "final/1 (7)",
        "final/1 (8)",
        "final/1 (9)",
        "final/1 (10)",
        "final/1 (11)",
        "final/1 (12)",
        "final/1 (13)",
        "final/1 (14)",
        "final/1 (15)",
        "final/1 (16)",
        "final/1 (17)",
        "final/1 (18)",
        "final/1 (19)",
        "final/1 (20)",
        # 可追加更多路径
    ]

    os.makedirs(target_folder, exist_ok=True)

    # 构建复制任务
    copy_tasks = []
    current_index = get_max_index(target_folder) + 1

    for folder in source_folders:
        if not os.path.exists(folder):
            print(f"⚠️ 跳过不存在的目录：{folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".jpg") or "_" not in fname:
                continue
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            angle = parts[1]
            new_name = f"{current_index}_{angle}"
            src_path = os.path.join(folder, fname)
            dst_path = os.path.join(target_folder, new_name)
            copy_tasks.append((src_path, dst_path))
            current_index += 1

    # 并发复制
    print(f"📦 开始复制 {len(copy_tasks)} 张图片...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(copy_file, copy_tasks))

    print(f"✅ 复制完成，共复制 {len(copy_tasks)} 张图片。")

# --------- 程序入口 ---------
if __name__ == '__main__':
    main()
