import cv2
import numpy as np
import os

# ================= 配置参数 =================
INPUT_VIDEO = "2024-12-04 16-30-09.mp4"           # 原始的 3x3 视频路径
OUTPUT_VIDEO = "video_2x2.mp4"                    # 输出的 2x2 视频路径
INPUT_MATRIX = "matrix_data_1203.txt"             # 原始 81 个参数的 txt
OUTPUT_MATRIX = "matrix_data_4.txt"               # 新的 36 个参数的 txt

IN_ROWS = 3
IN_COLS = 3

# === 单应性矩阵坐标系缩放配置 ===
OLD_WIDTH = 1024
OLD_HEIGHT = 512

NEW_WIDTH = 1920
NEW_HEIGHT = 1088  # 根据你在 C++ 里实际放大的子图尺寸进行修改

# === 视频输出分辨率配置 (4K) ===
TARGET_OUT_WIDTH = 3840
TARGET_OUT_HEIGHT = 2160

# ================= 视角选择 =================
# 在 3x3 的九宫格中，相机的索引定义如下：
# [0]  [1]  [2]
# [3]  [4]  [5]
# [6]  [7]  [8]
# 
# 假设我们提取十字位置的四个：
SELECTED_VIEWS = [1, 3, 5, 7] 

def scale_homography(H_old, w_old, h_old, w_new, h_new):
    """
    根据分辨率变化，对 3x3 单应性矩阵进行数学重构 (相似变换)
    """
    sx = w_new / w_old
    sy = h_new / h_old
    
    # 构建缩放矩阵 S 和 逆矩阵 S_inv
    S = np.array([
        [sx,  0,  0],
        [ 0, sy,  0],
        [ 0,  0,  1]
    ])
    
    S_inv = np.array([
        [1/sx,    0,  0],
        [   0, 1/sy,  0],
        [   0,    0,  1]
    ])
    
    # 核心数学公式： H_new = S * H_old * S_inv
    H_new = S @ H_old @ S_inv
    
    # 归一化（保证右下角的元素为 1.0，维持尺度不变性）
    H_new = H_new / H_new[2, 2]
    
    return H_new

def process_matrix():
    print(">>> 正在提取并缩放单应性矩阵...")
    if not os.path.exists(INPUT_MATRIX):
        print(f"❌ 找不到矩阵文件: {INPUT_MATRIX}")
        return

    with open(INPUT_MATRIX, 'r') as f:
        # split() 自动处理空格和换行
        data = f.read().split()
    
    floats = [float(x) for x in data]
    if len(floats) != IN_ROWS * IN_COLS * 9:
        print(f"⚠️ 警告：原矩阵参数数量 {len(floats)} 不是标准的 81 个！")
        
    out_floats = []
    # 按 SELECTED_VIEWS 的顺序提取并缩放对应的 3x3 矩阵参数
    for view_idx in SELECTED_VIEWS:
        start = view_idx * 9
        end = start + 9
        
        # 将 9 个数字转换为 3x3 的 numpy 矩阵
        H_old = np.array(floats[start:end]).reshape(3, 3)
        
        # 对矩阵进行分辨率缩放
        H_new = scale_homography(H_old, OLD_WIDTH, OLD_HEIGHT, NEW_WIDTH, NEW_HEIGHT)
        
        # 展平回 9 个元素的一维数组
        out_floats.extend(H_new.flatten().tolist())
        
    # 写入新文件：按行写入，一行只写一个数字，一共 36 行，完美适配 C++ 读取逻辑
    with open(OUTPUT_MATRIX, 'w') as f:
        for val in out_floats:
            f.write(f"{val:.6f}\n")
            
    print(f"✅ 成功提取 {len(SELECTED_VIEWS)} 个相机的矩阵参数！")
    print(f"✅ 坐标系已从 {OLD_WIDTH}x{OLD_HEIGHT} 映射到 {NEW_WIDTH}x{NEW_HEIGHT}。")
    print(f"✅ 矩阵已按“一行一个数据”的格式保存（共 36 行），文件路径：{OUTPUT_MATRIX}")

def process_video():
    print("\n>>> 正在处理视频画面...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {INPUT_VIDEO}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. 计算原视频每个子图的宽高
    sub_w = width // IN_COLS
    sub_h = height // IN_ROWS
    
    # 2. 计算拼接出的原始四宫格大小
    out_w = sub_w * 2
    out_h = sub_h * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 写入器直接使用目标的 4K 尺寸
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (TARGET_OUT_WIDTH, TARGET_OUT_HEIGHT))
    
    print(f"原视频分辨率: {width}x{height}  -->  中间裁剪图: {out_w}x{out_h}  -->  最终缩放至: {TARGET_OUT_WIDTH}x{TARGET_OUT_HEIGHT}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 创建空白的 2x2 画布
        new_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # 遍历选中的 4 个视图，依次贴到新的左上、右上、左下、右下
        for new_idx, orig_view_idx in enumerate(SELECTED_VIEWS):
            # 在 3x3 原图中的行列
            orig_r = orig_view_idx // IN_COLS
            orig_c = orig_view_idx % IN_COLS
            
            # 在 2x2 新图中的行列
            new_r = new_idx // 2
            new_c = new_idx % 2
            
            # 裁剪子图
            roi = frame[orig_r * sub_h : (orig_r + 1) * sub_h, 
                        orig_c * sub_w : (orig_c + 1) * sub_w]
                        
            # 粘贴到新画布
            new_frame[new_r * sub_h : (new_r + 1) * sub_h, 
                      new_c * sub_w : (new_c + 1) * sub_w] = roi
                      
        # === 关键步骤：缩放至 4K (3840x2160) ===
        final_frame = cv2.resize(new_frame, (TARGET_OUT_WIDTH, TARGET_OUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        out.write(final_frame)
        count += 1
        
        # 每处理 50 帧打印一次进度
        if count % 50 == 0:
            print(f"  处理进度: {count} / {total_frames} 帧")
            
    cap.release()
    out.release()
    print(f"✅ 成功生成 4K 四宫格视频，已保存至 {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process_matrix()
    process_video()