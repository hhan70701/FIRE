# 🚀 Real-Time 3D Light Field Rendering Engine

**基于 TensorRT 与 CUDA 的实时三维光场渲染引擎**

本项目是一个面向裸眼 3D 光场显示器的高性能、实时光场合成与渲染管线。基于 C++ 编写，深度集成了 **TensorRT** (深度学习光流推理)、**CUDA** (并行图像处理与光场交织) 以及 **OpenGL** (零拷贝实时渲染)。支持将多相机阵列视频或单路视频实时转化为超高分辨率（如 8K）的交织光场图像。
![video-lightfield](../FIRE_C++/asset/rife_lf.png)

## ✨ 核心特性 (Key Features)

- ⚡ **极致性能 (Extreme Performance)**：完全抛弃 Python，采用全 C++ 管线。深度学习模型（基于 RIFE 的双向光流与掩码预测）经 ONNX 导出并使用 TensorRT 引擎加速，渲染过程零 CPU 介入。
- 📏 **严格极线几何约束 (Epipolar Constrained)**：内置极线校正逻辑，物理级锁死 Y 轴（垂直方向）光流噪声，仅保留水平视差偏移，彻底消除深度学习模型产生的垂直抖动伪影，并适应三维光场显示器的特点。
- 🔍 **亚像素级精度 (Sub-pixel Accuracy)**：GPU 内核中手搓了**高精度的双线性亚像素采样器** (Bilinear Sub-pixel Sampler) 与 Mask Sigmoid 防溢出保护，彻底解决图像边缘拉扯、重影和颜色溢出问题。
- 🎥 **8K 实时零拷贝渲染 (Zero-copy Rendering)**：使用 CUDA-OpenGL 互操作性（PBO），将 CUDA 合成好的 8K 巨大光栅图像直接映射到 GPU 纹理显存，避免了极其耗时的 Device-to-Host 内存拷贝。
- ⚙️ **热插拔配置 (Config-Driven)**：提供轻量级 `config.txt` 解析，动态加载硬件参数、模型路径、显示器光栅映射关系。

------

## 📐 核心数学原理与渲染算法 (Mathematical Principles & Rendering Engine)

本引擎不仅实现了极速的工程管线，还在底层 CUDA 算子中严格遵循了多视几何与光场编码的数学定律，确保了 8K 物理级光场的无伪影重构。

### 1. 多视点单应性极线校正 (Homography & Epipolar Rectification)
在 `One2Nine` 核函数中，输入视频或多机阵列图像首先需要进行极线对齐，以满足水平光栅阵列的光学要求。系统利用预标定的单应性矩阵 $\mathbf{H}$ 对像素坐标进行二维投影变换：

$$
\begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{00} & h_{01} & h_{02} \\ h_{10} & h_{11} & h_{12} \\ h_{20} & h_{21} & h_{22} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

变换后的目标像素坐标 $(x_{dst}, y_{dst})$ 映射为：

$$
x_{dst} = \frac{X'}{W'}, \quad y_{dst} = \frac{Y'}{W'}
$$

同时，在核函数内部引入严格的越界拦截（Boundary Clamp），防止齐次除法导致异常坐标引发的显存溢出（`Illegal Memory Access`）。

### 2. 亚像素光栅映射 (Sub-pixel Epipolar Constrained Ray-tracing)
三维光场屏幕的物理光栅（Lenticular Lens / Parallax Barrier）会导致 8K 屏幕上的空间坐标 $(i, j)$ 需要精细映射到多个离散视点 $V_{num}$ 中。本算法基于倾角 $\theta$ (`InclinationAngle`) 和节距 $L$ (`LineNum`)，计算任意屏幕像素所属的视点映射 $v$：

$$
P(i, j) = 3i + 3j \cdot \theta + c
$$

$$
v(i, j) = \lfloor \frac{P(i, j) \bmod L}{L / V_{num}} \rfloor
$$

### 3. 光流驱动的亚像素双向插值 (Bidirectional Flow-based Sub-pixel Interpolation)
这是 `BackEncode` 核函数的核心逻辑。深度学习网络 (TensorRT) 输出的为两个相邻关键帧间的光流场 $F_{0 \to 1}$ 与 $F_{1 \to 0}$。
为彻底消除神经网络引入的垂直噪声，并适应三维光场显示**系统在底层将 Y 轴（垂直）光流分量强制锁死为 $0$**，使得采样偏移严格局限在核线（水平线）上：

$$
\begin{aligned}
\Delta x_0 &= F_{0 \to 1}^u \cdot 2t, \quad &\Delta y_0 &= 0 \\
\Delta x_1 &= F_{1 \to 0}^u \cdot 2(1-t), \quad &\Delta y_1 &= 0
\end{aligned}
$$

随后，使用高维双线性插值 (Bilinear Interpolation) 获取亚像素偏移 $(x + \Delta x, y)$ 处的 RGB 色彩响应，彻底消除传统最邻近插值带来的“台阶状”锯齿与拉扯现象：

$$
I(x+u, y+v) \approx (1-u)(1-v)I_{00} + u(1-v)I_{01} + (1-u)v I_{10} + uv I_{11}
$$

### 4. Mask 权重与 Sigmoid 溢出保护 (Mask-guided Temporal Blending)
神经网络输出的遮罩张量 $M_{logits}$ 是未激活的连续标量分布。如果直接将其乘入颜色空间，会导致数值爆炸（即赛博朋克色带或黑色空洞）。
我们在 CUDA 寄存器内手动实现了 Sigmoid 激活与截断逻辑，将遮罩映射回严谨的概率空间 $[0, 1]$：

$$
M = \text{clamp}\left( \frac{1}{1 + e^{-M_{logits}}}, \, 0, \, 1 \right)
$$

最终的 8K 屏幕级光场像素 $C_{final}$ 通过时间步 $t$ 与遮罩 $M$ 实现平滑的双向光流交织：

$$
C_{final} = \frac{(1-t)M \cdot I_0(x+\Delta x_0) + t(1-M) \cdot I_1(x+\Delta x_1)}{(1-t)M + t(1-M)}
$$

------

## 🛠️ 环境依赖 (Prerequisites)

本项目在 Linux (Ubuntu 20.04 / 22.04) 环境下开发与测试。在编译之前，请确保系统中已安装以下依赖：

- **NVIDIA 驱动** & **CUDA Toolkit** (建议版本 11.8 或 12.2，支持 Ampere / Ada Lovelace 架构如 RTX 3090 / 4090)
- **TensorRT** (测试版本：8.6.1.6)
- **OpenCV** (C++ 版本，建议 4.x 及以上)
- **OpenGL 核心库**：`GL`, `GLEW`, `GLFW3`
- **CMake** (3.16 及以上版本)

------

## 🚀 编译与运行 (Build & Run)

1. **克隆本项目**：

   Bash

   ```
   git clone https://github.com/YourUsername/Practical-RIFE-LightField.git
   cd Practical-RIFE-LightField
   ```

2. **配置与编译**：

   请确保你在 `CMakeLists.txt` 中修改了自己机器上正确的 CUDA 和 TensorRT 绝对路径，然后执行：

   Bash

   ```
   mkdir build && cd build
   cmake ..
   make -j16
   ```

3. **配置参数 (`config.txt`)**：

   项目使用纯文本配置文件进行管理。请在可执行文件同级目录下创建或修改 `config.txt`：

   ```
    # --- 核心文件路径 (File Paths) ---
    # 推理引擎需要和输入宫格数、输入分辨率大小严格对应
    ENGINE_PATH = rife_flow_mask_2x2.engine
    VIDEO_PATH = video_2x2.mp4
    MATRIX_PATH = matrix_data_4.txt
    
    # --- 硬件与分辨率 (Hardware & Resolution) ---
    # 输入网络的视点分辨率，需要时32的整倍数
    WIDTH = 1920
    HEIGHT = 1088
    # 光场渲染分辨率
    OUT_WIDTH = 7680
    OUT_HEIGHT = 4320
    
    # --- 光场与算法参数 (Algorithm Parameters) ---
    # 生成虚拟视点数量
    VIRTUAL_NUMBER = 11
    # 输入宫格数=IGRID_COLS*GRID_ROWS
    IGRID_COLS=2
    GRID_ROWS=2
    # 宫格数-1
    INPUT_NUMBER=3
    # 视点总数量=INPUT_NUMBER*VIRTUAL_NUMBER+输入宫格数
    VIEW_NUM = 96
    # 线数、倾角、偏移和零平面
    LINE_NUM = 28.07911
    INCLINATION_ANGLE = 0.151
    MOVE_VALUE = 15.0
    ZERO_VALUE = 0.0
   ```

4. **一键启动**：

   Bash

   ```
   ./RIFE_TRT_CPP
   ```
------
![logs](../FIRE_C++/asset/logs.png)

## 🧩 核心代码架构 (Code Architecture)

本管线分为两大核心模块：**调度与渲染控制模块** (`main.cpp`) 以及 **GPU 并行运算算子** (`kernels.cu`)。

### 📄 `main.cpp`

- **配置加载与资源分配**：解析 `config.txt`，在显存 (Device Memory) 中预先分配好视频帧、光流张量、九宫格缓存的超大连续内存。
- **TensorRT 动态绑定**：自动扫描引擎的输入输出接口（`I0`, `I1`, `flow`, `mask`），无论导出的 ONNX 节点名称和顺序如何，均能通过动态字符串匹配正确绑定显存指针。
- **渲染循环 (Main Loop)**：
  1. CPU 使用 OpenCV (`VideoCapture`) 读取一帧视频并推入显存。
  2. 依序触发 CUDA 核函数进行 Resize、极线校正切分、色彩空间转换。
  3. 执行 `context->enqueueV2` 进行 TensorRT 光流计算。
  4. 触发最终的 `BackEncode` 视点插值交织合成大图。
  5. 映射 PBO 缓存，OpenGL 直接将显存数据渲染为全屏四边形纹理。

### 📄 `kernels.cu`

包含项目最底层的 CUDA 性能代码，所有算法均下沉至 GPU：

- `Resize_GPU`: 自定义极速双线性缩放算子。
- `One2Nine`: 结合单应性变换矩阵 (Homography Matrix)，将输入图像精确切分、变形、极线对齐至九宫格排布，带严密的显存越界保护。
- `packTRTInputs`: 处理硬件底层的图像格式差异，执行 `HWC -> CHW`、`uint8 -> float32` 并在显存级别完成 `BGR -> RGB` 的翻转。
- **`BackEncode` (核心)**：
  - **单向光流双线性采样**：读取 TensorRT 生成的 X 轴光流（摒弃 Y 轴干扰），结合光场屏幕物理公式计算视点时间步 $t$。
  - **Mask 平滑融合**：引入 Sigmoid 防爆色机制，**应用 1-t和 t权重双向混合左右视点像素，在亚像素精度下生成最终交织的光场图。**

## 🤝 贡献与交流 (Contributing)

本项目证明了深度学习与传统光场几何结合在 C++ 部署侧的巨大潜力。如果你在运行中遇到光流漂移、显存报错或 OpenGL 渲染异常等问题，欢迎在 Issues 中带上终端日志与你的 GPU 硬件环境，随时交流或提交 PR！
