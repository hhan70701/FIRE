/**
 * @designer ldh
 * @file kernels.cu
 * @brief CUDA kernels for Holographic Light Field Synthesis (Parameterized Grid)
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#define CLAMP_X(val, max_val) ((val) < 0 ? 0 : ((val) >= (max_val) ? (max_val) - 1 : (val)))

// ==========================================
// 1. Image Resizing Kernel (改进：中心点对齐 + 防发灰亮度保护)
// ==========================================
__global__ static void resizeGPU(unsigned char* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (i >= dstWidth || j >= dstHeight) return;

    // 【画质补丁 1】：OpenCV 严格的中心点亚像素对齐
    float scale_x = (float)srcWidth / (float)dstWidth;
    float scale_y = (float)srcHeight / (float)dstHeight;
    float orgx = (i + 0.5f) * scale_x - 0.5f;
    float orgy = (j + 0.5f) * scale_y - 0.5f;

    // 防止越界采样
    orgx = fmaxf(0.0f, fminf(orgx, (float)srcWidth - 1.0f));
    orgy = fmaxf(0.0f, fminf(orgy, (float)srcHeight - 1.0f));

    // 使用 floorf 向下取整获取锚点坐标
    int x_f = floorf(orgx);
    int y_f = floorf(orgy);

    // 计算插值权重
    float u = orgx - x_f;
    float v = orgy - y_f;

    int x0 = max(0, min(srcWidth - 1, x_f));
    int x1 = max(0, min(srcWidth - 1, x_f + 1));
    int y0 = max(0, min(srcHeight - 1, y_f));
    int y1 = max(0, min(srcHeight - 1, y_f + 1));

    int idx_dst = j * dstWidth + i;

    for (int k = 0; k < 3; ++k) {
        float p00 = src[(y0 * srcWidth + x0) * 3 + k];
        float p01 = src[(y0 * srcWidth + x1) * 3 + k];
        float p10 = src[(y1 * srcWidth + x0) * 3 + k];
        float p11 = src[(y1 * srcWidth + x1) * 3 + k];

        float top = p00 * (1.0f - u) + p01 * u;
        float bot = p10 * (1.0f - u) + p11 * u;
        float val = top * (1.0f - v) + bot * v;

        // +0.5f 四舍五入，防止转 char 时丢失全局对比度！
        dst[idx_dst * 3 + k] = (unsigned char)(fmaxf(0.0f, fminf(255.0f, val + 0.5f)));
    }
}

extern "C" void Resize_GPU(unsigned char* src, unsigned char* dst, int src_w, int src_h, int dst_w, int dst_h) {
    dim3 thread(32, 32);
    dim3 block((dst_w + thread.x - 1) / thread.x, (dst_h + thread.y - 1) / thread.y);
    resizeGPU<<<block, thread>>>(src, dst, src_w, src_h, dst_w, dst_h);
}

// ==========================================
// 2. View Splitting & Epipolar Rectification Kernel (动态宫格 + 双精度抗锯齿优化版)
// ==========================================
// 旧版，有黑边
//// 亚像素边界平滑融合器 (设备函数)
//__device__ inline float fetch_soft_border(unsigned char* src, int x, int y, int w, int w_sub, int h_sub, int m, int n, int k) {
//    // 只有在子图有效范围内才抓取像素，超出边界平滑返回 0.0f (黑边)
//    if (x >= 0 && y >= 0 && x < w_sub && y < h_sub) {
//        int global_x = m * w_sub + x;
//        int global_y = n * h_sub + y;
//        return (float)src[(global_y * w + global_x) * 3 + k];
//    }
//    return 0.0f;
//}
//
//__global__ static void splitViews(unsigned char* src, unsigned char* dst, int w, int h, int w_sub, int h_sub, int size_sub, float *g_warp_data, int grid_cols, int grid_rows) {
//    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    // 【性能优化】：直接算出来该像素属于哪个宫格，干掉毫无必要的 for 循环
//    int m = i / w_sub; // 属于哪一列
//    int n = j / h_sub; // 属于哪一行
//
//    // 如果超出了整个动态网格的范围，直接退出
//    if (m >= grid_cols || n >= grid_rows || i >= grid_cols * w_sub || j >= grid_rows * h_sub) return;
//
//    int cam_idx = n * grid_cols + m; // 当前相机索引
//    int edata = cam_idx * 9;         // 矩阵参数偏移量
//
//    // 改进：强制双精度 (double) 计算透视矩阵，防止大分辨率下的斜边波动锯齿
//    double orgx_d = (double)(i - m * w_sub);
//    double orgy_d = (double)(j - n * h_sub);
//
//    double X0 = (double)g_warp_data[0 + edata] * orgx_d + (double)g_warp_data[1 + edata] * orgy_d + (double)g_warp_data[2 + edata];
//    double Y0 = (double)g_warp_data[3 + edata] * orgx_d + (double)g_warp_data[4 + edata] * orgy_d + (double)g_warp_data[5 + edata];
//    double W0 = (double)g_warp_data[6 + edata] * orgx_d + (double)g_warp_data[7 + edata] * orgy_d + (double)g_warp_data[8 + edata];
//
//    double xxx = X0 / W0;
//    double yyy = Y0 / W0;
//
//    // 转回 float 用于插值计算
//    int x_f = floor(xxx);
//    int y_f = floor(yyy);
//    float a = (float)(xxx - (double)x_f);
//    float b = (float)(yyy - (double)y_f);
//
//    // 提前算好该写入的起始位置 (注意：此处对应 size_sub 的偏移量)
//    int dst_idx = ((j - n * h_sub) * w_sub + i - m * w_sub + cam_idx * size_sub) * 3;
//
//    for (int k = 0; k < 3; ++k) {
//        // 调用亚像素融合器提取四个角的像素，消除黑色边缘的锯齿
//        float p00 = fetch_soft_border(src, x_f, y_f, w, w_sub, h_sub, m, n, k);
//        float p01 = fetch_soft_border(src, x_f + 1, y_f, w, w_sub, h_sub, m, n, k);
//        float p10 = fetch_soft_border(src, x_f, y_f + 1, w, w_sub, h_sub, m, n, k);
//        float p11 = fetch_soft_border(src, x_f + 1, y_f + 1, w, w_sub, h_sub, m, n, k);
//
//        // 双线性插值计算
//        float top = p00 * (1.0f - a) + p01 * a;
//        float bot = p10 * (1.0f - a) + p11 * a;
//        float val = top * (1.0f - b) + bot * b;
//
//        // 通道反转存储 (BGR -> RGB) 或者保持你原来的格式
//        // 加上 +0.5f 保持对比度不流失，并安全截断在 [0, 255] 之间
//        dst[dst_idx + (2 - k)] = (unsigned char)(fmaxf(0.0f, fminf(255.0f, val + 0.5f)));
//    }
//}
//
//extern "C" void SplitViews(unsigned char* src, unsigned char* dst, int w, int h, int w_sub, int h_sub, int size_sub, float *g_warp_data, int grid_cols, int grid_rows) {
//    dim3 thread(32, 32);
//    dim3 block((w + thread.x - 1) / thread.x, (h + thread.y - 1) / thread.y);
//    splitViews<<<block, thread>>>(src, dst, w, h, w_sub, h_sub, size_sub, g_warp_data, grid_cols, grid_rows);
//}

__device__ inline float fetch_soft_border(unsigned char* src, int px, int py, int w, int w_sub, int h_sub, int m, int n, int channel) {
    // 边界钳制 (Clamp-to-Edge)：防止双线性插值时越界，消除边缘硬截断产生的黑边/高频伪影
    // 若超边界，则用边界像素填充
    px = px < 0 ? 0 : (px >= w_sub ? w_sub - 1 : px);
    py = py < 0 ? 0 : (py >= h_sub ? h_sub - 1 : py);

    // 将局部子图坐标映射至全局源图像坐标
    int src_x = px + m * w_sub;
    int src_y = py + n * h_sub;

    // 基于 HWC (交错排布) 格式读取像素值
    return (float)src[(src_y * w + src_x) * 3 + channel];
}


__global__ static void splitViews(unsigned char* src, unsigned char* dst, int w, int h, int w_sub, int h_sub, int size_sub, float *g_warp_data, int grid_cols, int grid_rows) {
    // 获取当前线程在全局网格中的二维绝对索引 (对应目标图像的像素坐标)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算当前像素属于哪一个子视图 (行 n，列 m)
    int m = i / w_sub;
    int n = j / h_sub;

    // 线程越界保护
    if (m >= grid_cols || n >= grid_rows || i >= grid_cols * w_sub || j >= grid_rows * h_sub) return;

    // 计算当前子视图的线性索引及对应单应性矩阵在数组中的偏移量
    int view_idx = n * grid_cols + m;
    int edata = view_idx * 9;

    // 计算当前像素在当前子视图内的局部相对坐标
    double orgx_d = (double)(i - m * w_sub);
    double orgy_d = (double)(j - n * h_sub);

    // 应用逆单应性矩阵 (Inverse Homography) 计算源图像中的映射坐标
    double X0 = (double)g_warp_data[0 + edata] * orgx_d + (double)g_warp_data[1 + edata] * orgy_d + (double)g_warp_data[2 + edata];
    double Y0 = (double)g_warp_data[3 + edata] * orgx_d + (double)g_warp_data[4 + edata] * orgy_d + (double)g_warp_data[5 + edata];
    double W0 = (double)g_warp_data[6 + edata] * orgx_d + (double)g_warp_data[7 + edata] * orgy_d + (double)g_warp_data[8 + edata];

    // 齐次坐标归一化
    double xxx = X0 / W0;
    double yyy = Y0 / W0;

    // 提取映射坐标的整数部分与小数部分，用于双线性插值
    int x_f = floor(xxx);
    int y_f = floor(yyy);
    float u = (float)(xxx - (double)x_f);
    float v = (float)(yyy - (double)y_f);

    // 计算目标内存的写入基址索引
    // 排布逻辑：[视图偏移] + [子视图内部行偏移] + [子视图内部列偏移]
    int dst_idx = (j - n * h_sub) * w_sub + (i - m * w_sub) + view_idx * (w_sub * h_sub);

    // 遍历通道执行双线性插值 (Bilinear Interpolation)
    for (int k = 0; k < 3; ++k) {
        // 采集邻域四个像素点 (已包含边缘钳制保护)
        float p00 = fetch_soft_border(src, x_f, y_f, w, w_sub, h_sub, m, n, k);
        float p01 = fetch_soft_border(src, x_f + 1, y_f, w, w_sub, h_sub, m, n, k);
        float p10 = fetch_soft_border(src, x_f, y_f + 1, w, w_sub, h_sub, m, n, k);
        float p11 = fetch_soft_border(src, x_f + 1, y_f + 1, w, w_sub, h_sub, m, n, k);

        // X轴方向插值
        float top = p00 * (1.0f - u) + p01 * u;
        float bottom = p10 * (1.0f - u) + p11 * u;
        // Y轴方向插值
        float final_val = top * (1.0f - v) + bottom * v;

        // 像素截断保护并执行 BGR -> RGB 的就地转换 (2 - k)
        dst[dst_idx * 3 + (2 - k)] = (unsigned char)(final_val < 0.0f ? 0.0f : (final_val > 255.0f ? 255.0f : final_val));
    }
}

extern "C" void SplitViews(unsigned char* src, unsigned char* dst, int w, int h, int w_sub, int h_sub, int size_sub, float *g_warp_data, int grid_cols, int grid_rows) {
    // 定义二维线程块大小，32x32 = 1024 达到当前 GPU 架构单 Block 线程数上限，最大化并行度
    dim3 thread(32, 32);
    // 依据源图像尺寸动态计算所需的 Grid 尺寸
    dim3 block((w + thread.x - 1) / thread.x, (h + thread.y - 1) / thread.y);

    // 启动核函数
    splitViews<<<block, thread>>>(src, dst, w, h, w_sub, h_sub, size_sub, g_warp_data, grid_cols, grid_rows);
}

// ==========================================
// 3. TensorRT Input Packaging
// ==========================================
__global__ void packTRTInputs(unsigned char* dst_views, float* d_I0, float* d_I1, int w, int h, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && b < batch_size) {
        int channel_stride = h * w;
        int src0_idx = (b * channel_stride + y * w + x) * 3;
        int src1_idx = ((b + 1) * channel_stride + y * w + x) * 3;
        int dst_idx = b * 3 * channel_stride + y * w + x;

        d_I0[dst_idx + 0 * channel_stride] = dst_views[src0_idx + 2] / 255.0f;
        d_I0[dst_idx + 1 * channel_stride] = dst_views[src0_idx + 1] / 255.0f;
        d_I0[dst_idx + 2 * channel_stride] = dst_views[src0_idx + 0] / 255.0f;

        d_I1[dst_idx + 0 * channel_stride] = dst_views[src1_idx + 2] / 255.0f;
        d_I1[dst_idx + 1 * channel_stride] = dst_views[src1_idx + 1] / 255.0f;
        d_I1[dst_idx + 2 * channel_stride] = dst_views[src1_idx + 0] / 255.0f;
    }
}

extern "C" void PrepareTRTInputs(unsigned char* dst_views, float* d_I0, float* d_I1, int w, int h, int batch_size) {
    dim3 thread(16, 16, 1);
    dim3 block((w + thread.x - 1) / thread.x, (h + thread.y - 1) / thread.y, batch_size);
    packTRTInputs<<<block, thread>>>(dst_views, d_I0, d_I1, w, h, batch_size);
}

// ==========================================
// Sub-pixel Bilinear Sampler
// ==========================================
__device__ inline float bilinear_sample_gpu(unsigned char* img_ptr, int offset, float x, float y, int width, int height, int k) {
    int x_f = floorf(x);
    int y_f = floorf(y);
    float u = x - x_f;
    float v = y - y_f;

    int x0 = max(0, min(width - 1, x_f));
    int x1 = max(0, min(width - 1, x_f + 1));
    int y0 = max(0, min(height - 1, y_f));
    int y1 = max(0, min(height - 1, y_f + 1));

    float p00 = img_ptr[(offset + y0 * width + x0) * 3 + k];
    float p01 = img_ptr[(offset + y0 * width + x1) * 3 + k];
    float p10 = img_ptr[(offset + y1 * width + x0) * 3 + k];
    float p11 = img_ptr[(offset + y1 * width + x1) * 3 + k];

    float top = p00 * (1.0f - u) + p01 * u;
    float bot = p10 * (1.0f - u) + p11 * u;
    return top * (1.0f - v) + bot * v;
}

// ==========================================
// 4. Optical Flow Encoding & Interleaving Kernel
// ==========================================
__global__ static void backencode_with_mask(
        unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst,
        int width, int height, int virtualnum, bool ifReverse,
        int outw, int outh, int viewnum, float LineNum,
        float InclinationAngle, float MoveValue, float ZeroValue,
        int input_number) // <--- 新增动态防越界参数
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * outw + i;

    if (MoveValue < 0) MoveValue = viewnum - abs(MoveValue);
    int midnum = viewnum / 2;
    float wid = ZeroValue * width / 100.0f;

    if (j < outh && i < outw)
    {
        float step_value = 1.0f / viewnum * LineNum;
        for (int k = 0; k < 3; k++)
        {
            float value_pixel = i * 3 + 3 * j * InclinationAngle + k;
            float judge_value = value_pixel;
            if (LineNum != 0) judge_value = value_pixel - floorf(value_pixel / LineNum) * LineNum;
            if (judge_value < 0) judge_value += LineNum;

            int view_point_number = floorf(judge_value / step_value);
            int thisnum = (view_point_number % viewnum + (viewnum - (int)MoveValue)) % viewnum;
            if (!ifReverse) thisnum = viewnum - thisnum - 1;

            float movepixel = wid * (thisnum - midnum) / (float)midnum;
            float orgx = i * width / (outw * 1.0f) + movepixel;
            float orgy = j * height / (outh * 1.0f);

            int xxx_min = floorf(orgx);
            int xxx_max = ceilf(orgx);
            int yyy_min = floorf(orgy);
            int yyy_max = ceilf(orgy);

            float a = xxx_max - orgx;
            float b = yyy_max - orgy;

            if ((yyy_min + 1) < height && (xxx_min + 1) < width &&  xxx_min >= 0 && yyy_min >= 0)
            {
                int sparsenum = floorf(thisnum / (virtualnum + 1));
                
                // === 关键修改：动态防越界 ===
                if (sparsenum >= input_number) sparsenum = input_number - 1;
                if (sparsenum < 0) sparsenum = 0;

                int modnum = thisnum % (virtualnum + 1);
                float t = (float)modnum / (virtualnum + 1);
                float scale0 = 2.0f * t;
                float scale1 = 2.0f * (1.0f - t);

                int HW = height * width;
                int b_idx = sparsenum;

                int c_minmin = yyy_min * width + xxx_min;
                int c_maxmin = yyy_max * width + xxx_min;
                int c_minmax = yyy_min * width + xxx_max;
                int c_maxmax = yyy_max * width + xxx_max;


                // 加入y轴光流，改善一些抖的问题 2026年4月20日08:34:08
                // 恢复 2D 光流！同时提取水平 (U) 和 垂直 (V) 光流通道
                float* p_u0 = d_flow + b_idx * 4 * HW + 0 * HW; // 前向 X
                float* p_v0 = d_flow + b_idx * 4 * HW + 1 * HW; // 前向 Y (新增)
                float* p_u1 = d_flow + b_idx * 4 * HW + 2 * HW; // 后向 X
                float* p_v1 = d_flow + b_idx * 4 * HW + 3 * HW; // 后向 Y (新增)
                float* p_m  = d_mask + b_idx * HW;

#define B_INTERP(ptr) (ptr[c_minmin]*a*b + ptr[c_maxmin]*a*(1.0f-b) + ptr[c_minmax]*(1.0f-a)*b + ptr[c_maxmax]*(1.0f-a)*(1.0f-b))

                // X 轴和 Y 轴光流全部参与双线性插值计算
                float dx0 = B_INTERP(p_u0) * scale0;
                float dy0 = B_INTERP(p_v0) * scale0; // 新增 Y 轴偏移
                float dx1 = B_INTERP(p_u1) * scale1;
                float dy1 = B_INTERP(p_v1) * scale1; // 新增 Y 轴偏移
                float m   = B_INTERP(p_m);
#undef B_INTERP

                // Apply Sigmoid to mask to prevent overflow from raw logits
//                if (m < 0.0f || m > 1.0f) {       //  bug fixed 2024.04.18 by ldh
                m = 1.0f / (1.0f + expf(-m));
//                }
                m = fmaxf(0.0f, fminf(1.0f, m));

                // Apply sub-pixel shift on BOTH X-axis and Y-axis (打破极线强制对齐的限制)
                float warp_x0 = orgx + dx0;
                float warp_y0 = orgy + dy0; // 加上 Y 轴偏移
                float warp_x1 = orgx + dx1;
                float warp_y1 = orgy + dy1; // 加上 Y 轴偏移

                int offset0 = sparsenum * HW;
                int offset1 = (sparsenum + 1) * HW;

                float sample0 = bilinear_sample_gpu(d_src, offset0, warp_x0, warp_y0, width, height, k);
                float sample1 = bilinear_sample_gpu(d_src, offset1, warp_x1, warp_y1, width, height, k);

                float final_weight0 = (1.0f - t) * m;
                float final_weight1 = t * (1.0f - m);
                float weight_sum = final_weight0 + final_weight1;
                if(weight_sum > 0) {
                    final_weight0 /= weight_sum;
                    final_weight1 /= weight_sum;
                } else {
                    final_weight0 = 0.5f; final_weight1 = 0.5f;
                }

                d_dst[idx * 3 + 2 - k] = (unsigned char)(final_weight0 * sample0 + final_weight1 * sample1);
            }
            else
            {
                d_dst[idx * 3 + k] = 0;
            }
        }
    }
}

extern "C" void BackEncode(
        unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst,
        int w, int h, int virtualnum, bool ifReverse, int outw, int outh,
        int viewnum, float LineNum, float InclinationAngle, float MoveValue, float ZeroValue, int input_number)
{
    dim3 thread(32, 32);
    dim3 block((outw + thread.x - 1) / thread.x, (outh + thread.y - 1) / thread.y);
    cudaDeviceSynchronize();
    backencode_with_mask<<<block, thread>>>(
            d_src, d_flow, d_mask, d_dst, w, h, virtualnum, ifReverse,
            outw, outh, viewnum, LineNum, InclinationAngle, MoveValue, ZeroValue, input_number);
}


// =========================================================================
// 验证对照组：Fast版插值核函数 (仅推理1次，用 2.0*t 强行缩放光流和 Mask)
// =========================================================================
__global__ static void generate_novel_view_fast_kernel(
        unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst_view,
        int width, int height, int pair_idx, float t)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int HW = height * width;
    int idx = y * width + x;

    float* p_u0 = d_flow + pair_idx * 4 * HW + 0 * HW;
    float* p_v0 = d_flow + pair_idx * 4 * HW + 1 * HW;
    float* p_u1 = d_flow + pair_idx * 4 * HW + 2 * HW;
    float* p_v1 = d_flow + pair_idx * 4 * HW + 3 * HW;
    float* p_m  = d_mask + pair_idx * HW;

    // 1. 恢复 Sigmoid (因为引擎吐出的是未激活值)
    float m = 1.0f / (1.0f + expf(-p_m[idx]));
    m = fmaxf(0.0f, fminf(1.0f, m));

    // 2. 【复刻原工程逻辑】：引擎只给 t=0.5 的光流，我们用 t 强行线性缩放
    float scale0 = 2.0f * t;
    float scale1 = 2.0f * (1.0f - t);

    float dx0 = p_u0[idx] * scale0;
    float dy0 = p_v0[idx] * scale0;
    float dx1 = p_u1[idx] * scale1;
    float dy1 = p_v1[idx] * scale1;

    // 计算映射坐标并钳制边界
    float warp_x0 = fmaxf(0.0f, fminf(width - 1.0f, x + dx0));
    float warp_y0 = fmaxf(0.0f, fminf(height - 1.0f, y + dy0));
    float warp_x1 = fmaxf(0.0f, fminf(width - 1.0f, x + dx1));
    float warp_y1 = fmaxf(0.0f, fminf(height - 1.0f, y + dy1));

    int offset0 = pair_idx * HW;
    int offset1 = (pair_idx + 1) * HW;

    for (int k = 0; k < 3; k++) {
        float sample0 = bilinear_sample_gpu(d_src, offset0, warp_x0, warp_y0, width, height, k);
        float sample1 = bilinear_sample_gpu(d_src, offset1, warp_x1, warp_y1, width, height, k);

        // 3. 【复刻原工程逻辑】：强加时间衰减公式
        float final_weight0 = (1.0f - t) * m;
        float final_weight1 = t * (1.0f - m);

        float weight_sum = final_weight0 + final_weight1;
        if(weight_sum > 0.0f) {
            final_weight0 /= weight_sum;
            final_weight1 /= weight_sum;
        } else {
            final_weight0 = 0.5f; final_weight1 = 0.5f;
        }

        // 写入结果 (+0.5f 保证测试公平，不丢失基础亮度)
        d_dst_view[idx * 3 + 2 - k] = (unsigned char)(fmaxf(0.0f, fminf(255.0f, final_weight0 * sample0 + final_weight1 * sample1 + 0.5f)));

//        d_dst_view[idx * 3 + 2 - k] = (unsigned char)(m*sample0 + (1-m)*sample1 );
    }
}

extern "C" void GenerateNovelView_Fast(
        unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst_view,
        int width, int height, int pair_idx, float t) {
    dim3 thread(32, 32);
    dim3 block((width + thread.x - 1) / thread.x, (height + thread.y - 1) / thread.y);
    generate_novel_view_fast_kernel<<<block, thread>>>(d_src, d_flow, d_mask, d_dst_view, width, height, pair_idx, t);
}