/**
 * @designer ldh
 * @file main.cpp
 * @brief Real-time Holographic Light Field Rendering Pipeline (Config Driven)
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include <NvInfer.h>
#include <chrono>

#include <sys/stat.h>
#include <sys/types.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

using namespace cv;
using namespace std;

// ==========================================
// 全局参数 (将由 config.txt 覆盖)
// ==========================================
string ENGINE_PATH = "rife_flow_mask.engine";
string VIDEO_PATH = "video.mp4";
string MATRIX_PATH = "matrix_data.txt";

// --- 新增网格布局参数 ---
int GRID_COLS = 3; // 默认 2x2 四宫格
int GRID_ROWS = 3;

unsigned int WIDTH = 1024;
unsigned int HEIGHT = 512;
int OutWidth = 7680;
int OutHeight = 4320;
int VIRTUAL_NUMBER = 11;
int INPUT_NUMBER = 3; // N个相机产生 N-1 对光流，4宫格默认为 3
int ViewNum = 96;
float LineNum = 28.07911f;
float InclinationAngle = 0.151f;
float MoveValue = 15.0f;
float ZeroValue = 0.0f;

unsigned int SCR_WIDTH = 7680;
unsigned int SCR_HEIGHT = 4320;

// ==========================================
// OpenGL 全局变量
// ==========================================
GLuint Buffer;
GLuint Texture;
struct cudaGraphicsResource *cuda_pbo_rsc;

// ==========================================
// CUDA Extern Interfaces (更新签名)
// ==========================================
extern "C" void Resize_GPU(unsigned char* src, unsigned char* dst, int src_w, int src_h, int dst_w, int dst_h);
// One2Nine 重命名为 SplitViews，并加入 grid_cols 和 grid_rows 参数
extern "C" void SplitViews(unsigned char* src, unsigned char* dst, int w, int h, int w_sub, int h_sub, int size_sub, float *g_warp_data, int grid_cols, int grid_rows);
extern "C" void PrepareTRTInputs(unsigned char* dst_views, float* d_I0, float* d_I1, int w, int h, int batch_size);
// BackEncode 增加 input_number 用于防止动态视差越界
extern "C" void BackEncode(unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst, int w, int h, int virtualnum, bool ifReverse, int outw, int outh, int viewnum, float LineNum, float InclinationAngle, float MoveValue, float ZeroValue, int input_number);

extern "C" void GenerateNovelView_Fast(
        unsigned char* d_src, float* d_flow, float* d_mask, unsigned char* d_dst_view,
        int width, int height, int pair_idx, float t);

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) cout << "[TensorRT] " << msg << endl;
    }
} gLogger;

string Trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

void LoadConfig(const string& filename) {
    ifstream file(filename);
    if(!file.is_open()) {
        cout << "[Warning] Cannot find '" << filename << "', using hardcoded default parameters." << endl;
        return;
    }
    string line;
    while(getline(file, line)) {
        size_t commentPos = line.find('#');
        if(commentPos != string::npos) line = line.substr(0, commentPos);
        size_t eqPos = line.find('=');
        if(eqPos != string::npos) {
            string key = Trim(line.substr(0, eqPos));
            string val = Trim(line.substr(eqPos + 1));

            if(key == "ENGINE_PATH") ENGINE_PATH = val;
            else if(key == "VIDEO_PATH") VIDEO_PATH = val;
            else if(key == "MATRIX_PATH") MATRIX_PATH = val;
            else if(key == "GRID_COLS") GRID_COLS = stoi(val);
            else if(key == "GRID_ROWS") GRID_ROWS = stoi(val);
            else if(key == "WIDTH") WIDTH = stoi(val);
            else if(key == "HEIGHT") HEIGHT = stoi(val);
            else if(key == "OUT_WIDTH") OutWidth = stoi(val);
            else if(key == "OUT_HEIGHT") OutHeight = stoi(val);
            else if(key == "VIRTUAL_NUMBER") VIRTUAL_NUMBER = stoi(val);
            else if(key == "INPUT_NUMBER") INPUT_NUMBER = stoi(val);
            else if(key == "VIEW_NUM") ViewNum = stoi(val);
            else if(key == "LINE_NUM") LineNum = stof(val);
            else if(key == "INCLINATION_ANGLE") InclinationAngle = stof(val);
            else if(key == "MOVE_VALUE") MoveValue = stof(val);
            else if(key == "ZERO_VALUE") ZeroValue = stof(val);
        }
    }
    SCR_WIDTH = OutWidth;
    SCR_HEIGHT = OutHeight;

    // 安全钳制：确保 INPUT_NUMBER 不会越界 (最大为总相机数 - 1)
    int max_input = (GRID_COLS * GRID_ROWS) - 1;
    if (INPUT_NUMBER > max_input) {
        cout << "[Warning] INPUT_NUMBER (" << INPUT_NUMBER << ") 超过相机对数上限，强制重置为 " << max_input << endl;
        INPUT_NUMBER = max_input;
    }
}

void initGL() {
    glGenBuffers(1, &Buffer);
    glBindBuffer(GL_ARRAY_BUFFER, Buffer);
    glBufferData(GL_ARRAY_BUFFER, OutWidth * OutHeight * sizeof(GLchar) * 3, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_rsc, Buffer, cudaGraphicsMapFlagsWriteDiscard);
}

void renderToPBO(unsigned char* d_data) {
    cudaDeviceSynchronize();
    cudaGraphicsMapResources(1, &cuda_pbo_rsc, 0);
    size_t bytes;
    unsigned char *d_output;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &bytes, cuda_pbo_rsc);
    cudaMemcpy(d_output, d_data, OutWidth * OutHeight * 3, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cuda_pbo_rsc, 0);
}

// ==========================================
// Load Homography Matrix (参数化加载)
// ==========================================
float* HomoWarp(string path_warpTxt, int grid_cols, int grid_rows) {
    int camera_count = grid_cols * grid_rows;
    float Warp_M[3000] = {0};
    float* d_warp_data = nullptr;

    ifstream matfile(path_warpTxt);
    if(!matfile.is_open()) {
        cerr << "[Error] Failed to open homography matrix file: " << path_warpTxt << endl;
        exit(EXIT_FAILURE);
    }

    // 每个相机拥有一个 3x3 (共9个元素) 的单应性矩阵
    for(int i = 0; i < camera_count * 9; i++){
        matfile >> Warp_M[i];
        if(matfile.fail()) {
            cout << "[Warning] Matrix read interrupted at index " << i << ". Expected " << camera_count * 9 << " values." << endl;
            break;
        }
    }
    matfile.close();

    CHECK_CUDA(cudaMalloc((void **)&d_warp_data, camera_count * 9 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_warp_data, Warp_M, camera_count * 9 * sizeof(float), cudaMemcpyHostToDevice));

    return d_warp_data;
}

int main() {
    LoadConfig("config.txt");

    if (!glfwInit()) { cerr << "glfwInit failed." << endl; return -1; }
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Holographic Synthesis (TRT)", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    initGL();

    ifstream file(ENGINE_PATH, ios::binary);
    if(!file.good()) { cerr << "[Error] Engine file not found: " << ENGINE_PATH << endl; return -1; }

    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    context->setBindingDimensions(0, nvinfer1::Dims4{INPUT_NUMBER, 3, HEIGHT, WIDTH});
    context->setBindingDimensions(1, nvinfer1::Dims4{INPUT_NUMBER, 3, HEIGHT, WIDTH});
    context->setBindingDimensions(2, nvinfer1::Dims2{INPUT_NUMBER, 1});

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // --- 动态显存分配 (替代硬编码的 3 和 9) ---
    int CAMERA_COUNT = GRID_COLS * GRID_ROWS;
    unsigned char *d_video_frame, *d_grid_resize, *d_dst_views, *d_img_dst;
    float *d_I0, *d_I1, *d_timestep, *d_flow, *d_mask;

    // 缩放后的整张大图显存
    CHECK_CUDA(cudaMalloc((void**)&d_grid_resize, GRID_COLS * WIDTH * GRID_ROWS * HEIGHT * 3));
    // 切分极线对齐后的各个子图显存
    CHECK_CUDA(cudaMalloc((void**)&d_dst_views, CAMERA_COUNT * WIDTH * HEIGHT * 3));
    CHECK_CUDA(cudaMalloc((void**)&d_img_dst, OutWidth * OutHeight * 3));

    size_t tensor_bytes = INPUT_NUMBER * 3 * HEIGHT * WIDTH * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_I0, tensor_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_I1, tensor_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_timestep, INPUT_NUMBER * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_flow, INPUT_NUMBER * 4 * HEIGHT * WIDTH * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_mask, INPUT_NUMBER * 1 * HEIGHT * WIDTH * sizeof(float)));

    vector<float> h_ts(INPUT_NUMBER, 0.5f);
    CHECK_CUDA(cudaMemcpy(d_timestep, h_ts.data(), INPUT_NUMBER * sizeof(float), cudaMemcpyHostToDevice));

    float* d_warp_data = HomoWarp(MATRIX_PATH, GRID_COLS, GRID_ROWS);

    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) { cerr << "[Error] Cannot open video stream: " << VIDEO_PATH << endl; return -1; }

    int real_w = cap.get(CAP_PROP_FRAME_WIDTH);
    int real_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    CHECK_CUDA(cudaMalloc((void**)&d_video_frame, real_w * real_h * 3));

    void* bindings[5];
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; i++) {
        string name = engine->getBindingName(i);
        if (name == "I0" || name == "input0" || name == "input") bindings[i] = d_I0;
        else if (name == "I1" || name == "input1") bindings[i] = d_I1;
        else if (name == "timestep" || name == "input2") bindings[i] = d_timestep;
        else if (name == "flow" || name == "output0" || name == "output") bindings[i] = d_flow;
        else if (name == "mask" || name == "output1") bindings[i] = d_mask;
    }

    while (cap.isOpened() && !glfwWindowShouldClose(window)) {
        Mat frame;
        cap >> frame;
        if(frame.empty()) break;

        clock_t time_start = clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 参数化调用核函数 ---
        CHECK_CUDA(cudaMemcpyAsync(d_video_frame, frame.data, frame.cols * frame.rows * 3, cudaMemcpyHostToDevice, stream));
        
        // 1. 将原图等比例缩放至 (GRID_COLS * WIDTH) x (GRID_ROWS * HEIGHT)
        Resize_GPU(d_video_frame, d_grid_resize, frame.cols, frame.rows, GRID_COLS * WIDTH, GRID_ROWS * HEIGHT);
        
        // 2. 切分与单应性变换极线对齐 (传入 GRID_COLS 和 GRID_ROWS)
        SplitViews(d_grid_resize, d_dst_views, GRID_COLS * WIDTH, GRID_ROWS * HEIGHT, WIDTH, HEIGHT, WIDTH * HEIGHT, d_warp_data, GRID_COLS, GRID_ROWS);

        // 3. 组装给 TensorRT 的数据
        PrepareTRTInputs(d_dst_views, d_I0, d_I1, WIDTH, HEIGHT, INPUT_NUMBER);

        context->enqueueV2(bindings, stream, nullptr);

        // 4. 光场融合（传入 INPUT_NUMBER，防止显存越界）
        BackEncode(d_dst_views, d_flow, d_mask, d_img_dst, WIDTH, HEIGHT, VIRTUAL_NUMBER, 0, OutWidth, OutHeight, ViewNum, LineNum, InclinationAngle, MoveValue, ZeroValue, INPUT_NUMBER);

        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> tm = t_end - t_start;
        cout << "[Performance] Algorithm Inference FPS: " << 1000.0 / tm.count() << " (" << tm.count() << " ms)" << endl;


        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // 测试保存第一帧8k光场图像
        static bool is_saved = false; // 加个静态变量，保证只存一次
        if (!is_saved) {
            cout << "正在抓取第一帧 8K 图像到 CPU..." << endl;
            unsigned char* h_debug_8k = new unsigned char[OutWidth * OutHeight * 3];
            cudaMemcpy(h_debug_8k, d_img_dst, OutWidth * OutHeight * 3, cudaMemcpyDeviceToHost);
            cv::Mat final_8k(OutHeight, OutWidth, CV_8UC3, h_debug_8k);

            // ⚠️ 建议换成绝对路径，绝对丢不了！或者直接去 cmake-build-debug 找
            string save_path = "debug_03_8k_first_frame.jpg";
            bool success = cv::imwrite(save_path, final_8k);

            if(success) {
                cout << "8K 图像保存成功！路径: " << save_path << endl;
            } else {
                cout << "8K 图像保存失败！可能是权限或磁盘空间不足。" << endl;
            }

            delete[] h_debug_8k;
            is_saved = true; // 存过一次就锁死，不再疯狂覆写硬盘
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // ======================= ▼ 生成虚拟视点测试 ▼ ===================================================================
        static int frame_counter = 0;
        frame_counter++;

        if (frame_counter <= 200) {
            // --- 动态计算当前架构的视点数量参数 ---
            int CAMERA_COUNT = GRID_COLS * GRID_ROWS; // 比如 2x2 = 4
            int NUM_PAIRS = INPUT_NUMBER;             // 比如 4宫格就是 3对
            int NUM_INTERP_PER_PAIR = 11;             // 每对中间插 11 张
            int TOTAL_VIEWS = CAMERA_COUNT + (NUM_PAIRS * NUM_INTERP_PER_PAIR); // 4 + 3*11 = 37 张

            // 1. 创建文件夹，例如 "01", "02"...
            char folder_path[256];
            sprintf(folder_path, "output_frames/%02d", frame_counter);
            mkdir(folder_path, 0777);

            // 2. 准备基础数据 (Resize 和 SplitViews 已在前面运行)
            // 注意：使用重构后的 d_dst_views
            PrepareTRTInputs(d_dst_views, d_I0, d_I1, WIDTH, HEIGHT, NUM_PAIRS);

            // 3. 显存分配用于临时存放生成的插值图
            unsigned char* d_temp_view;
            cudaMalloc((void**)&d_temp_view, WIDTH * HEIGHT * 3);
            unsigned char* h_buffer = new unsigned char[WIDTH * HEIGHT * 3];

            // 建立一个简单的内存池来暂存所有的插值图 (自动计算大小)
            std::vector<cv::Mat> interp_pool(NUM_PAIRS * NUM_INTERP_PER_PAIR);

            std::cout << "输入相机数量：" << CAMERA_COUNT << " 个..." << std::endl;
            std::cout << "Generating " << TOTAL_VIEWS << " views for Frame " << frame_counter << "..." << std::endl;

            // 1. 【核心改变】：整个过程中，只调用 1 次 TensorRT 模型！
            // 强制传入 t=0.5 告诉模型生成正中间的光流
            std::vector<float> h_ts_half(NUM_PAIRS, 0.5f);
            cudaMemcpy(d_timestep, h_ts_half.data(), NUM_PAIRS * sizeof(float), cudaMemcpyHostToDevice);

            context->enqueueV2(bindings, stream, nullptr);
            cudaStreamSynchronize(stream);

            // 2. 引擎跑完后，利用这唯一的一份 d_flow 和 d_mask，循环计算插值画面
            for (int step = 1; step <= NUM_INTERP_PER_PAIR; step++) {
                float t_val = (float)step / (NUM_INTERP_PER_PAIR + 1.0f); // 自动适应步长比例

                for (int p = 0; p < NUM_PAIRS; p++) {
                    // 调用新的 Fast 核函数，通过传入 t_val 在内部计算缩放和权重
                    GenerateNovelView_Fast(d_dst_views, d_flow, d_mask, d_temp_view, WIDTH, HEIGHT, p, t_val);

                    cudaMemcpy(h_buffer, d_temp_view, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
                    interp_pool[p * NUM_INTERP_PER_PAIR + (step - 1)] = cv::Mat(HEIGHT, WIDTH, CV_8UC3, h_buffer).clone();
                }
            }

            // 4. 正式按顺序保存所有图 (原图+插值)
            int file_idx = 1;
            for (int p = 0; p < CAMERA_COUNT; p++) {
                // A. 保存原图 (View p)
                cudaMemcpy(h_buffer, d_dst_views + p * (WIDTH * HEIGHT * 3), WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
                cv::Mat orig(HEIGHT, WIDTH, CV_8UC3, h_buffer);
                cv::cvtColor(orig, orig, cv::COLOR_RGB2BGR); // 输出是 RGB，需转为 BGR 供 OpenCV 保存

                char name[512];
                sprintf(name, "%s/%02d.jpg", folder_path, file_idx++);
                cv::imwrite(name, orig);

                // B. 如果不是最后一张原图，则插入后面紧跟的插值图
                if (p < NUM_PAIRS) {
                    for (int s = 0; s < NUM_INTERP_PER_PAIR; s++) {
                        sprintf(name, "%s/%02d.jpg", folder_path, file_idx++);
                        cv::imwrite(name, interp_pool[p * NUM_INTERP_PER_PAIR + s]);
                    }
                }
            }

            // 释放临时显存
            cudaFree(d_temp_view);
            delete[] h_buffer;
        }
        // ======================= ▲ 测试 结束 ▲ =================================================================

        renderToPBO(d_img_dst);

        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        if (!Texture) {
            glGenTextures(1, &Texture);
            glBindTexture(GL_TEXTURE_2D, Texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OutWidth, OutHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        }

        glBindTexture(GL_TEXTURE_2D, Texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OutWidth, OutHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    delete context; delete engine; delete runtime;

    cudaFree(d_video_frame); cudaFree(d_grid_resize); cudaFree(d_dst_views);
    cudaFree(d_img_dst); cudaFree(d_I0); cudaFree(d_I1);
    cudaFree(d_timestep); cudaFree(d_flow); cudaFree(d_mask);
    cudaFree(d_warp_data);

    return 0;
}