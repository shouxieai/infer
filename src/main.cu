#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"

using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class,
                                // keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;
inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }
static __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            float *invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float objectness = pitem[4];
  if (objectness < confidence_threshold) return;

  float *class_confidence = pitem + 5;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }

  confidence *= objectness;
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  cudaStream_t stream) {
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);
  checkKernel(decode_kernel<<<grid, block, 0, stream>>>(
      predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
      parray, MAX_IMAGE_BOXES));

  grid = grid_dims(MAX_IMAGE_BOXES);
  block = block_dims(MAX_IMAGE_BOXES);
  checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
}

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  float m_x1 = warp_affine_matrix_2_3[0];
  float m_y1 = warp_affine_matrix_2_3[1];
  float m_z1 = warp_affine_matrix_2_3[2];
  float m_x2 = warp_affine_matrix_2_3[3];
  float m_y2 = warp_affine_matrix_2_3[4];
  float m_z2 = warp_affine_matrix_2_3[5];

  float src_x = m_x1 * dx + m_y1 * dy + m_z1;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;
    if (y_low >= 0) {
      if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
    }

    // same to opencv
    c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
    c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
    c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
  }

  float t = c2;
  c2 = c0;
  c0 = t;


  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

static void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value,
                                                     cudaStream_t stream) {
  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
      src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
      matrix_2_3));
}


static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

struct Box {
  float left, top, right, bottom, confidence;
  int class_label;

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),
        top(top),
        right(right),
        bottom(bottom),
        confidence(confidence),
        class_label(class_label) {}
};

struct AffineMatrix {
  float i2d[6];  // image to dst(network), 2x3 matrix
  float d2i[6];  // dst to image, 2x3 matrix

  void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
    float scale_x = get<0>(to) / (float)get<0>(from);
    float scale_y = get<1>(to) / (float)get<1>(from);
    float scale = std::min(scale_x, scale_y);
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
  }
};

std::vector<Box> detect(std::shared_ptr<trt::Infer> infer, const cv::Mat& image){

  AffineMatrix affine;
  affine.compute(std::make_tuple(image.cols, image.rows), std::make_tuple(640, 640));

  unsigned char* image_device = nullptr;
  float* input_device  = nullptr;
  float* affine_matrix = nullptr;
  checkRuntime(cudaMalloc(&image_device, image.cols * image.rows * 3));
  checkRuntime(cudaMalloc(&input_device, 3 * 640 * 640 * sizeof(float)));
  checkRuntime(cudaMalloc(&affine_matrix, 6 * sizeof(float)));
  checkRuntime(cudaMemcpy(image_device, image.data, image.rows * image.cols * 3, cudaMemcpyHostToDevice));
  checkRuntime(cudaMemcpy(affine_matrix, affine.d2i, 6 * sizeof(float), cudaMemcpyHostToDevice));

  warp_affine_bilinear_and_normalize_plane(
    image_device, image.cols * 3, image.cols, image.rows, input_device, 640, 640, affine_matrix, 114, nullptr
  );

  float* predict_device = nullptr;
  auto bbox_head_dims   = infer->static_dims(1);
  int predict_numel     = infer->numel(1);  // output predict
  checkRuntime(cudaMalloc(&predict_device, predict_numel * sizeof(float)));

  infer->forward({input_device, predict_device});

  float *boxarray_device = nullptr;
  float *boxarray_host   = nullptr;
  float confidence_threshold = 0.25;
  float nms_threshold = 0.5;
  int num_classes =  bbox_head_dims[2] - 5;
  size_t boxarray_size = (1 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT) * sizeof(float);
  checkRuntime(cudaMalloc(&boxarray_device, boxarray_size));
  checkRuntime(cudaMallocHost(&boxarray_host, boxarray_size));
  checkRuntime(cudaMemset(boxarray_device, 0, sizeof(int)));
  decode_kernel_invoker(predict_device, bbox_head_dims[1], num_classes,
                            bbox_head_dims[2], confidence_threshold, nms_threshold,
                            affine_matrix, boxarray_device, MAX_IMAGE_BOXES, nullptr);
  checkRuntime(cudaMemcpy(boxarray_host, boxarray_device, boxarray_size, cudaMemcpyDeviceToHost));
  int count = min(MAX_IMAGE_BOXES, (int)*boxarray_host);

  std::vector<Box> output;
  output.reserve(count);

  for (int i = 0; i < count; ++i) {
    float *pbox = boxarray_host + 1 + i * NUM_BOX_ELEMENT;
    int label = pbox[5];
    int keepflag = pbox[6];
    if (keepflag == 1) {
      output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
    }
  }
  checkRuntime(cudaFreeHost(boxarray_host));
  checkRuntime(cudaFree(boxarray_device));
  checkRuntime(cudaFree(affine_matrix));
  checkRuntime(cudaFree(input_device));
  checkRuntime(cudaFree(image_device));
  return output;
}

int main() {
  cv::Mat image = cv::imread("inference/car.jpg");
  auto yolo = trt::load("yolov5s.engine");
  if (yolo == nullptr) return 0;

  auto objs = detect(yolo, image);
  for (auto &obj : objs) {
    uint8_t b = 0, g = 255, r  = 0;
    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);

    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
  }

  printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
  cv::imwrite("Result.jpg", image);
  return 0;
}