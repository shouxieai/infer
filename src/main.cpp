
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

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

yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

void perf() {
  int max_infer_batch = 16;
  int batch = 16;
  std::vector<cv::Mat> images{cv::imread("inference/car.jpg"), cv::imread("inference/gril.jpg"),
                              cv::imread("inference/group.jpg")};

  for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 3]);

  cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
  bool ok = cpmi.start([] { return yolo::load("yolov8n.transd.engine", yolo::Type::V8); },
                       max_infer_batch);

  if (!ok) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

  trt::Timer timer;
  for (int i = 0; i < 5; ++i) {
    timer.start();
    cpmi.commits(yoloimages).back().get();
    timer.stop("BATCH16");
  }

  for (int i = 0; i < 5; ++i) {
    timer.start();
    cpmi.commit(yoloimages[0]).get();
    timer.stop("BATCH1");
  }
}

void batch_inference() {
  std::vector<cv::Mat> images{cv::imread("inference/car.jpg"), cv::imread("inference/gril.jpg"),
                              cv::imread("inference/group.jpg")};
  auto yolo = yolo::load("yolov8n.transd.engine", yolo::Type::V8);
  if (yolo == nullptr) return;

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
  auto batched_result = yolo->forwards(yoloimages);
  for (int ib = 0; ib < (int)batched_result.size(); ++ib) {
    auto &objs = batched_result[ib];
    auto &image = images[ib];
    for (auto &obj : objs) {
      uint8_t b, g, r;
      tie(b, g, r) = yolo::random_color(obj.class_label);
      cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);

      auto name = cocolabels[obj.class_label];
      auto caption = cv::format("%s %.2f", name, obj.confidence);
      int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
      cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                    cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
      cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                  16);
    }
    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite(cv::format("Result%d.jpg", ib), image);
  }
}

void single_inference() {
  cv::Mat image = cv::imread("inference/car.jpg");
  auto yolo = yolo::load("yolov8n-seg.b1.transd.engine", yolo::Type::V8Seg);
  if (yolo == nullptr) return;

  auto objs = yolo->forward(cvimg(image));
  int i = 0;
  for (auto &obj : objs) {
    uint8_t b, g, r;
    tie(b, g, r) = yolo::random_color(obj.class_label);
    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);

    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

    if (obj.seg) {
      cv::imwrite(cv::format("%d_mask.jpg", i),
                  cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
      i++;
    }
  }

  printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
  cv::imwrite("Result.jpg", image);
}

int main() {
  perf();
  batch_inference();
  single_inference();
  return 0;
}