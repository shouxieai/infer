
#include "yolo.hpp"
#include "infer.hpp"
#include <opencv2/opencv.hpp>
#include "cpm.hpp"

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

yolo::Image cvimg(const cv::Mat& image){
    return yolo::Image(image.data, image.cols, image.rows);
}

void perf(){
    
    int batch = 16;
    std::vector<cv::Mat> images{
        cv::imread("inference/car.jpg"),
        cv::imread("inference/gril.jpg"),
        cv::imread("inference/group.jpg")
    };

    for(int i = images.size(); i < batch; ++i)
        images.push_back(images[i % 3]);

    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
    bool ok = cpmi.start([]{
        return yolo::load("yolov5s.engine", yolo::Type::V5);
    }, batch);

    if(!ok) return;

    std::vector<yolo::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

    trt::Timer timer;
    for(int i = 0; i < 5; ++i){
        timer.start();
        cpmi.commits(yoloimages).back().get();
        timer.stop("BATCH16");
    }

    for(int i = 0; i < 5; ++i){
        timer.start();
        cpmi.commit(yoloimages[0]).get();
        timer.stop("BATCH1");
    }
}

void inference(){

    int batch = 5;
    std::vector<cv::Mat> images{
        cv::imread("inference/car.jpg"),
        cv::imread("inference/gril.jpg"),
        cv::imread("inference/group.jpg")
    };

    for(int i = images.size(); i < batch; ++i)
        images.push_back(images[i % 3]);

    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
    bool ok = cpmi.start([]{
        return yolo::load("yolov5s.engine", yolo::Type::V5);
    }, batch);

    if(!ok) return;

    std::vector<yolo::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

    auto result = cpmi.commits(yoloimages);
    for(int i = 0; i < result.size(); ++i){
        auto& image = images[i];
        auto& objs  = result[i].get();
        printf("result = %d\n", objs.size());

        for(auto& obj : objs){
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        printf("Save result to infer.jpg, %d objects\n", objs.size());
        cv::imwrite(cv::format("infer_%d.jpg", i), image);
    }
}

int main(){
    
    perf();
    // inference();
    return 0;
}