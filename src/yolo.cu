#include "yolo.hpp"
#include "infer.hpp"

namespace yolo{

using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)  \
    do{                                                 \
        auto ___call__ret_code__ = (call);              \
        if (___call__ret_code__ != cudaSuccess) {       \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]",        \
                #call, cudaGetErrorString(___call__ret_code__),                         \
                cudaGetErrorName(___call__ret_code__), ___call__ret_code__);                                                    \
            abort();                                                                    \
        }                                                                               \
    }while(0)

#define checkKernel(...)                                                                \
do{                                                                                     \
    {(__VA_ARGS__);}                                                                    \
    checkRuntime(cudaPeekAtLastError());                                            \
}while(0)

enum class NormType : int{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

enum class ChannelType : int{
    None          = 0,
    SwapRB        = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

    // None
    static Norm None();
};

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

    Norm out;
    out.type  = NormType::MeanStd;
    out.alpha = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std,  std,  sizeof(out.std));
    return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    out.channel_type = channel_type;
    return out;
}

Norm Norm::None(){
    return Norm();
}

const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
const int MAX_IMAGE_BOXES = 1024;
inline int upbound(int n, int align = 32){return (n + align - 1) / align * align;}
static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES){  

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= MAX_IMAGE_BOXES)
        return;

    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int MAX_IMAGE_BOXES, float threshold){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, MAX_IMAGE_BOXES);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
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

static void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float nms_threshold, float* invert_affine_matrix, float* parray, int MAX_IMAGE_BOXES, cudaStream_t stream){
    
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    /* 如果核函数有波浪线，没关系，他是正常的，你只是看不顺眼罢了 */
    checkKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES));

    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
}

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm){

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

    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // same to opencv
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    if(norm.channel_type == ChannelType::SwapRB){
        float t = c2;
        c2 = c0;  c0 = t;
    }

    if(norm.type == NormType::MeanStd){
        c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
        c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
        c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    }else if(norm.type == NormType::AlphaBeta){
        c0 = c0 * norm.alpha + norm.beta;
        c1 = c1 * norm.alpha + norm.beta;
        c2 = c2 * norm.alpha + norm.beta;
    }

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

static void warp_affine_bilinear_and_normalize_plane(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
    float* matrix_2_3, uint8_t const_value, const Norm& norm,
    cudaStream_t stream) {
    
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);
    
    checkKernel(warp_affine_bilinear_and_normalize_plane_kernel << <grid, block, 0, stream >> > (
        src, src_line_size,
        src_width, src_height, dst,
        dst_width, dst_height, const_value, matrix_2_3, norm
    ));
}

const char* type_name(Type type){
    switch(type){
    case Type::V5: return "YoloV5";
    case Type::V3: return "YoloV3";
    case Type::V7: return "YoloV7";
    case Type::X: return "YoloX";
    default: return "Unknow";
    }
}

struct AffineMatrix{
    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to){
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * get<0>(from)  * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * get<1>(from) * 0.5  + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0]*i2d[4] - i2d[1]*i2d[3];
        D = D != 0. ? double(1.)/D : double(0.);
        double A11 = i2d[4]*D, A22 = i2d[0]*D, A12 = -i2d[1]*D, A21 = -i2d[3]*D;
        double b1 = -A11*i2d[2] - A12*i2d[5];
        double b2 = -A21*i2d[2] - A22*i2d[5];

        d2i[0] = A11; d2i[1] = A12; d2i[2] = b1;
        d2i[3] = A21; d2i[4] = A22; d2i[5] = b2;
    }
};


class InferImpl : public Infer{
public:
    shared_ptr<trt::Infer> trt_;
    string engine_file_;
    Type type_;
    float confidence_threshold_;
    float nms_threshold_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_, predict_buffer_, output_boxarray_;
    int input_width_, input_height_;
    Norm normalize_;
    vector<int> network_output_dims_;
    int num_classes_ = 0;

    void adjust_memory(){
        // the inference batch_size
        int batch_size     = trt_->run_dims(0)[0];
        size_t input_numel = input_width_ * input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        predict_buffer_.gpu(batch_size * network_output_dims_[1] * network_output_dims_[2]);
        output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if(preprocess_buffers_.size() < batch_size){
            for(int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
        }
    }

    void preprocess(int ibatch, const Image& image, shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, void* stream = nullptr){

        AffineMatrix affine;
        affine.compute(make_tuple(image.width, image.height), make_tuple(input_width_, input_height_));

        size_t input_numel            = input_width_ * input_height_ * 3;
        float* input_device           = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image             = image.width * image.height * 3;
        size_t size_matrix            = upbound(sizeof(affine.d2i), 32);
        uint8_t* gpu_workspace        = preprocess_buffer->gpu(size_matrix + size_image);
        float* affine_matrix_device   = (float*)gpu_workspace;
        uint8_t* image_device         = gpu_workspace + size_matrix;

        uint8_t* cpu_workspace        = preprocess_buffer->cpu(size_matrix + size_image);
        float* affine_matrix_host     = (float*)cpu_workspace;
        uint8_t* image_host           = cpu_workspace + size_matrix;

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

        warp_affine_bilinear_and_normalize_plane(
            image_device,      image.width * 3, image.width,       image.height, 
            input_device, input_width_,    input_height_, 
            affine_matrix_device, 114, 
            normalize_, stream_
        );
    }

    bool load(const string& engine_file, Type type, float confidence_threshold, float nms_threshold){
        trt_ = trt::load(engine_file);
        if(trt_ == nullptr) return false;

        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        if(type == Type::V5 || type == Type::V3 || type == Type::V7){
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
        }else if(type == Type::X){
            //float mean[] = {0.485, 0.456, 0.406};
            //float std[]  = {0.229, 0.224, 0.225};
            //normalize_ = Norm::mean_std(mean, std, 1/255.0f, ChannelType::Invert);
            normalize_ = Norm::None();
        }else{
            INFO("Unsupport type %d", type);
        }

        auto input_dim = trt_->static_dims(0);
        network_output_dims_ = trt_->static_dims(1);
        num_classes_   = network_output_dims_[2] - 5;
        input_width_   = input_dim[3];
        input_height_  = input_dim[2];
        return true;
    }

    virtual BoxArray forward(const Image& image, void* stream = nullptr) override{
        auto output = forwards({image}, stream);
        if(output.empty()) return {};
        return output[0];
    }

    virtual vector<BoxArray> forwards(const vector<Image>& images, void* stream = nullptr) override{

        int batch_size = images.size();
        if(batch_size == 0) return {};

        auto input_dims = trt_->static_dims(0);
        input_dims[0]   = batch_size;
        if(!trt_->set_run_dims(0, input_dims)) return {};

        adjust_memory();

        cudaStream_t stream_ = (cudaStream_t)stream;
        for(int i = 0; i < images.size(); ++i)
            preprocess(i, images[i], preprocess_buffers_[i], stream);

        float* network_output = predict_buffer_.gpu();
        if(!trt_->forward({input_buffer_.gpu(), network_output}, stream)){
            INFO("Failed to tensorRT forward.");
            return {};
        }
        
        for(int ib = 0; ib < batch_size; ++ib){
            float* boxarray_device = output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float* boxarray_host   = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float* affine_matrix_device = (float*)preprocess_buffers_[ib]->gpu();
            float* network_output_device = network_output + ib * (network_output_dims_[1] * network_output_dims_[2]);
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_kernel_invoker(network_output_device, network_output_dims_[1], num_classes_, confidence_threshold_, nms_threshold_, affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES, stream_);
        }
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(), output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        vector<BoxArray> arrout(batch_size);
        for(int ib = 0; ib < batch_size; ++ib){
            float* parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count     = min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray& output = arrout[ib];
            output.reserve(count);
            for(int i = 0; i < count; ++i){
                float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                int label    = pbox[5];
                int keepflag = pbox[6];
                if(keepflag == 1){
                    output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                }
            }
        }
        return arrout;
    }
};

Infer* loadraw(const std::string& engine_file, Type type, float confidence_threshold, float nms_threshold){

    InferImpl* impl = new InferImpl();
    if(!impl->load(engine_file, type, confidence_threshold, nms_threshold)){
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer> load(const string& engine_file, Type type, float confidence_threshold, float nms_threshold){
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(engine_file, type, confidence_threshold, nms_threshold));
}

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

}; // namespace yolo