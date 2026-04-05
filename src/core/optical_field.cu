#include "optical_field.h"  // class implemented
#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>

using namespace umnholo;

 


/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

OpticalField::OpticalField()
{
    width = 0;
    height = 0;
    depth = 0;
    complex_representation = REAL_IMAGINARY;
    scale_min = 0;
    scale_max = 0;

    return;
}

/** Copy params over */
OpticalField::OpticalField(Hologram holo, OpticalFieldAllocation allocation_method)
{
    params = holo.getParams();
    width = holo.getData().getWidth();
    height = holo.getData().getHeight();
    depth = params.num_planes;
    complex_representation = REAL_IMAGINARY;
    scale_min = 0;
    scale_max = 0;
    
    cv::minMaxLoc(holo.getData().getReal().getMatData(), &scale_min, &scale_max);

    if (allocation_method == OPTICALFIELD_ALLOCATE)
    {
        data.setDataType(CV_32FC2);
        data.setWidth(width);
        data.setHeight(height);
        data.setDepth(depth);
        data.allocateCuData();
    }
    else
    {
        state = OPTICALFIELD_STATE_UNALLOCATED;
    }

    if (holo.isState(HOLOGRAM_STATE_NORM_MEAND_ZERO))
        scale = SCALE_DECONV_NORM;
    else
        scale = SCALE_UNKNOWN;

    hologram_original_mean = holo.getOriginalMean();

    return;
}

OpticalField::OpticalField(Parameters params, size_t width, size_t height, size_t depth)
{
    this->params = params;
    this->width = width;
    this->height = height;
    this->depth = depth;
    complex_representation = REAL_IMAGINARY;
    scale_min = 0;
    scale_max = 0;

    data.setDataType(CV_32FC2);
    data.setWidth(width);
    data.setHeight(height);
    data.setDepth(depth);
    data.allocateCuData();

    scale = SCALE_UNKNOWN;

    hologram_original_mean = 0;

    return;
}

void OpticalField::destroy()
{
    data.destroy();
    return;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

CuMat OpticalField::combinedXY(CmbMethod method)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->state != OPTICALFIELD_STATE_RECONSTRUCTED)
    {
        std::cout << "OpticalField::combinedXY Error: state must be one of: " << std::endl;
        std::cout << "  OPTICALFIELD_STATE_RECONSTRUCTED" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    cv::Mat result;
    if (method == MIN_CMB)
    {
        result = cv::Mat::ones(height, width, CV_32F);
        result *= 255;
    }
    else if (method == MAX_CMB)
    {
        result = cv::Mat::zeros(height, width, CV_32F);
    }
    else
    {
        std::cout << "OpticalField::combinedXY: Invalid Argument" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    CuMat single_plane;
    size_t plane_size = width * height;
    //single_plane.setCuData(data.getCuData(), data.getRows(), data.getCols(), data.getDataType());
    single_plane.allocateCuData(width, height, 1, data.getElemSize());

    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        //void* plane_start = (float*)data.getCuData() + zidx*plane_size;
        //single_plane.setCuData(plane_start);

        single_plane = this->getPlane(zidx);

        if (method == MIN_CMB)
            cv::min(result, single_plane.getMatData(), result);
        else if (method == MAX_CMB)
            cv::max(result, single_plane.getMatData(), result);
        else
            throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    //result.convertTo(result, CV_8U);
    CuMat cumat_result;
    cumat_result.setMatData(result);

    return cumat_result;
}

template <unsigned int blockSize>
__global__ void my_mean_kernel(float* data_d, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    while (i < size) { sdata[tid] += data_d[i] + data_d[i+blockSize]; i += gridSize; }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
    __syncthreads();
}

template <unsigned int blockSize>
__global__ void my_std_kernel(float* data_d, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Sum the squares for calculating standard deviation
    
    while (i < size)
    {
        sdata[tid] += data_d[i]*data_d[i] + data_d[i+blockSize]*data_d[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

void OpticalField::calcMeanStd(float &mean, float &std)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->data.getDataType() != CV_32F)
    {
        std::cout << "OpticalField::calcMeanStd: Error: data must be of type CV_32F\n" << std::endl;
        std::cout << "actual type is " << this->data.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    float* mean_d = NULL;
    float* std_d = NULL;
    CUDA_SAFE_CALL( cudaMalloc((void**)&mean_d, 1 * sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&std_d, 1 * sizeof(float)) );
    int size = width*height*depth;
    
    // Calculate Mean
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    CHECK_FOR_ERROR("before my_mean_kernel");
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    my_mean_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_mean_kernel");
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    double host_sum = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        host_sum += buffer_h[i];
    }
    double new_mean = host_sum / size;
    
    // Calculate standard deviation
    cudaMemset(buffer_d, 0, dimGrid.x);
    my_std_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_std_kernel");
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    host_sum = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        host_sum += buffer_h[i];
    }
    new_mean = new_mean * size / (size - 1);
    double new_std = sqrt(host_sum/(size-1) - new_mean*new_mean);
    
    mean = new_mean;
    std = new_std;
    
    cudaFree(buffer_d);
    cudaFree(mean_d);
    cudaFree(std_d);
    free(buffer_h);
    
    CHECK_FOR_ERROR("OpticalField::calcMeanStd");
    return;
}

template <unsigned int blockSize>
__global__ void my_sum_kernel(float2* data_d, float2* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float2 sdata2[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata2[tid].x = 0.0;
    sdata2[tid].y = 0.0;
    
    while (i < size)
    {
        sdata2[tid].x += data_d[i].x + data_d[i+blockSize].x;
        sdata2[tid].y += data_d[i].y + data_d[i+blockSize].y; 
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata2[tid].x += sdata2[tid + 256].x;
            sdata2[tid].y += sdata2[tid + 256].y;
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata2[tid].x += sdata2[tid + 128].x;
            sdata2[tid].y += sdata2[tid + 128].y;
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata2[tid].x += sdata2[tid + 64].x;
            sdata2[tid].y += sdata2[tid + 64].y;
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        if (blockSize >= 64)
        {
            sdata2[tid].x += sdata2[tid + 32].x;
            sdata2[tid].y += sdata2[tid + 32].y;
        }
        __syncthreads();
        if (blockSize >= 32)
        {
            sdata2[tid].x += sdata2[tid + 16].x;
            sdata2[tid].y += sdata2[tid + 16].y;
        }
        __syncthreads();
        if (blockSize >= 16)
        {
            sdata2[tid].x += sdata2[tid + 8].x;
            sdata2[tid].y += sdata2[tid + 8].y;
        }
        __syncthreads();
        if (blockSize >= 8)
        {
            sdata2[tid].x += sdata2[tid + 4].x;
            sdata2[tid].y += sdata2[tid + 4].y;
        }
        __syncthreads();
        if (blockSize >= 4)
        {
            sdata2[tid].x += sdata2[tid + 2].x;
            sdata2[tid].y += sdata2[tid + 2].y;
        }
        __syncthreads();
        if (blockSize >= 2)
        {
            sdata2[tid].x += sdata2[tid + 1].x;
            sdata2[tid].y += sdata2[tid + 1].y;
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        buffer[blockIdx.x].x = sdata2[0].x;
        buffer[blockIdx.x].y = sdata2[0].y;
    }
    __syncthreads();
}

void OpticalField::calcSum(float2 &sum)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->data.getDataType() != CV_32FC2)
    {
        std::cout << "OpticalField::calcSum: Error: data must be of type CV_32FC2\n" << std::endl;
        std::cout << "actual type is " << this->data.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float2);
    
    CHECK_FOR_ERROR("before my_sum_kernel");
    float2* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float2));
    my_sum_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_sum_kernel");
    
    float2* buffer_h = (float2*)malloc(dimGrid.x * sizeof(float2));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float2), cudaMemcpyDeviceToHost);
    float2 host_sum = {0,0};
    for (int i = 0; i < dimGrid.x; ++i)
    {
        host_sum.x += buffer_h[i].x;
        host_sum.y += buffer_h[i].y;
    }
    sum.x = host_sum.x;
    sum.y = host_sum.y;
}

template <unsigned int blockSize>
__global__ void my_summag_kernel(float2* data_d, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        sdata[tid] += data_d[i].x*data_d[i].x + 
                      data_d[i+blockSize].x*data_d[i+blockSize].x;
        sdata[tid] += data_d[i].y*data_d[i].y + 
                      data_d[i+blockSize].y*data_d[i+blockSize].y;
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

void OpticalField::calcSumMagnitude(float &sumsqr)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->data.getDataType() != CV_32FC2)
    {
        std::cout << "OpticalField::calcSumSquared: Error: data must be of type CV_32FC2\n" << std::endl;
        std::cout << "actual type is " << this->data.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    CHECK_FOR_ERROR("before my_sumsqr_kernel");
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    my_summag_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_sumsqr_kernel");
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    sumsqr = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sumsqr += buffer_h[i];
    }
    
    cudaFree(buffer_d);
    free(buffer_h);
    return;
}

template <unsigned int blockSize>
__global__ void my_l1norm_kernel(float2* data_d, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        sdata[tid] += sqrt(data_d[i].x*data_d[i].x + 
                      data_d[i].y*data_d[i].y);
        sdata[tid] += sqrt(data_d[i+blockSize].x*data_d[i+blockSize].x + 
                      data_d[i+blockSize].y*data_d[i+blockSize].y);
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

void OpticalField::calcL1Norm(float &sumsqr)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->data.getDataType() != CV_32FC2)
    {
        std::cout << "OpticalField::calcL1Norm: Error: data must be of type CV_32FC2\n" << std::endl;
        std::cout << "actual type is " << this->data.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    CHECK_FOR_ERROR("before my_l1norm_kernel");
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    my_l1norm_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_l1norm_kernel");
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    sumsqr = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sumsqr += buffer_h[i];
    }
    
    cudaFree(buffer_d);
    return;
}

template <unsigned int blockSize>
__global__ void my_l1norm_complex_kernel(float2* data_d, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        sdata[tid] += sqrt(data_d[i].x*data_d[i].x) + 
                      sqrt(data_d[i].y*data_d[i].y);
        sdata[tid] += sqrt(data_d[i+blockSize].x*data_d[i+blockSize].x) + 
                      sqrt(data_d[i+blockSize].y*data_d[i+blockSize].y);
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

void OpticalField::calcL1NormComplex(float &sumsqr)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->data.getDataType() != CV_32FC2)
    {
        std::cout << "OpticalField::calcL1NormComplex: Error: data must be of type CV_32FC2\n" << std::endl;
        std::cout << "actual type is " << this->data.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    CHECK_FOR_ERROR("before my_l1norm_complex_kernel");
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    my_l1norm_complex_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)this->data.getCuData(), buffer_d, size);
    CHECK_FOR_ERROR("after my_l1norm_complex_kernel");
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    sumsqr = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sumsqr += buffer_h[i];
    }
    
    cudaFree(buffer_d);
    return;
}

double OpticalField::calcEntropy()
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    assert(this->data.getDataType() == CV_32F);
    
    NppiSize roi = {width*height, depth};
    int num_bins = 256;
    int num_levels = num_bins+1;
    int line_step = width*height*sizeof(float);
    float* data_d = (float*)this->data.getCuData();
	
	// First, create or get the NppStreamContext
	//NppStreamContext nppStreamCtx = CreateNppStreamContext();
	NppStreamContext nppStreamCtx;
	NPP_SAFE_CALL(nppGetStreamContext(&nppStreamCtx));
    
	// First, find the required buffer size
	//int buffer_size = 0;
	//int alt_buffer_size = 0;
	size_t buffer_size = 0;
	size_t alt_buffer_size = 0;


	// OLD: NPP_SAFE_CALL(nppiHistogramRangeGetBufferSize_32f_C1R(roi, num_levels, &buffer_size));
	NPP_SAFE_CALL(nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(roi, num_levels, &buffer_size, nppStreamCtx));

	// OLD: NPP_SAFE_CALL(nppiMinMaxGetBufferHostSize_32f_C1R(roi, &alt_buffer_size));
	NPP_SAFE_CALL(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(roi, &alt_buffer_size, nppStreamCtx));

	buffer_size = std::max(buffer_size, alt_buffer_size);
    
    
    // Allocate buffer
    Npp8u* buffer_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_d, buffer_size));
	
	// Find the min and max of the data
	Npp32f* data_min_d;
	Npp32f* data_max_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_min_d, 1 * sizeof(Npp32f)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_max_d, 1 * sizeof(Npp32f)));

	//NppStreamContext nppStreamCtx = CreateNppStreamContext();

	// Old API (pre-CUDA 12)
	// NPP_SAFE_CALL(nppiMinMax_32f_C1R(data_d, line_step, roi, data_min_d, data_max_d, buffer_d));

	// New API with stream context
	NPP_SAFE_CALL(nppiMinMax_32f_C1R_Ctx(data_d, line_step, roi, data_min_d, data_max_d, buffer_d, nppStreamCtx));

	Npp32f data_min, data_max;
	CUDA_SAFE_CALL(cudaMemcpy(&data_min, data_min_d, 1 * sizeof(Npp32f), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&data_max, data_max_d, 1 * sizeof(Npp32f), cudaMemcpyDeviceToHost));
    
    // Compute histogram bins
    Npp32f* bins_h = (Npp32f*)malloc(num_levels*sizeof(Npp32f));
    Npp32f bin_size = (data_max - data_min) / num_bins;
    bins_h[0] = data_min;
    for (int i = 1; i < num_levels; ++i)
    {
        bins_h[i] = bins_h[i-1] + bin_size;
    }
    Npp32f* bins_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&bins_d, num_levels*sizeof(Npp32f)));
    CUDA_SAFE_CALL(cudaMemcpy(bins_d, bins_h, num_levels*sizeof(Npp32f), cudaMemcpyHostToDevice));
	
	// Fill Histogram
	Npp32s* hist_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&hist_d, num_bins * sizeof(Npp32s)));
	CUDA_SAFE_CALL(cudaMemset(hist_d, 0, num_bins * sizeof(Npp32s)));

	// Use helper function to get stream context
	//NppStreamContext nppStreamCtx = CreateNppStreamContext();

	// Old API (pre-CUDA 12)
	// NPP_SAFE_CALL(nppiHistogramRange_32f_C1R(data_d, line_step, roi, hist_d, bins_d, num_levels, buffer_d));

	// New API with stream context
	NPP_SAFE_CALL(nppiHistogramRange_32f_C1R_Ctx(data_d, line_step, roi, hist_d, bins_d, num_levels, buffer_d, nppStreamCtx));

	Npp32s* hist_h = (Npp32s*)malloc(num_bins * sizeof(Npp32s));
	CUDA_SAFE_CALL(cudaMemcpy(hist_h, hist_d, num_bins * sizeof(Npp32s), cudaMemcpyDeviceToHost));

    
    // Convert histogram counts to probabilities
    double* probabilities = (double*)malloc(num_bins*sizeof(double));
    for (int i = 0; i < num_bins; ++i)
    {
        probabilities[i] = (double)hist_h[i] / (double)(width*height*depth);
    }
    
    // Compute the entropy
    double entropy = 0;
    for (int i = 0; i < num_bins; ++i)
    {
        if (hist_h[i] != 0)
            entropy += -probabilities[i] * log(probabilities[i]);
    }
        
    cudaFree(buffer_d);
    cudaFree(data_min_d);
    cudaFree(data_max_d);
    cudaFree(bins_d);
    cudaFree(hist_d);
    free(bins_h);
    free(hist_h);
    
    CHECK_FOR_ERROR("end OpticalField::calcEntropy");
    return entropy;
}

void OpticalField::calcHistogram(Npp32f* bins_h, Npp32s* hist_h)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    assert(this->data.getDataType() == CV_32F);
    
    NppiSize roi = {width*height, depth};
    int num_bins = 256;
    int num_levels = num_bins+1;
    int line_step = width*height*sizeof(float);
    float* data_d = (float*)this->data.getCuData();
	
	// First, create or get the NppStreamContext
	//NppStreamContext nppStreamCtx = CreateNppStreamContext();
	NppStreamContext nppStreamCtx;
	NPP_SAFE_CALL(nppGetStreamContext(&nppStreamCtx));

	// First, find the required buffer size
	//int buffer_size = 0;
	//int alt_buffer_size = 0;
	size_t buffer_size = 0;
	size_t alt_buffer_size = 0;


	// OLD: NPP_SAFE_CALL(nppiHistogramRangeGetBufferSize_32f_C1R(roi, num_levels, &buffer_size));
	NPP_SAFE_CALL(nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(roi, num_levels, &buffer_size, nppStreamCtx));

	// OLD: NPP_SAFE_CALL(nppiMinMaxGetBufferHostSize_32f_C1R(roi, &alt_buffer_size));
	NPP_SAFE_CALL(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(roi, &alt_buffer_size, nppStreamCtx));

	buffer_size = std::max(buffer_size, alt_buffer_size);

        
    // Allocate buffer
    Npp8u* buffer_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_d, buffer_size));
	
	// Find the min and max of the data
	Npp32f* data_min_d;
	Npp32f* data_max_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_min_d, 1 * sizeof(Npp32f)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_max_d, 1 * sizeof(Npp32f)));

	//NppStreamContext nppStreamCtx = CreateNppStreamContext();

	// Old API (pre-CUDA 12)
	// NPP_SAFE_CALL(nppiMinMax_32f_C1R(data_d, line_step, roi, data_min_d, data_max_d, buffer_d));

	// New API with stream context
	NPP_SAFE_CALL(nppiMinMax_32f_C1R_Ctx(data_d, line_step, roi, data_min_d, data_max_d, buffer_d, nppStreamCtx));

	Npp32f data_min, data_max;
	CUDA_SAFE_CALL(cudaMemcpy(&data_min, data_min_d, 1 * sizeof(Npp32f), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&data_max, data_max_d, 1 * sizeof(Npp32f), cudaMemcpyDeviceToHost));
    
    // Compute histogram bins
    bins_h = (Npp32f*)malloc(num_levels*sizeof(Npp32f));
    Npp32f bin_size = (data_max - data_min) / num_bins;
    bins_h[0] = data_min;
    for (int i = 1; i < num_levels; ++i)
    {
        bins_h[i] = bins_h[i-1] + bin_size;
    }
    Npp32f* bins_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&bins_d, num_levels*sizeof(Npp32f)));
    CUDA_SAFE_CALL(cudaMemcpy(bins_d, bins_h, num_levels*sizeof(Npp32f), cudaMemcpyHostToDevice));
	
	// Fill Histogram
	Npp32s* hist_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&hist_d, num_bins * sizeof(Npp32s)));
	CUDA_SAFE_CALL(cudaMemset(hist_d, 0, num_bins * sizeof(Npp32s)));

	// Create NppStreamContext using helper function
	//NppStreamContext nppStreamCtx = CreateNppStreamContext();

	// Old API (pre-CUDA 12)
	// NPP_SAFE_CALL(nppiHistogramRange_32f_C1R(data_d, line_step, roi, hist_d, bins_d, num_levels, buffer_d));

	// New API with stream context
	NPP_SAFE_CALL(nppiHistogramRange_32f_C1R_Ctx(data_d, line_step, roi, hist_d, bins_d, num_levels, buffer_d, nppStreamCtx));

	hist_h = (Npp32s*)malloc(num_bins * sizeof(Npp32s));
	CUDA_SAFE_CALL(cudaMemcpy(hist_h, hist_d, num_bins * sizeof(Npp32s), cudaMemcpyDeviceToHost));

    
    cudaFree(buffer_d);
    cudaFree(data_min_d);
    cudaFree(data_max_d);
    cudaFree(bins_d);
    cudaFree(hist_d);
}

void updateView(int zidx, void* ptr)
{
    OpticalField* field = (OpticalField*)ptr;
    int width = field->getWidth();
    int height = field->getHeight();
    double scale_min, scale_max;
    field->getScaleMinMax(&scale_min, &scale_max);
    
    if (field->isState(OPTICALFIELD_STATE_RECONSTRUCTED))
    {
        CuMat single_plane = field->getPlane(zidx);
        cv::Mat plane = single_plane.getMatData();
        if (scale_max == 0.0)
            cv::normalize(plane, plane, 0, 1, cv::NORM_MINMAX);
        else
            plane = (plane - scale_min) / (scale_max - scale_min);
        cv::imshow("Reconstruction", plane);
    }
    else
    {
        std::cout << "Error: OpticalField::view(): Invalid state" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
}

void OpticalField::view()
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    cv::namedWindow("Reconstruction", cv::WINDOW_AUTOSIZE);
    
    int plane_id = 0;
    cv::createTrackbar("Plane #", "Reconstruction",
        &plane_id, params.num_planes-1, updateView, this);
    
    updateView(0, this);
    
    /*if (this->state == OPTICALFIELD_STATE_RECONSTRUCTED)
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32F));
        size_t plane_size = width * height;

        cv::Mat plane;

        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            std::cout << "Displaying plane " << zidx << std::endl;

            void* plane_start = (float*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32F);
            plane = single_plane.getMatData();
            //plane.convertTo(plane, CV_8U);

            cv::imshow("Reconstruction", plane);
            cv::waitKey(0);
        }
    }
    else if (this->state == OPTICALFIELD_STATE_DECONVOLVED)
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        size_t plane_size = width * height;

        cv::Mat plane;

        std::cout << "Displaying plane 0000";
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            printf("\b\b\b\b%04d", zidx);

            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getReal().getMatData();

            cv::imshow("Reconstruction", plane);
            cv::waitKey(0);
        }
        std::cout << std::endl;
    }
    else if (this->state == OPTICALFIELD_STATE_FULL_COMPLEX)
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        size_t plane_size = width * height;

        cv::Mat plane;
        
        double global_min = 0;
        double global_max = 0;
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            
            double single_min = 0;
            double single_max = 0;
            minMaxLoc(plane, &single_min, &single_max);
            
            if (zidx == 0) {global_min = single_min; global_max = single_max;}
            global_min = (single_min < global_min)? single_min : global_min;
            global_max = (single_max > global_max)? single_max : global_max;
        }

        std::cout << "Displaying plane 0000";
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            printf("\b\b\b\b%04d", zidx);

            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            
            plane = (plane - global_min) / (global_max - global_min);

            cv::imshow("Reconstruction", plane);
            cv::waitKey(0);
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Error: OpticalField::view(): Invalid state" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    cv::destroyAllWindows();*/
    
    while (true)
    {
        char key = (char)cv::waitKey(1);
        if (key == 27) // 'esc' key was pressed
        {
            cv::destroyAllWindows();
            break;
        }
    }
}

void OpticalField::save()
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    this->save("");
}

void OpticalField::save(char* prefix)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    std::cout << "Saving OpticalField data" << std::endl;
    
    switch (this->state)
    {
    case OPTICALFIELD_STATE_RECONSTRUCTED:
    case OPTICALFIELD_STATE_DECONVOLVED_REAL:
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32F));
        size_t plane_size = width * height;

        cv::Mat plane;

        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32F);
            plane = single_plane.getMatData();
            plane.convertTo(plane, CV_8U, 255);

            char filename[FILENAME_MAX];
            sprintf(filename, "%s%splane_%04d.tif", 
                params.output_path, prefix, zidx);
            bool result = cv::imwrite(filename, plane);
            if (!result)
            {
                std::cerr << "OpticalField::save: imwrite failed" << std::endl;
                throw HOLO_ERROR_UNKNOWN_ERROR;
            }
        }
        break;
    }
    case OPTICALFIELD_STATE_DECONVOLVED:
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        size_t plane_size = width * height;

        cv::Mat plane;

        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getReal().getMatData();
            plane.convertTo(plane, CV_8U, 255);

            char filename[FILENAME_MAX];
            sprintf(filename, "%s%splane_%04d.tif", 
                params.output_path, prefix, zidx);
            bool result = cv::imwrite(filename, plane);
            if (!result)
            {
                std::cerr << "OpticalField::save: imwrite failed" << std::endl;
                throw HOLO_ERROR_UNKNOWN_ERROR;
            }
        }
        std::cout << std::endl;
        break;
    }
    case OPTICALFIELD_STATE_FULL_COMPLEX:
    case OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX:
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        size_t plane_size = width * height;

        cv::Mat plane;
        cv::Mat phase;
        
        double global_min = 0;
        double global_max = 0;
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            
            double single_min = 0;
            double single_max = 0;
            minMaxLoc(plane, &single_min, &single_max);
            
            if (zidx == 0) {global_min = single_min; global_max = single_max;}
            global_min = (single_min < global_min)? single_min : global_min;
            global_max = (single_max > global_max)? single_max : global_max;
        }
        
        //CuMat temp_writer;

        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            phase = single_plane.getPhase().getMatData();
            
            //temp_writer.setMatData(phase);
            //char plane_filename[FILENAME_MAX];
            //sprintf(plane_filename, "%s/phase_z%04d.txt", params.output_path, zidx);
            //temp_writer.save(plane_filename, FILEMODE_ASCII);
            
            plane = (plane - global_min) / (global_max - global_min);
            plane.convertTo(plane, CV_8U, 255);
            //phase = (phase - M_PI) / (2*M_PI);
            //phase = (phase) / (2*M_PI);
            phase = abs(phase - M_PI) / (M_PI);
            phase.convertTo(phase, CV_8U, 255);

            char filename[FILENAME_MAX];
            sprintf(filename, "%s%splane_%04d.tif",
                params.output_path, prefix, zidx);
            bool result = cv::imwrite(filename, plane);
            if (!result)
            {
                std::cerr << "OpticalField::save: imwrite failed" << std::endl;
                throw HOLO_ERROR_UNKNOWN_ERROR;
            }
            
            sprintf(filename, "%s%sphase_plane_%04d.tif",
                params.output_path, prefix, zidx);
            result = cv::imwrite(filename, phase);
            if (!result)
            {
                std::cerr << "OpticalField::save: imwrite phase failed" << std::endl;
                throw HOLO_ERROR_UNKNOWN_ERROR;
            }
        }
        break;
    }
    default:
    {
        std::cout << "Error: OpticalField::save(): Invalid state" << std::endl;
        std::cout << "State was " << this->state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    }
    std::cout << "wrote OpticalField to <" << params.output_path 
              << prefix << "plane_%04d.tif>" << std::endl;
}

void OpticalField::saveSparse(char* prefix)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    assert(this->data.getElemSize() == 4);
    
    size_t numel = width * height * depth;
    
    size_t plane_size = width * height;
    float* plane_h = (float*)malloc(plane_size * sizeof(float));
    
    char filename[FILENAME_MAX];
    sprintf(filename, "%s/%s.csv", params.output_path, prefix);
    FILE* fid = NULL;
    fid = fopen(filename, "w");
    if (fid == NULL)
    {
        std::cout << "Unable to open file: " << filename << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }
    fprintf(fid, "x, y, z, value\n");
    
    size_t num_lines = 0;
    size_t nnz = this->countNonZeros();
    
    for (int zid = 0; zid < depth; ++zid)
    {
        // Copy plane to host for processing
        float* plane_d = (float*)data.getCuData() + zid * plane_size;
        cudaMemcpy(plane_h, plane_d, plane_size*sizeof(float), cudaMemcpyDeviceToHost);
        
        size_t start_idx = zid * plane_size;
        for (size_t xid = 0; xid < width; ++xid)
        {
            for (size_t yid = 0; yid < height; ++yid)
            {
                size_t pidx = yid*width + xid;
                float value = plane_h[pidx];
                if (value != 0)
                {
                    fprintf(fid, "%d, %d, %d, %f\n", xid, yid, zid, value);
                    num_lines++;
                }
                if (num_lines > nnz) throw HOLO_ERROR_CRITICAL_ASSUMPTION;
            }
        }
    }
    
    fclose(fid);
    free(plane_h);
    CHECK_FOR_ERROR("end OpticalField::saveSparse");
}

void OpticalField::saveProjections(char* prefix, CmbMethod method)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    DECLARE_TIMING(saveProjections_compute);
    START_TIMING(saveProjections_compute);
    
    // Initialize the cmb images
    cv::Mat xycmb = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat xzcmb = cv::Mat::zeros(depth,  width, CV_32F);
    cv::Mat yzcmb = cv::Mat::zeros(height, depth, CV_32F);
    if (method == MIN_CMB)
    {
        xycmb = 1e6 * cv::Mat::ones(height, width, CV_32F);
    }
    
    cv::Mat xzvec = cv::Mat::zeros(1, width, CV_32F);
    cv::Mat yzvec = cv::Mat::zeros(height, 1, CV_32F);
    
    switch (this->state)
    {
    case OPTICALFIELD_STATE_FULL_COMPLEX:
    case OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX:
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        size_t plane_size = width * height;

        cv::Mat plane;
        cv::Mat phase;
        cv::Mat phaseA;
        cv::Mat phaseB;
        
        // Compute global min and max for normalization
        // TODO: This should be a separate method
        double global_min = 0;
        double global_max = 0;
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            
            double single_min = 0;
            double single_max = 0;
            minMaxLoc(plane, &single_min, &single_max);
            
            if (zidx == 0) {global_min = single_min; global_max = single_max;}
            global_min = (single_min < global_min)? single_min : global_min;
            global_max = (single_max > global_max)? single_max : global_max;
        }
        
        // Compute the combined images
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float2*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32FC2);
            plane = single_plane.getMag().getMatData();
            phase = single_plane.getPhase().getMatData();
            
            plane = (plane - global_min) / (global_max - global_min);
            
            cv::threshold(phase, phaseA, M_PI, 0, cv::THRESH_TOZERO_INV);
            cv::threshold(phase-2*M_PI, phaseB, -M_PI, 0, cv::THRESH_TOZERO);
            phase = phaseA + phaseB; // Convert to [-pi,pi] (matlab style)
            phase = abs(phase) / M_PI;
            
            if (method == MAX_CMB)
            {
                cv::reduce(plane, xzvec, 0, cv::REDUCE_MAX);
                cv::reduce(plane, yzvec, 1, cv::REDUCE_MAX);
                cv::max(plane, xycmb, xycmb);
            }
            else if (method == MIN_CMB)
            {
                cv::reduce(plane, xzvec, 0, cv::REDUCE_MIN);
                cv::reduce(plane, yzvec, 1, cv::REDUCE_MIN);
                cv::min(plane, xycmb, xycmb);
            }
            else if (method == PHASE_CMB)
            {
                cv::reduce(phase, xzvec, 0, cv::REDUCE_MAX);
                cv::reduce(phase, yzvec, 1, cv::REDUCE_MAX);
                cv::max(phase, xycmb, xycmb);
            }
            else throw HOLO_ERROR_INVALID_ARGUMENT;
            
            xzvec.copyTo(xzcmb(cv::Rect(0, zidx, width, 1)));
            yzvec.copyTo(yzcmb(cv::Rect(zidx, 0, 1, height)));
        }
        
        xycmb.convertTo(xycmb, CV_8U, 255);
        xzcmb.convertTo(xzcmb, CV_8U, 255);
        yzcmb.convertTo(yzcmb, CV_8U, 255);
        
        break;
    }
    case OPTICALFIELD_STATE_RECONSTRUCTED:
    case OPTICALFIELD_STATE_DECONVOLVED_REAL:
    {
        CuMat single_plane;
        single_plane.setMatData(cv::Mat::zeros(height, width, CV_32F));
        size_t plane_size = width * height;

        cv::Mat plane;

        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            void* plane_start = (float*)data.getCuData() + zidx*plane_size;
            single_plane.setCuData(plane_start, height, width, CV_32F);
            plane = single_plane.getMatData();
            
            if (method == MAX_CMB)
            {
                cv::reduce(plane, xzvec, 0, cv::REDUCE_MAX);
                cv::reduce(plane, yzvec, 1, cv::REDUCE_MAX);
                cv::max(plane, xycmb, xycmb);
            }
            else if (method == MIN_CMB)
            {
                cv::reduce(plane, xzvec, 0, cv::REDUCE_MIN);
                cv::reduce(plane, yzvec, 1, cv::REDUCE_MIN);
                cv::min(plane, xycmb, xycmb);
            }
            else throw HOLO_ERROR_INVALID_ARGUMENT;
            
            xzvec.copyTo(xzcmb(cv::Rect(0, zidx, width, 1)));
            yzvec.copyTo(yzcmb(cv::Rect(zidx, 0, 1, height)));
        }
        
        xycmb.convertTo(xycmb, CV_8U, 255);
        xzcmb.convertTo(xzcmb, CV_8U, 255);
        yzcmb.convertTo(yzcmb, CV_8U, 255);
        
        break;
    }
    case OPTICALFIELD_STATE_DECONVOLVED:
    default:
    {
        std::cout << "Error: OpticalField::saveProjections: Invalid state" << std::endl;
        std::cout << "State was " << this->state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    }
    
    SAVE_TIMING(saveProjections_compute);
    DECLARE_TIMING(saveProjections_write);
    START_TIMING(saveProjections_write);
    
    // Save the images
    char filename[FILENAME_MAX];
    sprintf(filename, "%s%s_xy.tif", params.output_path, prefix);
    bool result1 = cv::imwrite(filename, xycmb);
    sprintf(filename, "%s%s_xz.tif", params.output_path, prefix);
    bool result2 = cv::imwrite(filename, xzcmb);
    sprintf(filename, "%s%s_yz.tif", params.output_path, prefix);
    bool result3 = cv::imwrite(filename, yzcmb);
    if (!result1 || !result2 || !result3)
    {
        std::cerr << "OpticalField::saveProjections: imwrite failed" << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    
    SAVE_TIMING(saveProjections_write);
    return;
}

__global__ void renormalizeUndoDeconvNorm_kernel(float* data_d, float new_mean, int n)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int idx = tid + bid*blockDim.x*blockDim.y;

    data_d[idx] = (data_d[idx] + 1) * new_mean;
}

void OpticalField::renormalize(OpticalFieldScale new_scale)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->scale != SCALE_DECONV_NORM) throw HOLO_ERROR_INVALID_STATE;
    if (new_scale != SCALE_0_255) throw HOLO_ERROR_INVALID_ARGUMENT;

    int tile_size = 16;
    dim3 gridDim(width/tile_size, height/tile_size, depth);
    dim3 blockDim(tile_size, tile_size, 1);

    renormalizeUndoDeconvNorm_kernel<<<gridDim, blockDim>>>
        ((float*)this->data.getCuData(), this->hologram_original_mean, width*height*depth);
}

__global__ void scale_0_255_kernel(float* data_d, size_t size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int idx = tid + bid*blockDim.x*blockDim.y;

    if (idx < size)
    {
        data_d[idx] = data_d[idx] * 255.0;
    }
}

__global__ void scale_0_1_kernel(float* data_d, size_t size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int idx = tid + bid*blockDim.x*blockDim.y;

    if (idx < size)
    {
        data_d[idx] = data_d[idx] / 255.0;
    }
}


void OpticalField::scaleImage(OpticalFieldScale new_scale)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    int tile_size = 16;
    dim3 gridDim(width/tile_size, height/tile_size, depth);
    dim3 blockDim(tile_size, tile_size, 1);
    size_t numel = width*height*depth;
    
    if (this->data.getDataType() != CV_32F)
    {
        std::cout << "OpticalField::scaleImage Error: Type must be CV_32F" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    if (new_scale == SCALE_0_255)
    {
        if (this->scale != SCALE_0_1)
        {
            std::cout << "OpticalField::scaleImage Error: invalid state" << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
        
        scale_0_255_kernel<<<gridDim, blockDim>>>((float*)this->data.getCuData(), numel);
        scale = SCALE_0_255;
    }
    else if (new_scale == SCALE_0_1)
    {
        if (this->scale != SCALE_0_255)
        {
            std::cout << "OpticalField::scaleImage Error: invalid state" << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
        
        scale_0_1_kernel<<<gridDim, blockDim>>>((float*)this->data.getCuData(), numel);
        scale = SCALE_0_1;
    }
    else
    {
        std::cout << "OpticalField::scaleImage Error: Unkown new scale" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
}

__global__ void round_kernel(float* data_d, size_t size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int idx = tid + bid*blockDim.x*blockDim.y;

    if (idx < size)
    {
        data_d[idx] = round(data_d[idx]);
    }
}

void OpticalField::round()
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->scale != SCALE_0_255)
    {
        std::cout << "OpticalField::round: Scale must be SCALE_0_255" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    if (this->data.getDataType() != CV_32F)
    {
        std::cout << "OpticalField::round Error: Type must be CV_32F" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    int tile_size = 16;
    dim3 gridDim(width/tile_size, height/tile_size, depth);
    dim3 blockDim(tile_size, tile_size, 1);
    size_t numel = width*height*depth;
    
    round_kernel<<<gridDim, blockDim>>>((float*)this->data.getCuData(), numel);
}

__global__ void ap2ri_kernel(float2* data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float amp = data[idx].x;
    float phase = data[idx].y;
    
    data[idx].x = amp * cos(phase);
    data[idx].y = amp * sin(phase);
}

__global__ void ri2ap_kernel(float2* data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float re = data[idx].x;
    float im = data[idx].y;
    
    data[idx].x = sqrt(re*re + im*im);
    data[idx].y = atan2(im, re);
}

void OpticalField::convertComplexRepresentationTo(ComplexDataFormat rep)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (this->complex_representation == rep) return;
    
    size_t numel = width*height*depth;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    float2* data_d = (float2*)this->data.getCuData();
    
    switch (rep)
    {
    case REAL_IMAGINARY:
    {
        if (this->complex_representation != AMPLITUDE_PHASE)
        {
            std::cout << "OpticalField::convertComplexRepresentation: "
                << "Error: Unknown current representation" << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
        ap2ri_kernel<<<dim_grid, dim_block>>>(data_d, numel);
        break;
    }
    case AMPLITUDE_PHASE:
    {
        if (this->complex_representation != REAL_IMAGINARY)
        {
            std::cout << "OpticalField::convertComplexRepresentation: "
                << "Error: Unknown current representation" << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
        ri2ap_kernel<<<dim_grid, dim_block>>>(data_d, numel);
        break;
    }
    default:
    {
        std::cout << "OpticalField::convertComplexRepresentation: Unknown format" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    this->data.setCuData((void*)data_d);
    complex_representation = rep;
    CHECK_FOR_ERROR("end OpticalField::convertComplexRepresentation");
}

__global__ void fillField_kernel(float2* data, size_t* indices, size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        data[indices[idx]].x = 1.0;
        data[indices[idx]].y = 0.0;
    }
}

void OpticalField::fillField(ObjectCloud cloud)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    CHECK_FOR_ERROR("Begin OpticalField::fillField");
    
    // Count total number of voxels
    size_t num_voxels = 0;
    size_t num_objects = cloud.getNumObjects();
    for (int i = 0; i < num_objects; ++i)
    {
        num_voxels += cloud.getObject(i).getNumVoxels();
    }
    
    // Create array of voxel indices
    size_t* voxels = (size_t*)malloc(num_voxels*sizeof(size_t));
    size_t vid = 0;
    for (int i = 0; i < num_objects; ++i)
    {
        Blob3d obj = cloud.getObject(i);
        for (int n = 0; n < obj.getNumVoxels(); ++n)
        {
            cv::Point3i vox = obj.getVoxel(n);
            voxels[vid] = vox.z*width*height + vox.y*width + vox.x;
            vid++;
        }
    }
    
    size_t* voxels_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&voxels_d, num_voxels * sizeof(size_t)));
    CUDA_SAFE_CALL(cudaMemcpy(voxels_d, voxels, num_voxels*sizeof(size_t), cudaMemcpyHostToDevice));
    
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)num_voxels / (float)dim_block);
    float2* data_d = (float2*)this->data.getCuData();
    cudaMemset(data_d, 0, data.getDataSize());
    fillField_kernel<<<dim_grid, dim_block>>>(data_d, voxels_d, num_voxels);
    
    this->data.setCuData(data_d);
    
    cudaFree(voxels_d);
    CHECK_FOR_ERROR("End OpticalField::fillField");
    return;
}

//============================= ACCESS     ===================================

CuMat OpticalField::getPlane(size_t plane_idx)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    if (!data.isStoredCu()) throw HOLO_ERROR_INVALID_STATE;

    CuMat plane;
    
    if ((plane_idx >= depth) || (plane_idx < 0))
    {
        std::cout << "OpticalField::getPlane Error: plane out of bounds" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (data.getDataType() == CV_32F)
        plane.setCuData((float*)data.getCuData() + plane_idx*width*height, height, width, CV_32F);
    else if (data.getDataType() == CV_32FC2)
        plane.setCuData((float2*)data.getCuData() + plane_idx*width*height, height, width, CV_32FC2);
    else
    {
        std::cout << "OpticalField::getPlane Error: invalid type" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    return plane;
}

CuMat OpticalField::getRoi(int x, int y, int z, int width, int height, int depth)
{
    assert(this->state != OPTICALFIELD_STATE_UNALLOCATED);
    
    
    CHECK_FOR_ERROR("begin OpticalField::getRoi");
    
    CuMat roi;
    int data_type = data.getDataType();
    int elem_size = data.getElemSize();
    
    roi.allocateCuData(width, height, depth, elem_size);
    uint8_t* roi_data_d = (uint8_t*)roi.getCuData();
    uint8_t* source_data_d = (uint8_t*)data.getCuData();
    
    // Copy data over row by row
    int src_width = data.getWidth();
    int src_height = data.getHeight();
    int start_idx_src = z*src_width*src_height + y*src_width + x;
    size_t row_size = width*elem_size;
    for (int zid = 0; zid < depth; ++zid)
    {
        for (int yid = 0; yid < height; ++yid)
        {
            int row_start_src = start_idx_src + zid*src_width*src_height + yid*src_width;
            int row_start_dst = zid*width*height + yid*width;
            uint8_t* src_start = source_data_d + row_start_src*elem_size;
            uint8_t* dst_start = roi_data_d + row_start_dst*elem_size;
            cudaMemcpy(dst_start, src_start, row_size, cudaMemcpyDeviceToDevice);
        }
    }
    
    CHECK_FOR_ERROR("OpticalField::getRoi");
    return roi;
}

void OpticalField::setData(CuMat data)
{
    CHECK_FOR_ERROR("begin OpticalField::setData");
    this->data = data;
    this->width = data.getWidth();
    this->height = data.getHeight();
    this->depth = data.getDepth();
    CHECK_FOR_ERROR("end OpticalField::setData");
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
