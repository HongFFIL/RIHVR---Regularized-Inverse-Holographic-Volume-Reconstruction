#include "deconvolution.h"

#include <math.h>

using namespace umnholo;

Deconvolution::Deconvolution(Hologram holo) : psf(holo),
                                              Reconstruction(holo)
{
    iterative_mode = false;
    is_created_fft_plan_3d = false;
    is_created_psf = false;
}

__global__ void deconv_div_kernel(float2* num, float2* denom, float beta, int nX, int nY, int nZ)
{
    // Matlab expression is num = num./(denom + Beta);

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    int idx = z*nX*nY + y*nX + x;

    float2 t_num = num[idx];
    float2 t_denom = denom[idx];
    t_denom.x += beta;

    float mag = t_denom.x*t_denom.x + t_denom.y*t_denom.y;

    float scale = nX*nY*nZ; // Scale for IFFT

    num[idx].x = ((t_num.x*t_denom.x + t_num.y*t_denom.y) / mag) / scale;
    num[idx].y = ((t_num.y*t_denom.x - t_num.x*t_denom.y) / mag) / scale;
}

__global__ void zero_kernel(float2* data_d, int num_elements)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < num_elements)
    {
        data_d[idx].x = 0.0;
        data_d[idx].y = 0.0;
    }
}

__global__ void normalize_kernel(float2* data_d, int size, float2 min_h, float2 max_h)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        //FP scale = 255.0 / (max_h - min_h);
        float scale = 1.0 / (max_h.x - min_h.x);
        data_d[idx].x = (data_d[idx].x - min_h.x) * scale;
        data_d[idx].y = (data_d[idx].y - min_h.y) * scale;

        // Rescale from 0-1
        data_d[idx].x = (1 - abs(data_d[idx].x));
        data_d[idx].y = (1 - abs(data_d[idx].y));
    }
}

void Deconvolution::destroy()
{
    data.destroy();
    if (is_created_psf)
    {
        psf.getData().destroy();
        is_created_psf = false;
    }
    if (is_created_fft_plan_3d)
    {
        cufftDestroy(fft_plan_3d);
        is_created_fft_plan_3d = false;
    }
    return;
}

template <unsigned int blockSize>
__global__ void minReduceReal_kernel(float2* data_d, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    // Treat imaginary component of data as buffer
    
    // Load into shared data
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockSize*2) + tid;
    int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 100000;
    
    while (i < size)
    {
        sdata[tid] = min(min(data_d[i].x, data_d[i + blockSize].x), sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32)
    {
        if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]); __syncthreads();
        if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]); __syncthreads();
        if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]); __syncthreads();
        if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]); __syncthreads();
        if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]); __syncthreads();
        if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]); __syncthreads();
    }
    if (tid == 0) data_d[blockIdx.x].y = sdata[0];
}

template <unsigned int blockSize>
__global__ void maxReduceReal_kernel(float2* data_d, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    // Treat imaginary component of data as buffer
    
    // Load into shared data
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockSize*2) + tid;
    int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = -100000;
    
    while (i < size)
    {
        sdata[tid] = max(max(data_d[i].x, data_d[i + blockSize].x), sdata[tid]);
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32)
    {
        if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]); __syncthreads();
        if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]); __syncthreads();
        if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]); __syncthreads();
        if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]); __syncthreads();
        if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]); __syncthreads();
        if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]); __syncthreads();
    }
    if (tid == 0) data_d[blockIdx.x].y = sdata[0];
}

void updateMinMax(float2* data_d, int size, float2& min_h, float2& max_h)
{
    float2 old_min = min_h;
    float2 old_max = max_h;
    
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid = (ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512*sizeof(float);
    CHECK_FOR_ERROR("before minReduceReal_kernel");
    minReduceReal_kernel<512><<<dimGrid, dimBlock, smemSize>>>(data_d, size);
    CHECK_FOR_ERROR("after minReduceReal_kernel");
    float2 new_min;
    CUDA_SAFE_CALL( cudaMemcpy(&new_min, data_d, 1*sizeof(float2), cudaMemcpyDeviceToHost) );
    
    min_h.x = std::min(new_min.y, old_min.x);
    min_h.y = 0;
    
    maxReduceReal_kernel<512><<<dimGrid, dimBlock, smemSize>>>(data_d, size);
    float2 new_max;
    CUDA_SAFE_CALL( cudaMemcpy(&new_max, data_d, 1*sizeof(float2), cudaMemcpyDeviceToHost) );
    
    max_h.x = std::max(new_max.y, old_max.x);
    max_h.y = 0;    

    CHECK_FOR_ERROR("Deconvolution::updataMinMax");
    return;
}

void normalizeVolume(float2* data_d, int size, float2 min_h, float2 max_h)
{
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(size/1024, 1, 1);
    normalize_kernel<<<dimGrid, dimBlock>>>(data_d, size, min_h, max_h);

    CHECK_FOR_ERROR("Deconvolution::normalize_kernel");
    return;
}

void deconv_rescale(float2* data_d, int nX, int nY, int nZ)
{
    CHECK_FOR_ERROR("before Deconvolution::deconv_rescale");
    
    int two_planes = 2 * nX * nY;
    zero_kernel<<<two_planes/1024, 1024>>>(data_d, two_planes);

    float2 volume_min_h = { 1000.0, 1000.0 };
    float2 volume_max_h = { -1000.0, -1000.0 };
    updateMinMax(data_d, nX*nY*nZ, volume_min_h, volume_max_h);
    normalizeVolume(data_d, nX*nY*nZ, volume_min_h, volume_max_h);
    
    CHECK_FOR_ERROR("Deconvolution::deconv_rescale");
}

void Deconvolution::deconvolve(Hologram holo)
{
    CHECK_FOR_ERROR("before OpticalField::deconvolve");
    if (this->state != OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX)
        throw HOLO_ERROR_INVALID_STATE;

    bool was_created_psf = is_created_psf;
    if (!is_created_psf)
    {
        DECLARE_TIMING(DD_PSF_RECONSTRUCTION);
        START_TIMING(DD_PSF_RECONSTRUCTION);
        //Deconvolution psf(holo);
        psf.reconstruct(holo, RECON_MODE_FSP);
        is_created_psf = true;
        //CHECK_FOR_ERROR("after reconstruction");
        SAVE_TIMING(DD_PSF_RECONSTRUCTION);
    }

    DECLARE_TIMING(DD_CONJUGATE_ALLOCATE);
    START_TIMING(DD_CONJUGATE_ALLOCATE);
    // Multiply reconstructed volumes by complex conjugate
    this->multiply_conjugate();
    if (!was_created_psf) psf.multiply_conjugate();
    //CHECK_FOR_ERROR("after multiply_conjugate");

    this->data.allocateCuData(width, height, depth, sizeof(float2));
    float2* img_d = (float2*)this->data.getCuData();
    float2* psf_d = (float2*)psf.getData().getCuData();
    SAVE_TIMING(DD_CONJUGATE_ALLOCATE);

    DECLARE_TIMING(DD_FFT_CREATION_FORWARD);
    START_TIMING(DD_FFT_CREATION_FORWARD);
    // Create and run the 3D FFT
    if (!is_created_fft_plan_3d)
    {
        //cufftHandle fft_plan_3d;
        cufftPlan3d(&fft_plan_3d, depth, height, width, CUFFT_C2C);
        is_created_fft_plan_3d = true;
    }
    CUFFT_SAFE_CALL( cufftExecC2C(fft_plan_3d, img_d, img_d, CUFFT_FORWARD) );
    if (!was_created_psf) CUFFT_SAFE_CALL( cufftExecC2C(fft_plan_3d, psf_d, psf_d, CUFFT_FORWARD) );
    //CHECK_FOR_ERROR("after 3d fft");
    SAVE_TIMING(DD_FFT_CREATION_FORWARD);

    DECLARE_TIMING(DD_DECONV_DIV);
    START_TIMING(DD_DECONV_DIV);
    // Perform the deconvolution math
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y, depth/dimBlock.z);
    deconv_div_kernel<<<dimGrid, dimBlock>>>(img_d, psf_d, params.beta, width, height, depth);
    //CHECK_FOR_ERROR("after deconv_div_kernel");
    SAVE_TIMING(DD_DECONV_DIV);

    DECLARE_TIMING(DD_IFFT);
    START_TIMING(DD_IFFT);
    // Run inverse FFT
    CUFFT_SAFE_CALL( cufftExecC2C(fft_plan_3d, img_d, img_d, CUFFT_INVERSE) );
    SAVE_TIMING(DD_IFFT);

    //DECLARE_TIMING(DD_CLEANUP);
    //START_TIMING(DD_CLEANUP);
    //psf.data.destroy();
    //SAVE_TIMING(DD_CLEANUP);

    DECLARE_TIMING(DD_RESCALE);
    START_TIMING(DD_RESCALE);
    deconv_rescale(img_d, width, height, depth);
    this->scale = SCALE_0_1;

    this->state = OPTICALFIELD_STATE_DECONVOLVED;
    SAVE_TIMING(DD_RESCALE);

    DECLARE_TIMING(DD_MAKE_REAL);
    START_TIMING(DD_MAKE_REAL);
    this->data.makeReal();
    //renormalize(SCALE_0_255);
    SAVE_TIMING(DD_MAKE_REAL);

    DECLARE_TIMING(DD_ELIMINATE_CAUSTICS);
    START_TIMING(DD_ELIMINATE_CAUSTICS);
    this->eliminateCaustics();
    SAVE_TIMING(DD_ELIMINATE_CAUSTICS);
    
    DECLARE_TIMING(DD_CLEANUP);
    START_TIMING(DD_CLEANUP);
    this->state = OPTICALFIELD_STATE_DECONVOLVED_REAL;

    //psf.destroy();
    //cufftDestroy(fft_plan_3d);
    CHECK_FOR_ERROR("OpticalField::deconvolve");
    SAVE_TIMING(DD_CLEANUP);
    return;
}

template <unsigned int block_size>
__global__ void combinedXY_kernel
        (float* cmb_d, float* data_d, size_t width, size_t height, size_t depth)
{
    // Thread x and z are switched because of limits on allowable 
    // number of threads in z direction
    int x = threadIdx.z + blockIdx.z * blockDim.z;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.x + blockIdx.x * blockDim.x;
    int gidx = x + y * width + z * width * height;
    int widx = x + y * width;
    int tid = z;
    
    extern __shared__ float sdata[];
    sdata[tid] = 10000;
    
    int i = gidx;
    int block_step = block_size * width * height;
    while (i < width*height*depth)
    {
        sdata[tid] = min(data_d[i], sdata[tid]);
        i += block_step;
    }
    __syncthreads();
    
    if (block_size >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (block_size >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (block_size >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32)
    {
        if (block_size >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]); __syncthreads();
        if (block_size >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]); __syncthreads();
        if (block_size >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]); __syncthreads();
        if (block_size >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]); __syncthreads();
        if (block_size >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]); __syncthreads();
        if (block_size >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]); __syncthreads();
    }
    if (tid == 0) cmb_d[widx] = sdata[0];
}

CuMat Deconvolution::combinedXY()
{
    CHECK_FOR_ERROR("begin Deconvolution::combinedXY");
    if (this->state != OPTICALFIELD_STATE_DECONVOLVED_REAL)
        throw HOLO_ERROR_INVALID_STATE;

    /*
    cv::Mat result(height, width, CV_32F, 255);

    CuMat single_plane;
    single_plane.allocateCuData(width, height, 1, sizeof(float));
    size_t plane_size = width * height;

    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        int a = zidx;
        single_plane = this->getPlane(zidx);

        cv::min(result, single_plane.getMatData(), result);
    }

    if (this->scale == SCALE_0_255)
    {
        printf("Deconvolution::combinedXY: scale is SCALE_0_255, converting\n");
        result.convertTo(result, CV_8U);
    }
    CuMat cumat_result;
    cumat_result.setMatData(result);
    //*/
    
    cv::Mat result(height, width, CV_32F, 255);
    CuMat cumat_result;
    cumat_result.setMatData(result);
    
    float* cmb_d = (float*)cumat_result.getCuData();
    float* volume_data_d = (float*)this->data.getCuData();
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(1, height, width);
    size_t shared_mem = 256 * sizeof(float);
    CHECK_FOR_ERROR("Deconvolution::combinedXY before kernel");
    combinedXY_kernel<256><<<dimGrid, dimBlock, shared_mem>>>
        (cmb_d, volume_data_d, width, height, depth);
    cumat_result.setCuData(cmb_d);

    CHECK_FOR_ERROR("Deconvolution::combinedXY");
    return cumat_result.getReal();
}

__global__ void meanStdScale_kernel
        (float* data, size_t size, float m1, float m2, float s1, float s2)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        data[idx] = m1 + (data[idx] - m2) * (s1 / s2);
    }
}

void Deconvolution::matchMeanStd()
{
    if (initial_volume_mean == -1000) return;
    
    float* data_d = (float*)this->data.getCuData();
    size_t size = width*height*depth;
    
    float m1 = initial_volume_mean;
    float m2 = current_volume_mean;
    float s1 = initial_volume_std;
    float s2 = current_volume_std;
    
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(size/1024, 1, 1);
    meanStdScale_kernel<<<dimGrid, dimBlock>>>(data_d, size, m1, m2, s1, s2);
}

void Deconvolution::setInitialVolumeMeanStd(float mean, float std)
{
    initial_volume_mean = mean;
    initial_volume_std = std;
}

void Deconvolution::getVolumeMeanStd(float* mean, float* std)
{
    //printf("Deconvolution::getVolumeMeanStd: mean = %f, std = %f\n",
    //    initial_volume_mean, initial_volume_std);
    *mean = initial_volume_mean;
    *std = initial_volume_std;
}

__global__ void valueReplace_kernel(float* data_d, float val, size_t size)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < size)
    {
        data_d[idx] = val;
    }
}

void Deconvolution::eliminateCaustics()
{
    CHECK_FOR_ERROR("before Deconvolution::eliminateCaustics");
    
    // Calculate mean (and standard deviation for later)
    float mean, std;
    this->calcMeanStd(mean, std);
    
    current_volume_mean = mean;
    current_volume_std = std;
    initial_volume_mean = mean;
    initial_volume_std = std;
    
    // Do the replacing
    float* plane_d = (float*)this->data.getCuData();
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    valueReplace_kernel<<<dimGrid, dimBlock>>>(plane_d, current_volume_mean, width*height*2);
    
    CHECK_FOR_ERROR("Deconvolution::eliminateCaustics");
}
