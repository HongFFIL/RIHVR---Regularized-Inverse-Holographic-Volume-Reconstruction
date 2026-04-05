#include "sparse_gradient.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

void SparseGradient::initialize(Hologram holo, cufftHandle* fft_plan_p)
{
    fft_plan = *fft_plan_p;
    params = holo.getParams();
    width = holo.getWidth();
    height = holo.getHeight();
    CUDA_SAFE_CALL(cudaMalloc(&holo_fft_d, width*height*sizeof(float2)));
}

void SparseGradient::destroy()
{
    cudaFree(holo_fft_d);
    exponent_data.destroy();
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

__global__ void 
sparse_rs_exponent_kernel2(double* out_d, int Nx, int Ny, 
               double lambda, double reso)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    // Corrected off-by-one error if rs_mult_kernel that matched matlab
    double fx = (double)(((x + Nx / 2) % Nx)   - ((Nx / 2) )) / (double)Nx;
    double fy = (double)(((y + Ny / 2) % Ny)   - ((Ny / 2) )) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double sqrt_input = 1 - f2*(lambda / reso)*(lambda / reso);
    sqrt_input = (sqrt_input < 0) ? 0 : sqrt_input;
    out_d[idx] = -2 * M_PI*sqrt(sqrt_input) / lambda;
}

void SparseGradient::reconstruct(Hologram holo, ReconstructionMode mode)
{
    DECLARE_TIMING(SP_GRAD_RECONSTRUCT);
    START_TIMING(SP_GRAD_RECONSTRUCT);
    this->holo = holo;
    params = holo.getParams();
    
    if (mode != RECON_MODE_COMPLEX_CONSTANT_SUM)
    {
        std::cout << "SparseGradient::reconstruct mode must be RECON_MODE_COMPLEX_CONSTANT_SUM" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    float2* holo_d = (float2*)holo.getData().getCuData();
    cudaMemcpy(holo_fft_d, holo_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    cufftExecC2C(fft_plan, holo_fft_d, holo_fft_d, CUFFT_FORWARD);
    
    // Pre-calculate some of the rs exponent data to speed up computation
    exponent_data.allocateCuData(width, height, 1, sizeof(double));
    double* exp_d = (double*)exponent_data.getCuData();
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    sparse_rs_exponent_kernel2<<<grid_dim, block_dim>>>
        (exp_d, width, height, params.wavelength, params.resolution);
    
    SAVE_TIMING(SP_GRAD_RECONSTRUCT);
    CHECK_FOR_ERROR("after SparseGradient::reconstruct");
    use_reconstruction = true;
}

__global__ void 
sparse_rs_phase_mult_kernel2(float2* plane_d, float2* fft_holo_d, int Nx, int Ny, 
               double* exp_d, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    double exponent = z*exp_d[idx];

    // keep local copies of Hx[idx].{x|y}
    //double lHz_x = cos(exponent);
    //double lHz_y = sin(exponent);
    float lHz_x = 0.0;
    float lHz_y = 0.0;
    sincosf(exponent, &lHz_y, &lHz_x);
    
    float in_x = fft_holo_d[idx].x;
    float in_y = fft_holo_d[idx].y;

    // Result is complex multiplication of hologram fft with Hz kernel
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x = (lHz_x * in_x - lHz_y * in_y) / (Nx * Ny);
    plane_d[idx].y = (lHz_y * in_x + lHz_x * in_y) / (Nx * Ny);
}

__global__ void sparse_phase_scale_kernel2
    (float2* out_d, float2* in_d, size_t Nx, size_t Ny, double lambda, double z, double scale)
{
    scale = 1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    
    double phase = 2*M_PI*z/lambda;
    
    //double p_x = cos(phase);
    //double p_y = sin(phase);
    float p_x = 0.0;
    float p_y = 0.0;
    sincosf(phase, &p_y, &p_x);
    
    float2 old_data = in_d[idx];
    
    out_d[idx].x = scale * (old_data.x*p_x - old_data.y*p_y);
    out_d[idx].y = scale * (old_data.y*p_x + old_data.x*p_y);
}

void SparseGradient::getPlane(CuMat* plane, size_t plane_idx)
{
    DECLARE_TIMING(SP_GRAD_GET_PLANE);
    START_TIMING(SP_GRAD_GET_PLANE);
    assert(use_reconstruction);
    if (!use_reconstruction) return;
    
    DECLARE_TIMING(SP_GRAD_ALLOCATE_ZERO);
    START_TIMING(SP_GRAD_ALLOCATE_ZERO);
    float2* plane_d = (float2*)plane->getCuData();
    //cudaMemset(plane_d, 0, width*height*sizeof(float2));
    double* exp_d = (double*)exponent_data.getCuData();
    SAVE_TIMING(SP_GRAD_ALLOCATE_ZERO);
    
    float z = params.plane_list[plane_idx];
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    DECLARE_TIMING(SP_GRAD_PHASE_MULT_KERNEL);
    START_TIMING(SP_GRAD_PHASE_MULT_KERNEL);
    sparse_rs_phase_mult_kernel2<<<grid_dim, block_dim>>>
        (plane_d, holo_fft_d, width, height, exp_d, z);
    SAVE_TIMING(SP_GRAD_PHASE_MULT_KERNEL);
    
    DECLARE_TIMING(SP_GRAD_INVFFT);
    START_TIMING(SP_GRAD_INVFFT);
    cufftExecC2C(fft_plan, plane_d, plane_d, CUFFT_INVERSE);
    SAVE_TIMING(SP_GRAD_INVFFT);
    
    DECLARE_TIMING(SP_GRAD_PHASE_SCALE_KERNEL);
    START_TIMING(SP_GRAD_PHASE_SCALE_KERNEL);
    double scale = 2 / std::sqrt(2 * params.num_planes);
    sparse_phase_scale_kernel2<<<grid_dim, block_dim>>>
        (plane_d, plane_d, width, height, holo.getParams().wavelength, z, scale);
    SAVE_TIMING(SP_GRAD_PHASE_SCALE_KERNEL);
    
    plane->setCuData(plane_d);
    SAVE_TIMING(SP_GRAD_GET_PLANE);
    CHECK_FOR_ERROR("end SparseGradient::getPlane");
    return;
}

//============================= ACCESS     ===================================
//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
