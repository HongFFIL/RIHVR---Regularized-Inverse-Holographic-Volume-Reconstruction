#include "reconstruction.h"  // class implemented

#define _USE_MATH_DEFINES
#include <math.h>

using namespace umnholo;

__global__ void 
rs_mult_kernel(float2* plane_d, float2* fft_holo_d, int Nx, int Ny, 
               double lambda, double reso, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    double fx = (double)(((x + 1 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
    double fy = (double)(((y + 1 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double sqrt_input = 1 - f2*(lambda / reso)*(lambda / reso);
    sqrt_input = (sqrt_input < 0) ? 0 : sqrt_input;
    double exponent = 2 * M_PI*z*sqrt(sqrt_input) / lambda;

    // keep local copies of Hx[idx].{x|y}
    double lHz_x = cos(exponent);
    double lHz_y = -sin(exponent);

    // Result is complex multiplication of hologram fft with Hz kernel
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x = (lHz_x * fft_holo_d[idx].x - lHz_y * fft_holo_d[idx].y) / (Nx * Ny);
    plane_d[idx].y = (lHz_y * fft_holo_d[idx].x + lHz_x * fft_holo_d[idx].y) / (Nx * Ny);
}

__global__ void 
rs_phase_mult_kernel(float2* plane_d, float2* fft_holo_d, int Nx, int Ny, 
               double lambda, double reso, double z)
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
    double exponent = -2 * M_PI*z*sqrt(sqrt_input) / lambda;

    // keep local copies of Hx[idx].{x|y}
    double lHz_x = cos(exponent);
    double lHz_y = sin(exponent);
    
    float in_x = fft_holo_d[idx].x;
    float in_y = fft_holo_d[idx].y;

    // Result is complex multiplication of hologram fft with Hz kernel
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x = (lHz_x * in_x - lHz_y * in_y) / (Nx * Ny);
    plane_d[idx].y = (lHz_y * in_x + lHz_x * in_y) / (Nx * Ny);
}

__global__ void
kf_mult_kernel(float2* plane_d, float2* fft_holo_d, int Nx, int Ny,
double lambda, double reso, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    double fx = (double)(((x + 0 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
    double fy = (double)(((y + 0 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double exponent = (lambda * M_PI / (reso * reso)) * z * f2;

    // keep local copies of Hx[idx].{x|y}
    double lHz_x = cos(exponent);
    double lHz_y = -sin(exponent);
    float2 lHz;
    lHz.x = lHz_x;
    lHz.y = lHz_y;

    // Result is complex multiplication of hologram fft with Hz kernel
    // Pre divide by size to scale correctly after inverse FFT
    //plane_d[idx].x = (lHz_x * fft_holo_d[idx].x - lHz_y * fft_holo_d[idx].y) / (Nx * Ny);
    //plane_d[idx].y = (lHz_y * fft_holo_d[idx].x + lHz_x * fft_holo_d[idx].y) / (Nx * Ny);
    plane_d[idx] = cuCmulf(fft_holo_d[idx], lHz);
    plane_d[idx].x = plane_d[idx].x / (Nx * Ny);
    plane_d[idx].y = plane_d[idx].y / (Nx * Ny);
}

__global__ void
rs_kernel(float2* plane_d, int Nx, int Ny,
               double lambda, double reso, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    double fx = (double)(((x + 1 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
    double fy = (double)(((y + 1 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double sqrt_input = 1 - f2*(lambda / reso)*(lambda / reso);
    sqrt_input = (sqrt_input < 0) ? 0 : sqrt_input;
    double exponent = 2 * M_PI*z*sqrt(sqrt_input) / lambda;

    // Result is simply Hz
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x =  cos(exponent) / (Nx * Ny);
    plane_d[idx].y = -sin(exponent) / (Nx * Ny);
}

__global__ void
kf_kernel(float2* plane_d, int Nx, int Ny,
double lambda, double reso, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    double fx = (double)(((x + 1 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
    double fy = (double)(((y + 1 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double exponent = (lambda * M_PI / (reso * reso)) * z * f2;

    // Result is simply Hz
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x = cos(exponent) / (Nx * Ny);
    plane_d[idx].y = -sin(exponent) / (Nx * Ny);
}

__global__ void copy_abs_kernel(float* dest, float2* source, size_t Nx, size_t Ny)
{
    //Offset of pixel
    //int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * Nx;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {

        //Scale the output; CUFFT apparently ends up multiplying
        //the inverse FFT by the number of elements... so we need
        //to undo that (see CUFFT manual sec. 3.13.2)

        float xs = source[idx].x * source[idx].x;
        float ys = source[idx].y * source[idx].y;
        float val = sqrtf(xs + ys);
        //dest[idx] = min(255.0f, max(0.0f, val / (Ny * Nx)));
        //dest[idx] = min(255.0f, max(0.0f, val));
        dest[idx] = val;
    }

    return;
}

__global__ void copy_magsqr_kernel(float* dest, float2* source, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        float xs = source[idx].x * source[idx].x;
        float ys = source[idx].y * source[idx].y;
        float val = xs + ys;

        dest[idx] = val;
    }

    return;
}

__global__ void copy_real_kernel(float* dest, float2* source, size_t Nx, size_t Ny)
{
    //Offset of pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        //dest[idx] = min(255.0f, max(0.0f, source[idx].x));
        dest[idx] = source[idx].x;
        //float xs = source[idx].x * source[idx].x;
        //float ys = source[idx].y * source[idx].y;
        //float val = sqrtf(xs + ys);
        //dest[idx] = min(255.0f, max(0.0f, val));
    }

    return;
}

/*Reconstruction::Reconstruction(Hologram holo, ReconstructionInitialization init_method)
{
    switch (init_method)
    {
    case RECON_INIT_VOLUME:
    {
        std::cout << "Reconstruction::constructor: "
                  << "For init method RECON_INIT_VOLUME omit second argument"
                  << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
        break;
    }
    case RECON_INIT_PLANAR:
    {
        // Expected input, continue with processing
        break;
    }
    default:
    {
        std::cout << "Reconstruction::constructor: invalid init_method" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    // Duplicate much of the behavior of OpticalField constructor
    params = holo.getParams();
    width = holo.getData().getWidth();
    height = holo.getData().getHeight();
    depth = params.num_planes;
    complex_representation = REAL_IMAGINARY;

    if (holo.isState(HOLOGRAM_STATE_NORM_MEAND_ZERO))
        scale = SCALE_DECONV_NORM;
    else
        scale = SCALE_UNKNOWN;

    hologram_original_mean = holo.getOriginalMean();
    
    plan_created = false;
}*/

/** brief Reconstruct using specified kernel */
void Reconstruction::reconstruct(Hologram holo, ReconstructionMode mode)
{
    DECLARE_TIMING(reconstruct);
    START_TIMING(reconstruct);
    
    switch (mode)
    {
    case RECON_MODE_BASIC:
    {
        if (holo.isState(HOLOGRAM_STATE_NORM_MEAND) || holo.isState(HOLOGRAM_STATE_LOADED))
        {
            holo.fft();
            //this->recon_rs(holo);
            this->recon_basic(holo, KERNEL_KF);
            this->state = OPTICALFIELD_STATE_RECONSTRUCTED;
            holo.ifft();
            break;
        }
        else
        {
            std::cout << "Reconstruction::reconstruct: "
                      << "for RECON_MODE_BASIC hologram must be of state " 
                      << "HOLOGRAM_STATE_NORM_MEAND or HOLOGRAM_STATE_LOADED" 
                      << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
        break;
    }
    case RECON_MODE_DECONVOLVE:
    {
        if (holo.isState(HOLOGRAM_STATE_NORM_MEAND_ZERO))
        {
            holo.fft();
            this->recon_rs_complex(holo);
            this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
            holo.ifft();
            break;
        }
        else
            throw HOLO_ERROR_INVALID_STATE;
        break;
    }
    case RECON_MODE_FSP:
    {
        this->recon_rs_fsp(holo);
        this->state = OPTICALFIELD_STATE_PSF;
        break;
    }
    case RECON_MODE_COMPLEX:
    {
        holo.fft();
        this->recon_rs_phase_complex(holo);
        this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
        holo.ifft();
        break;
    }
    case RECON_MODE_COMPLEX_CONSTANT_SUM:
    {
        holo.fft();
        this->recon_rs_phase_complex(holo);
        //this->scaleData(1/params.num_planes);
        this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
        holo.ifft();
        break;
    }
    default:
    {
        throw HOLO_ERROR_UNKNOWN_MODE;
        break;
    }
    }
    
    SAVE_TIMING(reconstruct);
}

void Reconstruction::destroy()
{
    CHECK_MEMORY("Reconstruction::destroy begin");
    data.destroy();
    CHECK_MEMORY("Reconstruction::destroy step 1");
    buffer_data.destroy();
    CHECK_MEMORY("Reconstruction::destroy step 2");
    cufftDestroy(fft_plan);
    CHECK_MEMORY("Reconstruction::destroy end");
    return;
}

void Reconstruction::recon_basic(Hologram holo, ReconstructionKernel kernel)
{
    CHECK_FOR_ERROR("begin OpticalField::recon_rs");
    if (!holo.isState(HOLOGRAM_STATE_FOURIER)) throw HOLO_ERROR_INVALID_STATE;

    data.allocateCuData(width, height, depth, sizeof(float));

    float* data_d = (float*)data.getCuData();
    size_t plane_step = width * height;

    float2* holo_fft_d = (float2*)holo.getData().getCuData();

    float2* plane_d = NULL;
    cudaMalloc((void**)&plane_d, width * height * sizeof(float2));
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    cufftHandle fft_plan;
    cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
    //cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE);

    CHECK_FOR_ERROR("after mallocs");
    
    for (int zid = 0; zid < depth; ++zid)
    {
        float z = params.plane_list[zid];
        
        if (kernel == KERNEL_RS)
        {
            rs_mult_kernel<<<grid_dim, block_dim>>>
                (plane_d, holo_fft_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);
        }
        else if (kernel == KERNEL_KF)
        {
            kf_mult_kernel<<<grid_dim, block_dim>>>
                (plane_d, holo_fft_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);
        }
        else throw HOLO_ERROR_INVALID_ARGUMENT;
        CHECK_FOR_ERROR("rs_mult_kernel");

        cufftExecC2C(fft_plan, plane_d, plane_d, CUFFT_INVERSE);
        CHECK_FOR_ERROR("cufft inverse");

        float* plane_start_d = &(data_d[zid * plane_step]);
        copy_real_kernel<<<grid_dim, block_dim>>>(plane_start_d, plane_d, width, height);
        CHECK_FOR_ERROR("copy_abs_kernel");
    }

    cufftDestroy(fft_plan);

    CHECK_FOR_ERROR("end OpticalField::recon_rs");
    return;
}

void Reconstruction::recon_rs_complex(Hologram holo)
{
    CHECK_FOR_ERROR("begin OpticalField::recon_rs");
    if (!holo.isState(HOLOGRAM_STATE_FOURIER)) throw HOLO_ERROR_INVALID_STATE;
    if ((width == 0) || (height == 0))
    {
        std::cout << "Reconstruction::recon_rs_complex: Error: width or height is 0" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    data.allocateCuData(width, height, depth, sizeof(float2));

    float2* data_d = (float2*)data.getCuData();
    size_t plane_step = width * height;

    float2* holo_fft_d = (float2*)holo.getData().getCuData();

    float2* plane_d = NULL;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    if (!plan_created)
    {
        cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
        //cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE);
        plan_created = true;
    }

    CHECK_FOR_ERROR("after mallocs");
    
    float2* test_data_h = (float2*)malloc(width*height * sizeof(float2));
    CUDA_SAFE_CALL( cudaMemcpy(test_data_h, data_d, 1*sizeof(float2), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(test_data_h, data_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy(test_data_h, holo_fft_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost) );
    free(test_data_h);

    for (int zid = 0; zid < depth; ++zid)
    {
        float z = params.plane_list[zid];
        
        plane_d = data_d + zid * plane_step;
        rs_mult_kernel<<<grid_dim, block_dim>>>
            (plane_d, holo_fft_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);
        CHECK_FOR_ERROR("rs_mult_kernel");

        cufftExecC2C(fft_plan, plane_d, plane_d, CUFFT_INVERSE);
    }

    CHECK_FOR_ERROR("end OpticalField::recon_rs");
    return;
}

__global__ void phase_scale_kernel
    (float2* out_d, float2* in_d, size_t Nx, size_t Ny, double lambda, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    
    double phase = 2*M_PI*z/lambda;
    
    double p_x = cos(phase);
    double p_y = sin(phase);
    
    float2 old_data = in_d[idx];
    
    out_d[idx].x = old_data.x*p_x - old_data.y*p_y;
    out_d[idx].y = old_data.y*p_x + old_data.x*p_y;
}

void Reconstruction::recon_rs_phase_complex(Hologram holo)
{
    CHECK_FOR_ERROR("begin Reconstruction::recon_rs_phase_complex");
    if (!holo.isState(HOLOGRAM_STATE_FOURIER)) throw HOLO_ERROR_INVALID_STATE;
    if ((width == 0) || (height == 0))
    {
        std::cout << "Reconstruction::recon_rs_phase_complex: Error: width or height is 0" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    data.allocateCuData(width, height, depth, sizeof(float2));

    float2* data_d = (float2*)data.getCuData();
    size_t plane_step = width * height;

    float2* holo_fft_d = (float2*)holo.getData().getCuData();

    float2* plane_d = NULL;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    if (!plan_created)
    {
        cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
        plan_created = true;
    }

    CHECK_FOR_ERROR("after mallocs");

    for (int zid = 0; zid < depth; ++zid)
    {
        float z = params.plane_list[zid];
        
        plane_d = data_d + zid * plane_step;
        rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (plane_d, holo_fft_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);
        CHECK_FOR_ERROR("rs_mult_kernel");

        cufftExecC2C(fft_plan, plane_d, plane_d, CUFFT_INVERSE);
        
        phase_scale_kernel<<<grid_dim, block_dim>>>
            (plane_d, plane_d, width, height, holo.getParams().wavelength, z);
    }

    CHECK_FOR_ERROR("end Reconstruction::recon_rs_phase_complex");
    return;
}

void Reconstruction::recon_rs_fsp(Hologram holo)
{
    CHECK_FOR_ERROR("begin OpticalField::recon_rs");

    data.allocateCuData(width, height, depth, sizeof(float2));

    float2* data_d = (float2*)data.getCuData();
    size_t plane_step = width * height;

    float2* plane_d = NULL;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    cufftHandle fft_plan;
    cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
    //cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE);

    CHECK_FOR_ERROR("after mallocs");

    for (int zid = 0; zid < depth; ++zid)
    {
        float z = params.plane_list[zid];
        
        plane_d = data_d + zid * plane_step;
        rs_kernel<<<grid_dim, block_dim>>>
            (plane_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);
        CHECK_FOR_ERROR("rs_kernel");

        cufftExecC2C(fft_plan, plane_d, plane_d, CUFFT_INVERSE);
    }

    cufftDestroy(fft_plan);

    CHECK_FOR_ERROR("end OpticalField::recon_rs");
    return;
}

void Reconstruction::reconstructTo(Hologram& holo, float z, ReconstructionMode mode)
{
    if (holo.isState(HOLOGRAM_STATE_FOURIER))
    {
        std::cout << "Reconstruction::reconstructTo Error: Hologram cannot be in Fourier domain" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    CuMat holo_data = holo.getData();
    //std::cout << "begin reconstructTo:" << std::endl << holo_data.getReal().getMatData()(cv::Rect(0, 0, 10, 10)) << std::endl;
    //std::cout << "      imaginary:" << std::endl << holo_data.getImag().getMatData()(cv::Rect(0, 0, 10, 10)) << std::endl << std::endl;
    float2* holo_fft_d = (float2*)holo_data.getCuData();

    if (!plan_created)
    {
        CUFFT_SAFE_CALL( cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C) );
        //CUFFT_SAFE_CALL( cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE) );
        plan_created = true;
    }
    CUFFT_SAFE_CALL( cufftExecC2C(fft_plan, holo_fft_d, holo_fft_d, CUFFT_FORWARD) );

    float wavelength = holo.getParams().wavelength;
    float reso = holo.getParams().resolution;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    switch (mode)
    {
    case RECON_MODE_BASIC:
    {
        kf_mult_kernel<<<grid_dim, block_dim>>>
            (holo_fft_d, holo_fft_d, width, height, wavelength, reso, z);
        cufftExecC2C(fft_plan, holo_fft_d, holo_fft_d, CUFFT_INVERSE);
        break;
    }
    case RECON_MODE_COMPLEX:
    {
        rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (holo_fft_d, holo_fft_d, width, height, wavelength, reso, z);
        cufftExecC2C(fft_plan, holo_fft_d, holo_fft_d, CUFFT_INVERSE);
        phase_scale_kernel<<<grid_dim, block_dim>>>
            (holo_fft_d, holo_fft_d, width, height, wavelength, z);
        break;
    }
    default:
    {
        std::cout << "Reconstruction::reconstructTo: Unsupported mode: "
            << mode << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
        break;
    }
    }
    
    float holo_z = holo.getReconstructedPlane();
    holo.setReconstructedPlane(holo_z + z);
    
    holo_data.setCuData(holo_fft_d);
    holo.setData(holo_data);
    
    //imshow("end reconstructTo", holo_data.getReal().getMatData());
    //std::cout << "  wavelength = " << wavelength << ", resolution = " << reso << std::endl;
    //std::cout << "end reconstructTo:" << std::endl << holo_data.getReal().getMatData()(cv::Rect(0, 0, 10, 10)) << std::endl;
    //std::cout << "      imaginary:" << std::endl << holo_data.getImag().getMatData()(cv::Rect(0, 0, 10, 10)) << std::endl << std::endl;
    
    CHECK_FOR_ERROR("Reconstruction::reconstructTo");
}

void Reconstruction::reconstructTo(CuMat* out_plane, Hologram holo, float z, ReconstructionMode mode)
{
    if (!holo.isState(HOLOGRAM_STATE_FOURIER))
    {
        std::cout << "Reconstruction::reconstructTo: Hologram must be of state HOLOGRAM_STATE_FOURIER" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    out_plane->allocateCuData(width, height, 1, sizeof(float));

    float* out_data_d = (float*)out_plane->getCuData();

    float2* holo_fft_d = (float2*)holo.getData().getCuData();

    buffer_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* complex_plane_d = (float2*)buffer_data.getCuData();

    if (!plan_created)
    {
        cufftPlan2d(&fft_plan, holo.getHeight(), holo.getWidth(), CUFFT_C2C);
        //cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE);
        plan_created = true;
    }

    float wavelength = holo.getParams().wavelength;
    float reso = holo.getParams().resolution;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    switch (mode)
    {
    case RECON_MODE_BASIC:
    {
        kf_mult_kernel<<<grid_dim, block_dim>>>
            (complex_plane_d, holo_fft_d, width, height, wavelength, reso, z);
        cufftExecC2C(fft_plan, complex_plane_d, complex_plane_d, CUFFT_INVERSE);
        copy_real_kernel<<<grid_dim, block_dim>>>
            (out_data_d, complex_plane_d, width, height);
        break;
    }
    case RECON_MODE_COMPLEX:
    {
        rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (complex_plane_d, holo_fft_d, width, height, wavelength, reso, z);
        cufftExecC2C(fft_plan, complex_plane_d, complex_plane_d, CUFFT_INVERSE);
        phase_scale_kernel<<<grid_dim, block_dim>>>
            (complex_plane_d, complex_plane_d, width, height, wavelength, z);
        copy_real_kernel<<<grid_dim, block_dim>>>
            (out_data_d, complex_plane_d, width, height);
        break;
    }
    case RECON_MODE_COMPLEX_ABS:
    {
        printf("using RECON_MODE_COMPLEX_ABS\n");
        rs_mult_kernel<<<grid_dim, block_dim>>>
            (complex_plane_d, holo_fft_d, width, height, wavelength, reso, z);
        cufftExecC2C(fft_plan, complex_plane_d, complex_plane_d, CUFFT_INVERSE);
        copy_abs_kernel<<<grid_dim, block_dim>>>
            (out_data_d, complex_plane_d, width, height);
        break;
    }
    default:
    {
        std::cout << "Reconstruction::reconstructTo: Unsupported mode: "
            << mode << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
        break;
    }
    }
    
    CHECK_FOR_ERROR("Reconstruction::reconstructTo");
}

CuMat Reconstruction::combinedXY(Hologram& holo)
{
    if (this->state == OPTICALFIELD_STATE_RECONSTRUCTED) return OpticalField::combinedXY();

    if (!holo.isState(HOLOGRAM_STATE_LOADED))
    {
        std::cout << "Reconstruction::combinedXY Error: state must be HOLOGRAM_STATE_LOADED" << std::endl;
        std::cout << "State is " << this->state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    DECLARE_TIMING(combinedXY_total);
    DECLARE_TIMING(combinedXY_loop);
    DECLARE_TIMING(recon);
    DECLARE_TIMING(thresh);
    DECLARE_TIMING(mini);
    START_TIMING(combinedXY_total);

    cv::Mat cmb(height, width, CV_32F, cv::Scalar(255));
    CuMat cmb_data;
    cmb_data.setMatData(cmb);

    CuMat recon_plane;

    holo.fft();

    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        START_TIMING(combinedXY_loop);
        //printf("plane = %d of %d\n", zidx, params.num_planes);
        float z = params.plane_list[zidx];
        START_TIMING(recon);
        reconstructTo(&recon_plane, holo, z);
        STOP_TIMING(recon);

        START_TIMING(thresh);
        recon_plane.threshold(0.0, cv::THRESH_TOZERO);
        STOP_TIMING(thresh);
        START_TIMING(mini);
        min(cmb_data, recon_plane, cmb_data);
        STOP_TIMING(mini);
        STOP_TIMING(combinedXY_loop);
    }
    double ave_loop_time = GET_AVERAGE_TIMING(combinedXY_loop);
    printf("combinedXY: Average time for loop = %f ms (%d planes)\n", 
        ave_loop_time, params.num_planes);
    double ave_recon = GET_AVERAGE_TIMING(recon);
    double ave_thresh = GET_AVERAGE_TIMING(thresh);
    double ave_mini = GET_AVERAGE_TIMING(mini);
    printf("time for recon: %f, threshold: %f, min: %f\n", ave_recon, ave_thresh, ave_mini);

    cmb = cmb_data.getMatData();
    cmb.convertTo(cmb, CV_8U);

    CuMat result;
    result.setMatData(cmb);
    
    cmb_data.destroy();
    recon_plane.destroy();
    
    PRINT_TIMING(combinedXY_total);

    return result;
}

void Reconstruction::reconstructIntensityTo(CuMat* out_plane, Hologram holo, float z)
{
    assert(holo.isState(HOLOGRAM_STATE_FOURIER));

    out_plane->allocateCuData(width, height, 1, sizeof(float));

    float* out_data_d = (float*)out_plane->getCuData();

    float2* holo_fft_d = (float2*)holo.getData().getCuData();

    buffer_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* complex_plane_d = (float2*)buffer_data.getCuData();

    if (!plan_created)
    {
        //cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
        cufftPlan2d(&fft_plan, height, width, CUFFT_C2C);
        //cufftSetCompatibilityMode(fft_plan, CUFFT_COMPATIBILITY_NATIVE);
        plan_created = true;
    }
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    rs_mult_kernel<<<grid_dim, block_dim>>>
        (complex_plane_d, holo_fft_d, width, height, holo.getParams().wavelength, holo.getParams().resolution, z);

    cufftExecC2C(fft_plan, complex_plane_d, complex_plane_d, CUFFT_INVERSE);

    copy_magsqr_kernel<<<grid_dim, block_dim>>>(out_data_d, complex_plane_d, width, height);
    
    out_plane->setCuData(out_data_d);
    
    CHECK_FOR_ERROR("Reconstruction::reconstructIntensityTo");
}

CuMat Reconstruction::projectMaxIntensity(Hologram& holo)
{
    if (this->state == OPTICALFIELD_STATE_RECONSTRUCTED) return OpticalField::combinedXY();

    /*if (!holo.isState(HOLOGRAM_STATE_BG_DIVIDED))
    {
        std::cout << "Reconstruction::projectMaxIntensity Error: state must be HOLOGRAM_STATE_NORM_INVERSE" << std::endl;
        std::cout << "State is " << this->state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }*/
    
    DECLARE_TIMING(combinedXY_total);
    DECLARE_TIMING(combinedXY_loop);
    DECLARE_TIMING(recon);
    DECLARE_TIMING(thresh);
    DECLARE_TIMING(maxi);
    START_TIMING(combinedXY_total);

    cv::Mat cmb(height, width, CV_32F, cv::Scalar(0));
    CuMat cmb_data;
    cmb_data.setMatData(cmb);
    
    /*std::cout << "Hologram state is " << holo.getState() << std::endl;
    cv::namedWindow("HoloPMI", CV_WINDOW_AUTOSIZE);
    cv::Mat plane = holo.getData().getReal().getMatData();
    double plane_min, plane_max;
    cv::minMaxIdx(plane, &plane_min, &plane_max);
    plane = plane - plane_min;
    plane = plane / (plane_max - plane_min);
    cv::imshow("HoloPMI", plane);
    while (true)
    {
        char key = (char)cv::waitKey(1);
        if (key == 27) // 'esc' key was pressed
        {
            cv::destroyAllWindows();
            break;
        }
    }*/

    CuMat recon_plane;

    holo.fft();

    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        START_TIMING(combinedXY_loop);
        //printf("plane = %d of %d\n", zidx, params.num_planes);
        float z = params.plane_list[zidx];
        START_TIMING(recon);
        this->reconstructIntensityTo(&recon_plane, holo, z);
        STOP_TIMING(recon);
    
        /*cv::namedWindow("Recon", CV_WINDOW_AUTOSIZE);
        cv::Mat plane = recon_plane.getMatData();
        double plane_min, plane_max;
        cv::minMaxIdx(plane, &plane_min, &plane_max);
        plane = plane - plane_min;
        plane = plane / (plane_max - plane_min);
        cv::imshow("Recon", plane);
        while (true)
        {
            char key = (char)cv::waitKey(1);
            if (key == 27) // 'esc' key was pressed
            {
                cv::destroyAllWindows();
                break;
            }
        }*/

        START_TIMING(maxi);
        max(cmb_data, recon_plane, cmb_data);
        STOP_TIMING(maxi);
        STOP_TIMING(combinedXY_loop);
    }
    double ave_loop_time = GET_AVERAGE_TIMING(combinedXY_loop);
    //printf("combinedXY: Average time for loop = %f ms (%d planes)\n", 
    //    ave_loop_time, params.num_planes);
    double ave_recon = GET_AVERAGE_TIMING(recon);
    double ave_thresh = GET_AVERAGE_TIMING(thresh);
    double ave_maxi = GET_AVERAGE_TIMING(maxi);
    //printf("time for recon: %f, threshold: %f, max: %f\n", ave_recon, ave_thresh, ave_maxi);

    cmb = cmb_data.getMatData();
    //cmb.convertTo(cmb, CV_8U);

    CuMat result;
    result.setMatData(cmb);
    
    cmb_data.destroy();
    recon_plane.destroy();
    
    //PRINT_TIMING(combinedXY_total);

    return result;
}

void Reconstruction::projectMaxIntensity(Hologram& holo, CuMat cmb_out, CuMat arg_out)
{
    cv::Mat cmb(height, width, CV_32F, cv::Scalar(0));
    cv::Mat arg(height, width, CV_32F, cv::Scalar(0));
    cmb_out.setMatData(cmb);
    arg_out.setMatData(arg);
    
    CuMat recon_plane;
    
    holo.fft();
    
    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        float z = params.plane_list[zidx];
        this->reconstructIntensityTo(&recon_plane, holo, z);
        
        argmax(cmb_out, recon_plane, cmb_out, zidx, arg_out);
        // max(cmb_out, recon_plane, cmb_out);
    }
    
    // cmb = cmb_out.getMatData();
    // double minval = 0;
    // double maxval = 0;
    // cv::minMaxLoc(cmb, &minval, &maxval);
    // printf("Reconstruction::projectMaxIntensity: cmb min=%f, max=%f\n", minval, maxval);
    
    // arg = arg_out.getMatData();
    // cv::minMaxLoc(arg, &minval, &maxval);
    // printf("Reconstruction::projectMaxIntensity: arg min=%f, max=%f\n", minval, maxval);
    
    recon_plane.destroy();
}

__global__ void conj_mult_kernel(float2* data_d, int num_elements)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < num_elements)
    {
        data_d[idx] = cuCmulf(data_d[idx], cuConjf(data_d[idx]));
    }
}

void Reconstruction::multiply_conjugate()
{
    CHECK_FOR_ERROR("befor OpticalField::multiply_conjugate");
    if ((this->state != OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX) &&
            (this->state != OPTICALFIELD_STATE_PSF))
        throw HOLO_ERROR_INVALID_STATE;

    float2* data_d = (float2*)this->data.getCuData();
    
    int num_elements = width * height * depth;
    conj_mult_kernel<<< num_elements/1024, 1024>>>(data_d, num_elements);

    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX_REAL;
    CHECK_FOR_ERROR("OpticalField::multiply_conjugate");
    return;
}

__global__ void
focusMetric_kernel(float* A, float* B, float2* fft_holo_d, int Nx, int Ny,
double lambda, double reso, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    double fx = (double)(x/(double)Nx - 0.5);
    double fy = (double)(y/(double)Ny - 0.5);
    double f2 = fx*fx + fy*fy;

    double exponent = (lambda * M_PI / (reso * reso)) * z * f2;

    // keep local copies of Hx[idx].{x|y}
    double lHz_x = cos(exponent);
    double lHz_y = -sin(exponent);

    A[idx] =  abs(lHz_x * fft_holo_d[idx].x);
    B[idx] = -abs(lHz_y * fft_holo_d[idx].x);
}

FocusMetric Reconstruction::calcFocusMetric(Hologram* holo)
{
    if (!holo->isState(HOLOGRAM_STATE_NORM_INVERSE))
    {
        std::cout << "Hologram must be of state HOLOGRAM_STATE_NORM_INVERSE" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    cudaDeviceSynchronize();
    //DECLARE_TIMING(calcFocusMetric_startup);
    //DECLARE_TIMING(calcFocusMetric_kernel_sum);
    //DECLARE_TIMING(calcFocusMetric_final);

    //START_TIMING(calcFocusMetric_startup);

    holo->window(WINDOW_TUKEY, params.window_r);

    // Take FFT of hologram with prescribed zero padding
    holo->fft(params.zero_padding, params.zero_padding);
    width = params.zero_padding;
    height = params.zero_padding;

    holo->fftshift();
    holo->lowpass(WINDOW_GAUSSSIAN, params.window_alpha);

    // Allocate temporary space for initial focus metric data
    float* raw_real;
    float* raw_imag;
    cudaMalloc((void**)&raw_real, width * height * sizeof(float));
    cudaMalloc((void**)&raw_imag, width * height * sizeof(float));

    // Prep for kernel call
    float2* fft_holo_d = (float2*)holo->getData().getCuData();
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    // Prep for L1 norm calculation
    NppStatus err = NPP_SUCCESS;
    NppiSize oSizeROI = { width, height };
    //int buffer_size = 0;
	size_t buffer_size = 0;
    nppiSumGetBufferHostSize_32f_C1R(oSizeROI, &buffer_size);
    Npp8u* buffer;
    cudaMalloc((void**)&buffer, buffer_size);
    int source_step = width * sizeof(float);

    Npp64f* sum_raw_real;
    Npp64f* sum_raw_imag;
    cudaMalloc((void**)&sum_raw_real, params.num_planes * sizeof(Npp64f));
    cudaMalloc((void**)&sum_raw_imag, params.num_planes * sizeof(Npp64f));

    //SAVE_TIMING(calcFocusMetric_startup);

    //START_TIMING(calcFocusMetric_kernel_sum);
    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        float z = params.plane_list[zidx];
        focusMetric_kernel<<<grid_dim, block_dim>>>
            (raw_real, raw_imag, fft_holo_d, width, height, params.wavelength, params.resolution, z);
        cudaDeviceSynchronize();

        err = nppiSum_32f_C1R(raw_real, source_step, oSizeROI, buffer, sum_raw_real+zidx);
        err = nppiSum_32f_C1R(raw_imag, source_step, oSizeROI, buffer, sum_raw_imag+zidx);
        if (err != NPP_SUCCESS) throw HOLO_ERROR_UNKNOWN_ERROR;
        cudaDeviceSynchronize();
    }
    //SAVE_TIMING(calcFocusMetric_kernel_sum);

    //START_TIMING(calcFocusMetric_final);
    FocusMetric metric(params.num_planes);
    metric.setRawDevice(sum_raw_real, sum_raw_imag);

    cudaFree(raw_real);
    cudaFree(raw_imag);
    cudaFree(buffer);
    cudaFree(sum_raw_real);
    cudaFree(sum_raw_imag);
    //SAVE_TIMING(calcFocusMetric_final);

    CHECK_FOR_ERROR("Reconstruction::calcFocusMetric");
    return metric;
}

__global__ void
bulkFocusMetric_kernel(float* A, float* B, float2* fft_holo_d, int Nx, int Ny,
                       int num_images, double lambda, double reso, 
                       double start_z, double dz, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx3 = blockIdx.z * blockDim.z + threadIdx.z;
    int zid = idx3 % Nz;
    int image = idx3 / Nz;

    if ((x < Nx) && (y < Ny) && (zid < Nz) && (image < num_images))
    {
        int idx_read = image * Nx * Ny + y * Nx + x;

        // Only read hologram data once
        float2 in_val = fft_holo_d[idx_read];

        // Calculate f2
        // For non-shifted FFT
        double fx = (double)(x/(double)Nx - 0.5);
        double fy = (double)(y/(double)Ny - 0.5);
        double f2 = fx*fx + fy*fy;

        double z = start_z + dz * zid;
        double exponent = (lambda * M_PI / (reso * reso)) * z * f2;

        // keep local copies of Hx[idx].{x|y}
        double lHz_x = cos(exponent);
        double lHz_y = -sin(exponent);

        int idx_write = image*Nx*Ny*Nz + zid*Nx*Ny + y*Nx + x;
        A[idx_write] = abs(lHz_x * in_val.x);
        B[idx_write] = -abs(lHz_y * in_val.x);
    }
}

void Reconstruction::calcFocusMetric(HoloSequence* holo, FocusMetric* metrics, int* count)
{
    CHECK_FOR_ERROR("begin Reconstruction::calcFocusMetric (HoloSequence)");
    //CHECK_MEMORY("begining of calcFocusMetric");

    DECLARE_TIMING(fft);
    DECLARE_TIMING(reconstruction);
    DECLARE_TIMING(recon_kernel);
    DECLARE_TIMING(kernel_prep);
    START_TIMING(fft);

    // Window hologram before taking fft
    holo->window(WINDOW_TUKEY, params.window_r);
    //CHECK_MEMORY("after window");

    // Take FFT of hologram with prescribed zero padding
    holo->fft(params.zero_padding, params.zero_padding);
    //CHECK_MEMORY("after fft");

    holo->fftshift();
    holo->lowpass(WINDOW_GAUSSSIAN, params.window_alpha);
    cudaDeviceSynchronize();

    SAVE_TIMING(fft);
    START_TIMING(reconstruction);

    int width = holo->getWidth();
    int height = holo->getHeight();
    int num_images = holo->getNumHolograms();

    //CHECK_MEMORY("after prep stuff");

    // Allocate temporary space for initial focus metric data
    float* raw_real;
    float* raw_imag;
    CUDA_SAFE_CALL(cudaMalloc((void**)&raw_real, params.num_planes * num_images * width * height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&raw_imag, params.num_planes * num_images * width * height * sizeof(float)));
    CHECK_FOR_ERROR("mallocs");
    
    START_TIMING(kernel_prep);
    // Prep for kernel call
    double lambda = params.wavelength;
    double reso = params.resolution;
    double startz = params.start_plane;
    double dz = params.plane_stepsize;
    double nz = params.num_planes;
    float2* fft_holo_d = (float2*)holo->getCuData();
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y, num_images*nz);
    SAVE_TIMING(kernel_prep);
    
    // Primary kernel
    START_TIMING(recon_kernel);
    bulkFocusMetric_kernel<<<grid_dim, block_dim>>>
        (raw_real, raw_imag, fft_holo_d, width, height, num_images, 
         lambda, reso, startz, dz, nz);
    cudaDeviceSynchronize();
    SAVE_TIMING(recon_kernel);
    
    // Sum real and imaginary components for each plane to get metric
    NppStatus err = NPP_SUCCESS;
    NppiSize oSizeROI = { width, height };
    //int buffer_size = 0;
	size_t buffer_size = 0;
    nppiSumGetBufferHostSize_32f_C1R(oSizeROI, &buffer_size);
    Npp8u* buffer;
    cudaMalloc((void**)&buffer, buffer_size);
    int source_step = width * sizeof(float);
    Npp64f* raw_real_vec;
    Npp64f* raw_imag_vec;
    cudaMalloc((void**)&raw_real_vec, num_images * nz * sizeof(Npp64f));
    cudaMalloc((void**)&raw_imag_vec, num_images * nz * sizeof(Npp64f));
    for (int pidx = 0; pidx < num_images * nz; ++pidx)
    {
        size_t plane_size = width * height;
        err = nppiSum_32f_C1R(raw_real + pidx*plane_size, source_step, oSizeROI, buffer, raw_real_vec+pidx);
        err = nppiSum_32f_C1R(raw_imag + pidx*plane_size, source_step, oSizeROI, buffer, raw_imag_vec+pidx);
        if (err != NPP_SUCCESS) throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    for (int n = 0; n < num_images; ++n)
    {
        Npp64f* metric_real = raw_real_vec + (n*(int)nz);
        Npp64f* metric_imag = raw_imag_vec + (n*(int)nz);
        metrics[n].init(nz);
        metrics[n].setRawDevice(metric_real, metric_imag);
    }

    SAVE_TIMING(reconstruction);

    // Clear all temporary data
    cudaFree(raw_real);
    cudaFree(raw_imag);
    cudaFree(raw_real_vec);
    cudaFree(raw_imag_vec);
    cudaFree(buffer);

    *count = num_images;
    CHECK_FOR_ERROR("Reconstruction::calcFocusMetric (HoloSequence)");
    return;
}

__global__ void
plane_sum_kernel(float2* out, float2* in, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if ((x < Nx) && (y < Ny) && (z < Nz))
    {
        int in_idx = z*Nx*Ny + y*Nx + x;
        int out_idx = y*Nx + x;
        
        atomicAdd(&(out[out_idx].x), in[in_idx].x);///Nz);
        atomicAdd(&(out[out_idx].y), in[in_idx].y);///Nz);
    }
}

__global__ void discard_imag_kernel(float2* data, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        // Keep only real component as is
        data[idx].y = 0;
        
        //data[idx].x = data[idx].x * 2;
        
        // Keep intensity
        //float re = data[idx].x + 1;
        //float im = data[idx].y;
        //data[idx].x = re*re + im*im;
        //data[idx].y = 0;
        
        // Add cross-interference
        //float re = data[idx].x;/// + 1;
        //float im = data[idx].y;
        //data[idx].x = re*re + im*im + 2*re;
        //data[idx].y = 0;
    }
}

__global__ void simulate_recording_kernel(float2* data, float2* bg, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        // Add cross-interference
        float re = data[idx].x;
        float im = data[idx].y;
        float bgval = sqrt(bg[idx].x);
        data[idx].x = re*re + im*im + 2*bgval*re + bgval*bgval;
        data[idx].y = 0;
    }
}

__global__ void rect_conv_kernel(float2* data, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        double fx = (double)(((x + 1 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
        double fy = (double)(((y + 1 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
        
        double w = M_PI * sqrt(fx*fx + fy*fy);
        
    }
}

__global__ void low_pass_kernel(float2* data, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        double fx = (double)(((x + 1 + Nx / 2) % Nx)  + 1 - ((Nx / 2) + 1)) / (double)Nx;
        double fy = (double)(((y + 1 + Ny / 2) % Ny)  + 1 - ((Ny / 2) + 1)) / (double)Ny;
        double freq = sqrt(fx*fx + fy*fy);
        
        data[idx].x = (freq > 0.5)? data[idx].x : 0;
        data[idx].y = (freq > 0.5)? data[idx].y : 0;
    }
}

void Reconstruction::backPropagate(Hologram* holo, CuMat* buffer, ReconstructionMode mode)
{
    if (mode != RECON_MODE_COMPLEX)
    {
        std::cout << "Reconstruction::backPropagate: Error: Mode must be RECON_MODE_COMPLEX" << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
        return;
    }

    data.allocateCuData(width, height, depth, sizeof(float2));
    buffer->allocateCuData(width, height, depth, sizeof(float2));

    float2* data_d = (float2*)data.getCuData();
    float2* buffer_data_d = (float2*)buffer->getCuData();
    //cudaMemset(buffer_data_d, 0, width*height*depth*sizeof(float2));
    size_t plane_step = width * height;

    float2* plane_d = NULL;
    float2* buffer_plane_d = NULL;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    if (!plan_created)
    {
        cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
        plan_created = true;
    }

    // Propogate each plane independently to z=0;
    for (int zid = 0; zid < depth; ++zid)
    {
        float z = params.plane_list[zid];
        
        plane_d = data_d + zid * plane_step;
        buffer_plane_d = buffer_data_d + zid * plane_step;
        
        phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, plane_d, width, height, holo->getParams().wavelength, -z);
        
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        
        rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
             holo->getParams().wavelength, holo->getParams().resolution, -z);
        
        // Save inverse fft until after sum to save time
        //cufftExecC2C(fft_plan, temp_plane_d, buffer_plane_d, CUFFT_INVERSE);
    }
    
    // Sum planes together to get estimated hologram
    CuMat holo_data = holo->getData();
    holo_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* holo_d = (float2*)holo_data.getCuData();
    cudaMemset(holo_d, 0, width*height*sizeof(float2));
    block_dim.z = 1;
    grid_dim.z = depth;
    plane_sum_kernel<<<grid_dim, block_dim>>>
        (holo_d, buffer_data_d, width, height, depth);
    
    grid_dim.z = 1;
    //low_pass_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    //rect_conv_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);

    cufftExecC2C(fft_plan, holo_d, holo_d, CUFFT_INVERSE);
    
    // Actual estimate is only real component
    discard_imag_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    
    holo_data.setCuData((void*)holo_d);
    holo->setData(holo_data);
    //holo->setState(HOLOGRAM_STATE_NORM_MEAND_ZERO);
    int temp_state = holo->getState();
    
    // Apply low pass filter
    /*holo->fft();
    holo->fftshift();
    holo->lowpass(WINDOW_TUKEY, 0.5);
    holo->ifftshift();
    holo->ifft();
    holo->setState(temp_state);*/
    
    CHECK_FOR_ERROR("end Reconstruction::backPropagate");
}

/*void Reconstruction::backPropagate(Hologram* holo, Hologram& bg, CuMat* buffer, ReconstructionMode mode)
{
    if (mode != RECON_MODE_COMPLEX)
    {
        std::cout << "Reconstruction::backPropagate: Error: Mode must be RECON_MODE_COMPLEX" << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
        return;
    }
    if (!this->isState(OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX))
    {
        std::cout << "Reconstruction::backPropagate: Error: Volume must be reconstructed" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
        return;
    }

    data.allocateCuData(width, height, depth, sizeof(float2));
    buffer->allocateCuData(width, height, depth, sizeof(float2));

    float2* data_d = (float2*)data.getCuData();
    float2* buffer_data_d = (float2*)buffer->getCuData();
    //cudaMemset(buffer_data_d, 0, width*height*depth*sizeof(float2));
    size_t plane_step = width * height;

    float2* plane_d = NULL;
    float2* buffer_plane_d = NULL;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);

    if (!plan_created)
    {
        cufftPlan2d(&fft_plan, data.getRows(), data.getCols(), CUFFT_C2C);
        plan_created = true;
    }

    float z = params.start_plane;

    // Propogate each plane independently to z=0;
    for (int zid = 0; zid < depth; ++zid)
    {
        plane_d = data_d + zid * plane_step;
        buffer_plane_d = buffer_data_d + zid * plane_step;
        
        phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, plane_d, width, height, holo->getParams().wavelength, -z);
        
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        
        rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
             holo->getParams().wavelength, holo->getParams().resolution, -z);
        
        // Save inverse fft until after sum to save time
        //cufftExecC2C(fft_plan, temp_plane_d, buffer_plane_d, CUFFT_INVERSE);

        z += params.plane_stepsize;
    }
    
    // Sum planes together to get estimated hologram
    CuMat holo_data = holo->getData();
    holo_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* holo_d = (float2*)holo_data.getCuData();
    cudaMemset(holo_d, 0, width*height*sizeof(float2));
    block_dim.z = 1;
    grid_dim.z = depth;
    plane_sum_kernel<<<grid_dim, block_dim>>>
        (holo_d, buffer_data_d, width, height, depth);

    cufftExecC2C(fft_plan, holo_d, holo_d, CUFFT_INVERSE);
    
    // Actual estimate is only real component
    grid_dim.z = 1;
    //discard_imag_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    CHECK_FOR_ERROR("before simulate_recording_kernel\n");
    float2* bg_d = (float2*)bg.getData().getCuData();
    simulate_recording_kernel<<<grid_dim, block_dim>>>(holo_d, bg_d, width, height);
    CHECK_FOR_ERROR("after simulate_recording_kernel\n");
    
    holo_data.setCuData((void*)holo_d);
    holo->setData(holo_data);
    holo->setState(HOLOGRAM_STATE_NORM_MEAND_ZERO);
    
    CHECK_FOR_ERROR("end Reconstruction::backPropagate");
}*/

__global__ void scaleData_kernel
    (float2* data, double scale_factor, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx].x = data[idx].x * scale_factor;
        data[idx].y = data[idx].y * scale_factor;
    }
}

void Reconstruction::scaleData(double scale_factor)
{
    assert(this->data.isAllocated());
    float2* data_d = (float2*)this->data.getCuData();
    size_t numel = width*height*depth;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    scaleData_kernel<<<dim_grid, dim_block>>>(data_d, scale_factor, numel);
}
