#include "hologram.h"  // class implemented
#include "Test/test.h"
#include "reconstruction.h"

#include <math.h>
#include <sys/stat.h>

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

Hologram::Hologram()
{
    state = HOLOGRAM_STATE_EMPTY;
    scale = HOLOGRAM_SCALE_OTHER;
    original_mean = -1;
    sqrt_background_mean = 0.0;
    lower_saturation_limit = -1E9;
    upper_saturation_limit = 1E9;
    return;
}

Hologram::Hologram(const Parameters in)
{
    params = in;
    state = HOLOGRAM_STATE_EMPTY;
    scale = HOLOGRAM_SCALE_OTHER;
    original_mean = -1;
    sqrt_background_mean = 0.0;
    lower_saturation_limit = -1E9;
    upper_saturation_limit = 1E9;
    return;
}

void Hologram::init(const Parameters in)
{
    params = in;
    state = HOLOGRAM_STATE_EMPTY;
    scale = HOLOGRAM_SCALE_OTHER;
    original_mean = -1;
    sqrt_background_mean = 0.0;
    lower_saturation_limit = -1E9;
    upper_saturation_limit = 1E9;
    return;
}

void Hologram::destroy()
{
    CHECK_FOR_ERROR("begin Hologram::destroy");
    data.destroy();
    CHECK_FOR_ERROR("Hologram::destroy");
    return;
}

//============================= OPERATORS ====================================

void Hologram::copyTo(Hologram* output)
{
    // Plain data can just be copied
    output->params = this->params;
    output->state = this->state;
    output->scale = this->scale;
    output->original_mean = this->original_mean;
    output->original_scale = this->original_scale;
    output->lower_saturation_limit = this->lower_saturation_limit;
    output->upper_saturation_limit = this->upper_saturation_limit;
    output->reconstructed_plane = this->reconstructed_plane;
    output->previous_state = this->previous_state;
    output->sqrt_background_mean = this->sqrt_background_mean;
    
    // Need to handle actual data carefully
    size_t size = this->data.getDataSize();
    void* this_data_d = this->data.getCuData();
    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();
    
    if (output->data.getDataSize() != size)
    {
        if (output->data.getDataSize() > 0)
        {
            std::cout << "Hologram::copyTo: If allocated, holograms must be same size" << std::endl;
            throw HOLO_ERROR_INVALID_DATA;
        }
        // Otherwise, simply set the size of the output
        output->data.setWidth(width);
        output->data.setHeight(height);
        output->data.setDepth(1);
        output->data.setDataType(CV_32FC2);
    }
    if (!output->data.isAllocated())
    {
        output->data.allocateCuData();
    }
    
    void* out_data_d = output->data.getCuData();
    cudaMemcpy(out_data_d, this_data_d, size, cudaMemcpyDeviceToDevice);
    output->data.setCuData(out_data_d);
    
    // Next handle background data
    size_t bg_size = this->bg_data.getDataSize();
    if (bg_size != 0)
    {
        void* this_bg_data_d = this->bg_data.getCuData();
        size_t bg_width = this->bg_data.getWidth();
        size_t bg_height = this->bg_data.getHeight();
        
        if (output->bg_data.getDataSize() != bg_size)
        {
            if (output->bg_data.getDataSize() > 0)
            {
                std::cout << "Hologram::copyTo: If allocated, holograms must be same size" << std::endl;
                throw HOLO_ERROR_INVALID_DATA;
            }
            // Otherwise, simply set the size of the output
            output->bg_data.setWidth(bg_width);
            output->bg_data.setHeight(bg_height);
            output->bg_data.setDepth(1);
            output->bg_data.setDataType(CV_32FC2);
        }
        if (!output->bg_data.isAllocated())
        {
            output->bg_data.allocateCuData();
        }
        
        void* out_bg_data_d = output->bg_data.getCuData();
        cudaMemcpy(out_bg_data_d, this_bg_data_d, bg_size, cudaMemcpyDeviceToDevice);
        output->bg_data.setCuData(out_bg_data_d);
    }
    
    CHECK_FOR_ERROR("Hologram::copyTo");
    return;
}

//============================= OPERATIONS ===================================

/** Use OpenCV imread function to read data */
void Hologram::read(char* filename)
{
    DECLARE_TIMING(hologram_read);
    START_TIMING(hologram_read);
    cv::Mat mat_data_R;
    try
    {
        mat_data_R  = cv::imread(filename, cv::IMREAD_UNCHANGED);
    }
    catch (cv::Exception& e)
    {
        std::cout << "Error in cv::imread: " << e.what() << std::endl;
    }
    if (mat_data_R.empty())
    {
        std::cout << "Unable to read file: " << filename << std::endl;
        struct stat buffer;   
        bool file_exists =  (stat(filename, &buffer) == 0); 
        std::cout << "Test whether file exists: " << file_exists << std::endl;
        bool file_canread;
        if (FILE *file = fopen(filename, "r")) {
            fclose(file);
            file_canread = true;
        } else {
            file_canread = false;
        }
        std::cout << "Test whether file can be read: " << file_canread << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }

    // Convert color images
    if (mat_data_R.type() == CV_8UC3)
    {
        // Assume default OpenCV BGR format for color images
        cv::cvtColor(mat_data_R, mat_data_R, cv::COLOR_BGR2GRAY, 1);
    }

    if (mat_data_R.type() != CV_8U && mat_data_R.type() != CV_16U)
    {
        std::cout << "Unsupported image type, expected 8bit grayscale" << std::endl;
        std::cout << "Type was " << mat_data_R.type() << ", expected CV_8U = " << CV_8U << std::endl;
	std::cout << "  also would have accepted CV_16U = " << CV_16U << std::endl;
    }

    cv::Mat planes[] = { mat_data_R, cv::Mat::zeros(mat_data_R.size(), mat_data_R.type()) };
    cv::Mat mat_data_C;
    cv::merge(planes, 2, mat_data_C);         // Add to the expanded another plane with zeros
    mat_data_C.convertTo(mat_data_C, CV_32FC2);

    data.setMatData(mat_data_C);
    state = HOLOGRAM_STATE_LOADED;
    scale = HOLOGRAM_SCALE_0_255;
    
    lower_saturation_limit = 0.0;
    upper_saturation_limit = 255.0;

    STOP_TIMING(hologram_read);
    SAVE_TIMING(hologram_read);

    return;
}

double Hologram::read(char* filename, cv::Size output_size)
{
    this->read(filename);
    
    size_t raw_width = this->getWidth();
    size_t raw_height = this->getHeight();
    
    double width_ratio = (double)output_size.width / raw_width;
    double height_ratio = (double)output_size.height / raw_height;
    
    double ratio = std::min(width_ratio, height_ratio);
    
    cv::Mat src = this->data.getMatData();
    cv::resize(src, src, cv::Size(0,0), ratio, ratio, cv::INTER_AREA);
    cv::Scalar ave = cv::mean(src);
    std::cout << "Is Scalar average real? " << ave.isReal() << std::endl;
    
    //cv::Mat dst = cv::Mat::ones(output_size, CV_32FC2);
    cv::Mat dst(output_size, CV_32FC2, ave);
    
    cv::Rect roi(0, 0, src.cols, src.rows);
    src.copyTo(dst(roi));
    
    this->data.setMatData(dst);
    this->params.resolution /= ratio;
    return ratio;
}

void Hologram::load(char* filename, CuMatFileMode format, HologramState state)
{
    data.load(filename, format);
    data.convertTo(CV_32FC2);
    this->state = state;
}

/** Normalize on host using OpenCV functions */
void Hologram::normalize()
{
    if (state != HOLOGRAM_STATE_LOADED)
    {
        std::cout << "Hologram::normalize: Error: Incorrect state\n" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    cv::Mat img = data.getMatData();
    img.convertTo(img, CV_32F);

    cv::Scalar avg = cv::mean(img);
    original_mean = avg.val[0];
    original_scale = scale;

    data.setMatData(img / avg.val[0]);

    state = HOLOGRAM_STATE_NORM_MEAND;
    scale = HOLOGRAM_SCALE_OTHER;
    
    upper_saturation_limit = upper_saturation_limit / avg.val[0];

    return;
}

__global__ void normInverse_kernel(float2* img_d, float* mean, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        float avg = *mean * 2;

        img_d[idx].x = 1 - img_d[idx].x / avg;
    }
}

void Hologram::normalize_inverse()
{
    if (state != HOLOGRAM_STATE_LOADED)
    {
        std::cout << "Hologram::normalize_inverse: Error: Incorrect state\n" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    /** OpenCV host implementation */
    cv::Mat img = data.getMatData();
    img.convertTo(img, CV_32F);

    cv::Scalar avg = cv::mean(img);
    original_mean = avg.val[0];
    original_scale = scale;

    data.setMatData(1 - img / avg.val[0]);
    
    lower_saturation_limit = 1.0 - (lower_saturation_limit / avg.val[0]);
    upper_saturation_limit = 1.0 - (upper_saturation_limit / avg.val[0]);

    //*/

    /** CUDA implementation /
    float2* img_d = (float2*)data.getCuData();

    // Mean calculation
    int width = this->data.getWidth();
    int height = this->data.getHeight();

    int buffer_size;
    nppsMeanGetBufferSize_32f(2*width*height, &buffer_size);

    float* mean_d;
    Npp8u* buffer;
    cudaMalloc((void**)&mean_d, sizeof(float2));
    cudaMalloc((void**)&buffer, buffer_size*sizeof(Npp8u));

    // Cheat with the complex componenet because we know it is 0
    nppsMean_32f((float*)img_d, 2*width*height, mean_d, buffer);

    // Now perform the normalization
    int blockDim = 256;
    int gridDim = ceil(width*height / 256.0);
    normInverse_kernel<<<gridDim, blockDim>>>(img_d, mean_d, width*height);
    cudaDeviceSynchronize();

    cudaFree(mean_d);
    cudaFree(buffer);
    //*/

    state = HOLOGRAM_STATE_NORM_INVERSE;
    scale = HOLOGRAM_SCALE_OTHER;
    CHECK_FOR_ERROR("Hologram::normalize_inverse");
    return;
}

/** Normalize on host using OpenCV functions */
void Hologram::normalize_mean0()
{
    if (state != HOLOGRAM_STATE_LOADED)
    {
        std::cout << "Hologram::normalize_mean0: Error: Incorrect state\n" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    cv::Mat img = data.getMatData();
    img.convertTo(img, CV_32F);

    cv::Scalar avg = cv::mean(img);
    original_mean = avg.val[0];
    original_scale = scale;

    cv::Mat result = img / avg.val[0];
    cv::subtract(result, cv::Scalar(1, 0), result);
    data.setMatData(result);
    
    lower_saturation_limit = (lower_saturation_limit / avg.val[0]) - 1.0;
    upper_saturation_limit = (upper_saturation_limit / avg.val[0]) - 1.0;

    state = HOLOGRAM_STATE_NORM_MEAND_ZERO;
    scale = HOLOGRAM_SCALE_OTHER;

    return;
}

void Hologram::reverse_normalize()
{
    if (state == HOLOGRAM_STATE_NORM_MEAND)
    {
        cv::Mat img = data.getMatData();
        img.convertTo(img, CV_32F);

        data.setMatData(img * original_mean);
        lower_saturation_limit = lower_saturation_limit * original_mean;
        upper_saturation_limit = upper_saturation_limit * original_mean;
    }
    else if (state == HOLOGRAM_STATE_NORM_INVERSE)
    {
        cv::Mat img = data.getMatData();
        img.convertTo(img, CV_32F);

        data.setMatData((1 - img) * original_mean);
        lower_saturation_limit = (1.0 - lower_saturation_limit) * original_mean;
        upper_saturation_limit = (1.0 - upper_saturation_limit) * original_mean;
    }
    else if (state == HOLOGRAM_STATE_NORM_MEAND_ZERO)
    {
        cv::Mat img = data.getMatData();
        img.convertTo(img, CV_32F);

        cv::add(img, cv::Scalar(1, 0), img);
        cv::Mat result = img * original_mean;
        data.setMatData(result);
        lower_saturation_limit = (1.0 + lower_saturation_limit) * original_mean;
        upper_saturation_limit = (1.0 + upper_saturation_limit) * original_mean;
    }
    else
    {
        std::cout << "Hologram::reverse_normalize Error: Unknown state" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    state = HOLOGRAM_STATE_LOADED;
    scale = original_scale;
}

void Hologram::rescale(HologramScale new_scale)
{
    if (new_scale == this->scale) return;

    if (new_scale == HOLOGRAM_SCALE_0_1)
    {
        if (scale == HOLOGRAM_SCALE_0_255)
        {
            cv::Mat matdata = this->data.getMatData();
            matdata = matdata / 255;
            this->data.setMatData(matdata);
            this->scale = new_scale;
            upper_saturation_limit = upper_saturation_limit / 255;
        }
        else 
        {
            std::cout << "Hologram::rescale scale must be HOLOGRAM_SCALE_0_255 to scale to HOLOGRAM_SCALE_0_1" << std::endl;
            std::cout << "Scale is " << scale << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
    }
    else if (new_scale == HOLOGRAM_SCALE_0_255)
    {
        if (scale == HOLOGRAM_SCALE_0_1)
        {
            cv::Mat matdata = this->data.getMatData();
            matdata = matdata * 255;
            this->data.setMatData(matdata);
            this->scale = new_scale;
            upper_saturation_limit = upper_saturation_limit * 255;
        }
        else
        {
            std::cout << "Hologram::rescale scale must be HOLOGRAM_SCALE_0_1 to scale to HOLOGRAM_SCALE_0_255" << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }
    }
    else throw HOLO_ERROR_INVALID_ARGUMENT;
}

void Hologram::mag()
{
    cv::Mat planes[2];
    cv::split(this->data.getMatData(), planes);
    
    cv::magnitude(planes[0], planes[1], planes[0]);
    planes[1] = cv::Mat::zeros(planes[1].size(), CV_32F);
    
    cv::Mat complex;
    merge(planes, 2, complex);

    this->data.setMatData(complex);
}

void Hologram::matchMeanStd(cv::Scalar mean, cv::Scalar std)
{
    cv::Mat unscaled;
    this->data.getMatData().copyTo(unscaled);
    cv::Scalar new_mean, new_std;
    cv::meanStdDev(unscaled, new_mean, new_std);
    //printf("Hologram::matchMeanStd: old mean = %f, std = %f, new mean = %f, std = %f\n",
    //    mean.val[0], std.val[0], new_mean.val[0], new_std.val[0]);
    
    cv::Mat scaled = mean + (unscaled - new_mean)*(std.val[0]/new_std.val[0]);
    //cv::Mat scaled = unscaled - new_mean;
    //scaled = scaled * (std / new_std);
    //scaled = scaled + mean;
    this->data.setMatData(scaled);
    return;
}

__global__ void round_kernel(float2* data_d, size_t size)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int idx = tid + bid*blockDim.x*blockDim.y;

    if (idx < size)
    {
        data_d[idx].x = round(data_d[idx].x);
        data_d[idx].y = round(data_d[idx].y);
    }
}

void Hologram::round()
{
    if (this->scale != HOLOGRAM_SCALE_0_255)
    {
        std::cout << "Hologram::round: Scale must be HOLOGRAM_SCALE_0_255" << std::endl;
        std::cout << "    is: " << this->scale << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    if (this->data.getDataType() != CV_32FC2)
    {
        std::cout << "Hologram::round Error: Type must be CV_32FC2" << std::endl;
        std::cout << "    is: " << this->scale << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();
    int tile_size = 16;
    dim3 gridDim(width/tile_size, height/tile_size, 1);
    dim3 blockDim(tile_size, tile_size, 1);
    size_t numel = width*height;
    
    round_kernel<<<gridDim, blockDim>>>((float2*)this->data.getCuData(), numel);
}

void Hologram::fft()
{
    fft(data.getCols(), data.getRows());
    return;
}

/** Use cuda FFT functions */
void Hologram::fft(int Nx, int Ny)
{
    CHECK_FOR_ERROR("begin Hologram::fft");
    if (this->data.getDataType() != CV_32FC2)
        this->data.convertTo(CV_32FC2);

    int width = this->data.getWidth();
    int height = this->data.getHeight();
    if (((Nx > width) && (Ny < height)) || ((Nx < width) && (Ny > height)))
    {
        std::cout << "Hologram::fft: Image size is incompatible" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    // Check for base 2
    if ((Nx & (Nx-1)) || (Ny & (Ny-1)))
    {
        std::cout << "Hologram::fft: Image must be base 2 size" << std::endl
            << "Size was " << Nx << "x" << Ny << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    cufftHandle initialFFT;
    cufftPlan2d(&initialFFT, Nx, Ny, CUFFT_C2C);
    //cufftSetCompatibilityMode(initialFFT, CUFFT_COMPATIBILITY_NATIVE);

    if ((Nx > this->data.getWidth()) || (Ny > this->data.getHeight()))
    {
        cv::Mat padded;
        int bot_pad = Ny - height;
        int right_pad = Nx - width;
        cv::copyMakeBorder(this->data.getMatData(), padded, 0, bot_pad, 0, right_pad, 
            cv::BORDER_CONSTANT, cv::Scalar(0));
        this->data.setMatData(padded);
    }
    else
    {
        cv::Mat cropped;
        this->data.getMatData()(cv::Rect(0, 0, Nx, Ny)).copyTo(cropped);
        this->data.setMatData(cropped);
    }

    cufftComplex* data_d = (cufftComplex*)data.getCuData();

    cufftExecC2C(initialFFT, data_d, data_d, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    CHECK_FOR_ERROR("Hologram::fft");

    cufftDestroy(initialFFT);
    cudaDeviceSynchronize();

    //dft(this->data.getMatData(), this->data.getMatData());

    this->state = HOLOGRAM_STATE_FOURIER;

    return;
}

void Hologram::ifft()
{
    ifft(data.getRows(), data.getCols());
    return;
}

__global__ void ifft_scale_kernel(float2* in, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        in[idx].x = in[idx].x / size;
        in[idx].y = in[idx].y / size;
    }
}

void Hologram::ifft(int Nx, int Ny)
{
    if (this->state != HOLOGRAM_STATE_FOURIER)
    {
        std::cout << "Must take forward fft before inverse" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    CHECK_FOR_ERROR("begin Hologram::ifft");
    if (this->data.getDataType() != CV_32FC2)
        this->data.convertTo(CV_32FC2);

    int width = this->data.getWidth();
    int height = this->data.getHeight();
    if ((Nx > width) && (Ny < height)) throw HOLO_ERROR_INVALID_ARGUMENT;
    if ((Nx < width) && (Ny > height)) throw HOLO_ERROR_INVALID_ARGUMENT;

    cufftHandle initialFFT;
    cufftPlan2d(&initialFFT, Nx, Ny, CUFFT_C2C);

    if ((Nx > this->data.getWidth()) || (Ny > this->data.getHeight()))
    {
        cv::Mat padded;
        int bot_pad = Ny - height;
        int right_pad = Nx - width;
        cv::copyMakeBorder(this->data.getMatData(), padded, 0, bot_pad, 0, right_pad, 
            cv::BORDER_CONSTANT, cv::Scalar(0));
        this->data.setMatData(padded);
    }
    else
    {
        cv::Mat cropped;
        this->data.getMatData()(cv::Rect(0, 0, Nx, Ny)).copyTo(cropped);
        this->data.setMatData(cropped);
    }

    cufftComplex* data_d = (cufftComplex*)data.getCuData();

    cufftExecC2C(initialFFT, data_d, data_d, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    //ifft_scale_kernel<<<ceil(256 / (float)Nx*Ny), 256>>>(data_d, Nx*Ny);
    ifft_scale_kernel<<<ceil((float)Nx*Ny/(float)256), 256>>>(data_d, Nx*Ny);

    CHECK_FOR_ERROR("Hologram::ifft");

    cufftDestroy(initialFFT);
    cudaDeviceSynchronize();

    this->state = HOLOGRAM_STATE_LOADED;

    return;
}

__global__ void fftshift_kernel(float2* in, float2* buffer, int nX, int nY, int nZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z*nX*nY + y * nX + x;

    int xin = x + nX/2 - (x/(nX/2))*nX;
    int yin = y + nY/2 - (y/(nY/2))*nY;
    int zin = z + ceil(nZ/2.0) - (z >= (nZ/2))*nZ;
    int idxin = nX*nY*zin + nX*yin + xin;

    // buffer is needed because it is not a simple flip for odd nZ
    // FIXME: This is very inefficient. There must be a better way.
    in[idx] = buffer[idxin];
}

void Hologram::fftshift()
{
    if (this->state == HOLOGRAM_STATE_FOURIER_SHIFTED) return;
    if (this->state != HOLOGRAM_STATE_FOURIER)
    {
        std::cout << "FFT must be applied before using fftshift" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    /** OpenCV Host Implementation */
    cv::Mat f = this->data.getMatData();

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = f.cols / 2;
    int cy = f.rows / 2;
    
    if ((f.cols != cx*2) || (f.rows != cy*2))
    {
        std::cout << "Hologram::ifftshift: rows and columns must be divisible by 2" << std::endl;
        std::cout << "Rows = " << f.rows << ", Cols = " << f.cols << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }

    cv::Mat q0(f, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(f, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(f, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(f, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    this->data.setMatData(f);
    //*/

    /** CUDA implementation /
    float2* data_d = (float2*)this->data.getCuData();
    float2* buffer_d;
    cudaMalloc((void**)&buffer_d, this->data.getDataSize());
    cudaMemcpy(buffer_d, data_d, this->data.getDataSize(), cudaMemcpyDeviceToDevice);

    int width = this->data.getWidth();
    int height = this->data.getHeight();
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(ceil(width / (float)blockDim.x), ceil(height / (float)blockDim.y), 1);
    fftshift_kernel<<<gridDim, blockDim>>>(data_d, buffer_d, width, height, 1);
    cudaDeviceSynchronize();

    cudaFree(buffer_d);
    //*/

    this->state = HOLOGRAM_STATE_FOURIER_SHIFTED;
    CHECK_FOR_ERROR("Hologram::fftshift");
    return;
}

void Hologram::ifftshift()
{
    if (this->state != HOLOGRAM_STATE_FOURIER_SHIFTED)
    {
        std::cout << "FFT shift must be applied before using ifftshift" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    /** OpenCV Host Implementation */
    cv::Mat f = this->data.getMatData();

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = f.cols / 2;
    int cy = f.rows / 2;
    
    if ((f.cols != cx*2) || (f.rows != cy*2))
    {
        std::cout << "Hologram::ifftshift: rows and columns must be divisible by 2" << std::endl;
        std::cout << "Rows = " << f.rows << ", Cols = " << f.cols << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }

    cv::Mat q0(f, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(f, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(f, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(f, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    this->data.setMatData(f);
    //*/

    this->state = HOLOGRAM_STATE_FOURIER;
    CHECK_FOR_ERROR("Hologram::ifftshift");
    return;
}

Hologram Hologram::crop()
{
    if (this->params.roi.area() > 0.0)
    {
        return this->crop(this->params.roi);
    }
    
    cv::Point center;
    center.x = this->params.center_point.x;
    center.y = this->params.center_point.y;
    int size = this->params.roi_size;
    
    return this->crop(center, size);
}

Hologram Hologram::crop(cv::Point center, int size)
{
    int start_x = center.x - size/2;
    int start_y = center.y - size/2;
    int roi_width = size;
    int roi_height = size;

    // Check to make sure roi doesn't extend out of the image
    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (start_x + size > this->data.getWidth())
        start_x = this->data.getWidth() - size;
    if (start_y + size > this->data.getHeight())
        start_y = this->data.getHeight() - size;
    
    if (this->data.getWidth() < size)
    {
        start_x = 0;
        roi_width = this->data.getWidth();
    }
    if (this->data.getHeight() < size)
    {
        start_y = 0;
        roi_height = this->data.getHeight();
    }
    
    if (size < 1)
    {
        start_x = 0;
        start_y = 0;
        roi_width = this->data.getWidth();
        roi_height = this->data.getHeight();
    }

    cv::Rect roi(start_x, start_y, roi_width, roi_height);
    this->params.roi = roi;

    return this->crop(roi);
}

Hologram Hologram::crop(cv::Rect roi)
{
    Hologram out(this->params);

    cv::Mat cropped;
    this->data.getMatData().copyTo(cropped);
    
    // Determine if a border will be needed
    int top_border = 0;
    int bottom_border = 0;
    int left_border = 0;
    int right_border = 0;
    int width = cropped.cols;
    int height = cropped.rows;
    if (roi.y < 0)
    {
        top_border = 0 - roi.y;
        roi.y = 0;
        height += top_border;
    }
    if (roi.y + roi.height > height)
    {
        bottom_border = roi.y + roi.height - height;
        height = height + bottom_border;
    }
    if (roi.x < 0)
    {
        left_border = 0 - roi.x;
        roi.x = 0;
        width += left_border;
    }
    if (roi.x + roi.width > width)
    {
        right_border = roi.x + roi.height - width;
        width = width + right_border;
    }
    cv::copyMakeBorder(cropped, cropped,
        top_border, bottom_border, left_border, right_border,
        cv::BORDER_CONSTANT, 0.0);
    
    cropped = cropped(roi);
    out.data.setMatData(cropped);
    out.state = this->state;
    out.scale = this->scale;
    out.original_mean = this->original_mean;
    out.original_scale = this->original_scale;
    out.reconstructed_plane = this->reconstructed_plane;
    out.previous_state = this->reconstructed_plane;
    out.lower_saturation_limit = this->lower_saturation_limit;
    out.upper_saturation_limit = this->upper_saturation_limit;
    out.sqrt_background_mean = this->sqrt_background_mean;
    
    if (this->bg_data.getDataSize() > 0)
    {
        cv::Mat bg_cropped;
        this->bg_data.getMatData().copyTo(bg_cropped);
        cv::copyMakeBorder(bg_cropped, bg_cropped,
            top_border, bottom_border, left_border, right_border,
            cv::BORDER_CONSTANT, 0.0);
        bg_cropped = bg_cropped(roi);
        out.bg_data.setMatData(bg_cropped);
    }

    return out;
}

Hologram Hologram::zeroPad(size_t padding)
{
    Hologram out(this->params);
    out.state = this->state;
    out.scale = this->scale;
    out.original_mean = this->original_mean;
    out.original_scale = this->original_scale;
    out.reconstructed_plane = this->reconstructed_plane;
    out.previous_state = this->reconstructed_plane;
    out.lower_saturation_limit = this->lower_saturation_limit;
    out.upper_saturation_limit = this->upper_saturation_limit;
    out.sqrt_background_mean = this->sqrt_background_mean;
    
    
    size_t width = this->getWidth();
    size_t height = this->getHeight();
    cv::Mat padded = cv::Mat::zeros(width+2*padding, height+2*padding, CV_32FC2);
    
    cv::Rect roi(padding, padding, width, height);
    this->data.getMatData().copyTo(padded(roi));
    
    /*
    std::cout << "zero-padding with size: " << padding << std::endl;
    cv::Mat input_data = this->data.getMatData();
    cv::namedWindow("non-padded", cv::WINDOW_AUTOSIZE);
    cv::Mat planes[2];
    cv::split(input_data, planes);
    double min_val, max_val;
    cv::minMaxLoc(planes[0], &min_val, &max_val);
    cv::normalize(planes[0], planes[0], 0, 1, cv::NORM_MINMAX);
    cv::imshow("non-padded", planes[0]);
    
    //cv::Mat padded;
    //cv::copyMakeBorder(input_data, padded,
    //    padding, padding, padding, padding, cv::BORDER_CONSTANT, 0.0);
    
    cv::namedWindow("padded", cv::WINDOW_AUTOSIZE);
    cv::Mat planes2[2];
    cv::split(padded, planes2);
    cv::minMaxLoc(planes2[0], &min_val, &max_val);
    cv::normalize(planes2[0], planes2[0], 0, 1, cv::NORM_MINMAX);
    cv::imshow("padded", planes2[0]);
    cv::waitKey(0);
    cv::destroyAllWindows();
    */
    
    out.data.setMatData(padded);
    
    if (this->bg_data.getDataSize() > 0)
    {
        cv::Mat bg_padded = cv::Mat::zeros(width+2*padding, height+2*padding, CV_32FC2);
        this->bg_data.getMatData().copyTo(bg_padded(roi));
        out.bg_data.setMatData(bg_padded);
    }
    
    return out;
}

void Hologram::setToMean()
{
    this->data.setToMean();
    return;
}

__global__ void tukeyMult_kernel(float2* data, int n, float r)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * n;

    if ((x < n) && (y < n))
    {
        // Don't worry about the cases where r < 0 or r > 1
        double period = r / 2.0;

        // Replicate linspace behavior
        float interval = 1.0 / (n-1);
        float tval_x = x * interval;
        float tval_y = y * interval;

        // Create the window in three section (defined by tl, th)
        double tl = floor(period * (n - 1)) + 1;
        double th = n - tl + 1;
        float w_x = (x < tl) ? (1 + cos(M_PI / period * (tval_x - period))) / 2.0 :
            (x < th) ? 1.0 :
            (1 + cos(M_PI / period * (tval_x - 1.0 + period))) / 2.0;
        float w_y = (y < tl) ? (1 + cos(M_PI / period * (tval_y - period))) / 2.0 :
            (y < th) ? 1.0 :
            (1 + cos(M_PI / period * (tval_y - 1.0 + period))) / 2.0;
        
        data[idx].x = data[idx].x * w_x * w_y;
        data[idx].y = data[idx].y * w_x * w_y;
    }
}

void Hologram::window(WindowMethod method, float window_param)
{
    if (method != WINDOW_TUKEY)
    {
        std::cout << "Windows other than Tukey are not yet implemented" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    int size = data.getWidth();

    /** OpenCV Host Implementation */
    CuMat win = createTukeyWindow(size, window_param);
    win.convertTo(CV_32FC2);
    umnholo::mult(data, win, data);

    win.destroy();
    //*/

    /** CUDA Implementation /
    float2* img_d = (float2*)data.getCuData();
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(size / (float)blockDim.x), ceil(size / (float)blockDim.y));
    tukeyMult_kernel<<<gridDim, blockDim>>>(img_d, size, window_param);
    //*/

    CHECK_FOR_ERROR("Hologram::window");
    return;
}

__global__ void gaussianMult_kernel(float2* data, int L, float a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * L;
    
    if ((x < L) && (y < L))
    {
        float N = L - 1;
        float n_x = x - N / 2;
        float n_y = y - N / 2;

        float w_x = exp(-0.5 * (a*n_x / (N / 2)) * (a*n_x / (N / 2)));
        float w_y = exp(-0.5 * (a*n_y / (N / 2)) * (a*n_y / (N / 2)));

        data[idx].x = data[idx].x * w_x * w_y;
        data[idx].y = data[idx].y * w_x * w_y;
    }
}

void Hologram::lowpass(WindowMethod method, float window_param)
{
    CHECK_FOR_ERROR("begin Hologram::lowpass");
    if (this->state != HOLOGRAM_STATE_FOURIER_SHIFTED)
    {
        std::cout << "Low pass filter can only be applied in (shifted) Fourier domain" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    if (method == WINDOW_TUKEY)
    {
        this->window(method, window_param);
        return;
    }
    
    if (method != WINDOW_GAUSSSIAN)
    {
        std::cout << "Windows other than Gaussian are not yet implemented" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    int size = data.getWidth();

    /** OpenCV Host Implementation */
    CuMat win = createGaussianWindow(size, window_param);
    win.convertTo(CV_32FC2);
    umnholo::mult(data, win, data);

    win.destroy();
    //*/

    /** CUDA Implementation /
    float2* img_d = (float2*)data.getCuData();
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(size / (float)blockDim.x), ceil(size / (float)blockDim.y));
    gaussianMult_kernel<<<gridDim, blockDim>>>(img_d, size, window_param);
    //*/

    CHECK_FOR_ERROR("Hologram::lowpass");
    return;
}

void Hologram::removeObjects(ObjectCloud parts)
{
    /*if (this->state != HOLOGRAM_STATE_NORM_MEAND_ZERO)
    {
        std::cout << "Hologram::removeObjects Error: State must be HOLOGRAM_STATE_MEAND_ZERO" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }*/
    
    Reconstruction recon(*this);
    
    int num_parts = parts.getNumObjects();
    Blob3d obj;
    for (int n = 0; n < num_parts; ++n)
    {
        // Calculate initial mean and standard deviation for contrast recovery
        cv::Scalar orig_mean, orig_std;
        cv::meanStdDev(this->data.getMatData(), orig_mean, orig_std);
        
        // Reconstruct to in-focus plane
        obj = parts.getObject(n);
        float focus = obj.getFocusZ() + 1; // +1 is because of mistake in matlab
        float focus_plane = focus * params.plane_stepsize + params.start_plane;
        //std::cout << n << ": removing object at " << obj.getCentroid() << " on plane " << focus_plane << std::endl;

        //cv::Mat reconstructed;
        //this->getData().getReal().getMatData().copyTo(reconstructed);
        //reconstructed.convertTo(reconstructed, CV_8U);
        //cv::imshow("removal", reconstructed);
        //std::cout << "removeObjects: " << std::endl << reconstructed(cv::Rect(0,0,10,10)) << std::endl;
        //cv::waitKey(0);

        for (int it = 0; it < num_removal_iterations; ++it)
        {
            //printf("Part %d reconstruct to plane %f\n", n, focus_plane);
            //printf("plane stepsize = %f, start_plane = %f\n", params.plane_stepsize, params.start_plane);
            recon.reconstructTo(*this, -focus_plane);
            
            //this->getData().getReal().getMatData().copyTo(reconstructed);
            //reconstructed.convertTo(reconstructed, CV_8U);
            //cv::imshow("removal", reconstructed);
            //std::cout << "removeObjects: " << std::endl << reconstructed(cv::Rect(0,0,10,10)) << std::endl;
            //cv::waitKey(0);
            
            CuMat mask = obj.getFocusMask();
            mask.dilateDisk(removal_dilation);
            
            cv::Mat mat_mask = mask.getMatData();
            mat_mask.convertTo(mat_mask, CV_32F);
            //cv::imshow("mask", mat_mask);
            //cv::Rect roi(107, 203, 20, 20);
            //std::cout << "mask at " << roi << std::endl << mat_mask(roi) << std::endl;
            
            this->data.maskMean(mask);
            //this->getData().getReal().getMatData().copyTo(reconstructed);
            //reconstructed.convertTo(reconstructed, CV_8U);
            //cv::imshow("removal", reconstructed);
            //cv::waitKey(0);
            
            // Reconstruct back to original plane
            recon.reconstructTo(*this, focus_plane);
            this->mag();
            //this->getData().getReal().getMatData().copyTo(reconstructed);
            //reconstructed.convertTo(reconstructed, CV_8U);
            //cv::imshow("removal", reconstructed);
            //cv::waitKey(0);
            
            // Contrast recovery
            this->matchMeanStd(orig_mean, orig_std);
            this->mag();
        }
    }
    
    recon.destroy();
    CHECK_FOR_ERROR("Hologram::removeObjects");
    return;
}

void Hologram::findTiles(int size, std::vector<cv::Rect> &rois, int &count)
{
    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();
    
    if ((width < size) || (height < size))
    {
        std::cout << "Hologram::findTiles: Error: Unable to use tile size greater than image size" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    int buffer_width = size / 8;
    int valid_size = size - 2*buffer_width;
    
    // Estimate number of tiles and use that to initialize vector
    rois.resize(0);
    rois.reserve((width/size + 1) * (height/size + 1));
    int temp_count = 0;
    count = 0;
    
    // First tile must be valid
    cv::Rect roi(0, 0, size, size);
    
    bool processed_border_x = false;
    bool processed_border_y = false;
    
    bool continue_tiling = true;
    do
    {
        rois.push_back(roi);
        
        roi.x += valid_size;
        
        if (roi.x + size == width)
            processed_border_x = true;
        if (roi.y + size == height)
            processed_border_y = true;
        
        if (roi.x + size > width)
        {
            if (processed_border_x)
            {
                // Move on to the next row
                roi.x = 0;
                roi.y += valid_size;
                processed_border_x = false;
            }
            else
            {
                roi.x = width - size;
                processed_border_x = true;
            }
        }
        if (roi.y + size > height)
        {
            if (processed_border_y)
            {
                continue_tiling = false;
            }
            else
            {
                roi.y = height - size;
                processed_border_y = true;
            }
        }
        
        count++;
        temp_count++;
        if (temp_count > width) return; // Protection from infinite loop
    }while (continue_tiling);
    
    return;
}

__global__ void backgroundSubtraction_kernel
        (float2* holo_data, float2* bg_data, 
         double alpha, double beta, double LL_GS, 
         double bgmean, double img_mean, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        holo_data[idx].x = (holo_data[idx].x - bg_data[idx].x - img_mean + bgmean - alpha)*beta + LL_GS;
        holo_data[idx].y = 0;

        // limit range from 0-255
        holo_data[idx].x = (holo_data[idx].x < 0)? 0 : holo_data[idx].x;
        holo_data[idx].x = (holo_data[idx].x > 255)? 255 : holo_data[idx].x;
    }
}

void Hologram::subtractBackground(Hologram bg, double alpha, double beta, double LL_GS, double bgmean)
{
    DECLARE_TIMING(subtractBackground);
    START_TIMING(subtractBackground);
    if (this->state != bg.state)
    {
        std::cout << "Hologram::subtractBackground: States of this and bg must be match" << std::endl;
        std::cout << "    states are " << this->state << " and " << bg.state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    if (this->state == HOLOGRAM_STATE_RECONSTRUCTED)
    {
        std::cout << "Hologram::subtractBackground: Hologram cannot be reconstructed" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    size_t size = this->data.getRows() * this->data.getCols();
    if (size != (bg.data.getRows() * bg.data.getCols()))
    {
        std::cout << "Hologram::subtractBackground: Both holograms must be of the same size" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();

    // Subtraction on CPU
    {
        cv::Mat holo_data = this->data.getReal().getMatData();
        cv::Mat bg_data = bg.data.getReal().getMatData();

        holo_data = holo_data - bg_data;
        cv::Scalar ave = cv::mean(holo_data);
        holo_data = ((holo_data - ave - alpha) * beta) + LL_GS;

        cv::threshold(holo_data, holo_data, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(holo_data, holo_data, 0, 0, cv::THRESH_TOZERO);

        cv::Mat planes[] = { holo_data, cv::Mat::zeros(holo_data.size(), holo_data.type()) };
        cv::Mat mat_data_C;
        cv::merge(planes, 2, mat_data_C);
        this->data.setMatData(mat_data_C);
    }

    // GPU
    /*{
        float2* holo_data = (float2*)this->data.getCuData();
        float2* bg_data = (float2*)bg.data.getCuData();

        double holo_mean = cv::mean(this->data.getReal().getMatData())[0];
        if (bgmean < -900.0)
        {
            printf("calculating a new mean\n");
            bgmean = cv::mean(bg.data.getReal().getMatData())[0];
        }
        printf("bgmean = %f\n", bgmean);

        int blockDim = 256;
        int gridDim = ceil(width*height / 256.0);
        backgroundSubtraction_kernel<<<gridDim, blockDim>>>
            (holo_data, bg_data, alpha, beta, LL_GS, bgmean, holo_mean, width*height);
        
        this->data.setCuData(holo_data);
    }*/

    STOP_TIMING(subtractBackground);
    SAVE_TIMING(subtractBackground);
    double time = GET_TIMING(subtractBackground);
    //std::cout << "subtractBackground time = " << time << " ms" << std::endl;
}

void Hologram::divideBackground(Hologram bg)
{
    if (this->state != bg.state)
    {
        std::cout << "Hologram::divideBackground: States of this and bg must be match" << std::endl;
        std::cout << "    states are " << this->state << " and " << bg.state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    if (this->state == HOLOGRAM_STATE_RECONSTRUCTED)
    {
        std::cout << "Hologram::divideBackground: Hologram cannot be reconstructed" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    size_t size = this->data.getRows() * this->data.getCols();
    if (size != (bg.data.getRows() * bg.data.getCols()))
    {
        std::cout << "Hologram::divideBackground: Both holograms must be of the same size" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();

    // Division on CPU
    {
        cv::Mat holo_data = this->data.getReal().getMatData();
        cv::Mat bg_data = bg.data.getReal().getMatData();

        holo_data = (holo_data / bg_data) - 1;

        double minval;
        cv::minMaxIdx(bg_data, &minval, NULL);
        if (minval < 0.01)
        {
            std::cout << "Warning: Background has very low intensities,"
                << "preformance may be affected" << std::endl;
        }

        cv::Mat planes[] = { holo_data, cv::Mat::zeros(holo_data.size(), holo_data.type()) };
        cv::Mat mat_data_C;
        cv::merge(planes, 2, mat_data_C);
        this->data.setMatData(mat_data_C);
    }
    
    this->state = HOLOGRAM_STATE_NORM_MEAND_ZERO;
    this->scale = HOLOGRAM_SCALE_OTHER;
}

void Hologram::removeBackground(Hologram bg)
{
    if (this->state != bg.state)
    {
        std::cout << "Hologram::removeBackground: States of this and bg must be match" << std::endl;
        std::cout << "    states are " << this->state << " and " << bg.state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    if (this->state == HOLOGRAM_STATE_RECONSTRUCTED)
    {
        std::cout << "Hologram::removeBackground: Hologram cannot be reconstructed" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    size_t size = this->data.getRows() * this->data.getCols();
    if (size != (bg.data.getRows() * bg.data.getCols()))
    {
        std::cout << "Hologram::removeBackground: Both holograms must be of the same size" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    size_t width = this->data.getWidth();
    size_t height = this->data.getHeight();
    this->bg_data = bg.data;

    {
        cv::Mat holo_data = this->data.getReal().getMatData();
        cv::Mat bg_data = bg.data.getReal().getMatData();
        cv::Mat sqrt_bg;
        cv::sqrt(bg_data, sqrt_bg);
        cv::Scalar mean_bg_val = cv::mean(sqrt_bg);
        this->sqrt_background_mean = mean_bg_val.val[0];
		
		//  == debug code to find out the min and max of the background
		// Find the min and max values of sqrt_bg
		// double min_val, max_val;
		// cv::minMaxLoc(sqrt_bg, &min_val, &max_val);
		// Print the min and max values
		// std::cout << "Before applying max with epsilon:" << std::endl;
		// std::cout << "Min value: " << min_val << ", Max value: " << max_val << std::endl;
		//  == 
        
        // to deal with zero division, let's increase the minimum of the background
        const float epsilon = 0.1;  // Small value to prevent division by zero
        // Clip small background values to epsilon to avoid division by zero
        sqrt_bg = cv::max(sqrt_bg, epsilon);  // Set any values < epsilon to epsilon

        holo_data = (holo_data - bg_data) / sqrt_bg;
        //holo_data = (holo_data - bg_data) / mean_bg_val.val[0];

        double minval;
        cv::minMaxIdx(bg_data, &minval, NULL);
        if (minval < 0.01)
        {
            std::cout << "Warning: Background has very low intensities,"
                << "preformance may be affected" << std::endl;
        }

        cv::Mat planes[] = { holo_data, cv::Mat::zeros(holo_data.size(), holo_data.type()) };
        cv::Mat mat_data_C;
        cv::merge(planes, 2, mat_data_C);
        this->data.setMatData(mat_data_C);
    }
    
    this->state = HOLOGRAM_STATE_NORM_MEAND_ZERO;
    this->scale = HOLOGRAM_SCALE_OTHER;
}

__global__ void apply_limits_kernel(float2* data, float lower, float upper, size_t size)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        float2 temp = data[idx];
        temp.x = (temp.x < lower)? lower : temp.x;
        temp.x = (temp.x > upper)? upper : temp.x;
        temp.y = (temp.y < lower)? lower : temp.y;
        temp.y = (temp.y > upper)? upper : temp.y;
        
        data[idx] = temp;
    }
}

void Hologram::applySaturation(Hologram& reference)
{
    if ((reference.state != HOLOGRAM_STATE_LOADED) &&
        (reference.state != HOLOGRAM_STATE_NORM_MEAND) &&
        (reference.state != HOLOGRAM_STATE_NORM_MEAND_ZERO) &&
        (reference.state != HOLOGRAM_STATE_NORM_INVERSE) &&
        (reference.state != HOLOGRAM_STATE_RESIDUAL))
    {
        std::cout << "Hologram::applySaturation: Invalid reference state" << std::endl;
        std::cout << "State is " << reference.state << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    if ((this->state != HOLOGRAM_STATE_LOADED) &&
        (this->state != HOLOGRAM_STATE_NORM_MEAND) &&
        (reference.state != HOLOGRAM_STATE_NORM_MEAND_ZERO) &&
        (reference.state != HOLOGRAM_STATE_NORM_INVERSE) &&
        (this->state != HOLOGRAM_STATE_RESIDUAL))
    {
        std::cout << "Hologram::applySaturation: Invalid state" << std::endl;
        std::cout << "State is " << reference.state << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    this->lower_saturation_limit = reference.lower_saturation_limit;
    this->upper_saturation_limit = reference.upper_saturation_limit;
    
    printf("applySaturation: limits are %f and %f\n",
        lower_saturation_limit, upper_saturation_limit);
    
    size_t numel = this->data.getWidth() * this->data.getHeight();
    size_t dim_block = 256;
    size_t dim_grid = ceil(float(numel) / (float)dim_block);
    float2* holo_data_d = (float2*)this->data.getCuData();
    apply_limits_kernel<<<dim_grid, dim_block>>>
        (holo_data_d, lower_saturation_limit, upper_saturation_limit, numel);
    this->data.setCuData((void*)holo_data_d);
    
    CHECK_FOR_ERROR("end Hologram::applySaturation");
}

void Hologram::getMinMax(double* min_val, double* max_val)
{
    cv::Mat planes[2];
    cv::split(this->data.getMatData(), planes);
    
    cv::minMaxLoc(planes[0], min_val, max_val);
    
    return;
}

//============================= ACCESS     ===================================

void Hologram::show()
{
    this->show("Hologram");
}

void Hologram::show(char* name, bool delete_after)
{
    if (this->state == HOLOGRAM_STATE_EMPTY)
    {
        std::cout << "Hologram::show: HOLO_ERROR_MISSING_DATA" << std::endl;
        throw HOLO_ERROR_MISSING_DATA;
    }
    else if (this->state == HOLOGRAM_STATE_LOADED)
    {
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::Mat planes[2];
        cv::split(this->data.getMatData(), planes);
        planes[0].convertTo(planes[0], CV_8U);
        cv::imshow(name, planes[0]);
        cv::waitKey(0);
        if (delete_after)
            cv::destroyAllWindows();
    }
    else if (this->state == HOLOGRAM_STATE_NORM_MEAND_ZERO)
    {
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::Mat planes[2];
        cv::split(this->data.getMatData(), planes);
        cv::normalize(planes[0], planes[0], 0, 255, cv::NORM_MINMAX);
        planes[0].convertTo(planes[0], CV_8U);
        cv::imshow(name, planes[0]);
        cv::waitKey(0);
        if (delete_after)
            cv::destroyAllWindows();
    }
    else if ((this->state == HOLOGRAM_STATE_FOURIER) ||
             (this->state == HOLOGRAM_STATE_FOURIER_SHIFTED))
    {
        // Calculate magnitudd of complex data
        cv::Mat fourier = this->getData().getMatData();
        cv::Mat fourier_planes[2];
        cv::split(fourier, fourier_planes);
        fourier_planes[0].convertTo(fourier_planes[0], CV_32F);
        fourier_planes[1].convertTo(fourier_planes[1], CV_32F);
        //cv::Mat mag(fourier.size(), CV_32F);
        //cv::sqrt(fourier_planes[0].mul(fourier_planes[0]) - fourier_planes[1].mul(fourier_planes[1]), mag);
        cv::magnitude(fourier_planes[0], fourier_planes[1], fourier_planes[0]);

        cv::Mat mag = fourier_planes[0];

        // Use log for better visualization
        mag += cv::Scalar::all(1);
        cv::log(mag, mag);

        cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);

        // Now display the data
        cv::namedWindow("FFT of Hologram", cv::WINDOW_AUTOSIZE);
        cv::imshow("FFT of Hologram", mag);
        cv::waitKey(0);
        if (delete_after)
            cv::destroyAllWindows();
    }
    else
    {
        // std::cout << "Holograms of state " << this->state
        //           << " cannot be displayed" << std::endl;
        
        // std::cout << "Displaying as min/max scale of real" << std::endl;
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::Mat planes[2];
        cv::split(this->data.getMatData(), planes);
        
        double min_val, max_val;
        cv::minMaxLoc(planes[0], &min_val, &max_val);
        // std::cout << "Min = " << min_val << ", Max = " << max_val << std::endl;
        cv::normalize(planes[0], planes[0], 0, 1, cv::NORM_MINMAX);
        cv::imshow(name, planes[0]);
        cv::waitKey(0);
        if (delete_after)
            cv::destroyAllWindows();
    }

    return;
}

void Hologram::write(char* filename)
{
    cv::Mat planes[2];
    cv::split(this->data.getMatData(), planes);
    
    if (this->state != HOLOGRAM_STATE_LOADED)
    {
        // std::cout << "Holograms of state " << this->state
        //           << " cannot be written" << std::endl;
        
        // std::cout << "Writing as min/max scale of real" << std::endl;
        
        double min_val, max_val;
        cv::minMaxLoc(planes[0], &min_val, &max_val);
        // std::cout << "Min = " << min_val << ", Max = " << max_val << std::endl;
        cv::normalize(planes[0], planes[0], 0, 255, cv::NORM_MINMAX);
    }
    
    planes[0].convertTo(planes[0], CV_8U);

    if (planes[0].empty())
    {
        std::cout << "Hologram::write: Error: Hologram is empty" << std::endl;
        throw HOLO_ERROR_INVALID_DATA;
    }

    cv::imwrite(filename, planes[0]);
    
    return;
}

void Hologram::write(char* filename, HologramScale new_scale)
{
    int orig_state = this->state;
    HologramScale orig_scale = this->scale;
    this->state = HOLOGRAM_STATE_LOADED;
    this->scale = HOLOGRAM_SCALE_0_1;

    this->rescale(HOLOGRAM_SCALE_0_255);
    this->write(filename);
    this->rescale(HOLOGRAM_SCALE_0_1);

    this->state = orig_state;
    this->scale = orig_scale;
}

void Hologram::write(char* filename, double min_val, double max_val)
{
    cv::Mat planes[2];
    cv::split(this->data.getMatData(), planes);
    
    assert(max_val > min_val);
    planes[0] = 255 * (planes[0] - min_val) / (max_val - min_val);
    
    planes[0].convertTo(planes[0], CV_8U);

    if (planes[0].empty())
    {
        std::cout << "Hologram::write: Error: Hologram is empty" << std::endl;
        throw HOLO_ERROR_INVALID_DATA;
    }

    cv::imwrite(filename, planes[0]);
    
    return;
}

void Hologram::setReconstructedPlane(float z)
{
    if (this->state != HOLOGRAM_STATE_RECONSTRUCTED)
        previous_state = this->state;
    
    if (z == 0.0)
    {// Hologram has been restored to camera plane
        this->state = previous_state;
    }
    
    reconstructed_plane = z;
    return;
}

void Hologram::getSaturationLimits(float* lower, float* upper)
{
    *lower = lower_saturation_limit;
    *upper = upper_saturation_limit;
}

void Hologram::setSize(size_t width, size_t height)
{
    if (!data.isEmpty())
    {
        this->data.setWidth(width);
        this->data.setHeight(height);
        return;
    }
    
    cv::Mat zero_data = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat planes[] = { zero_data, zero_data };
    cv::Mat mat_data_C;
    cv::merge(planes, 2, mat_data_C);         // Add to the expanded another plane with zeros
    mat_data_C.convertTo(mat_data_C, CV_32FC2);

    data.setMatData(mat_data_C);
    state = HOLOGRAM_STATE_ARBITRARY;
    
    return;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////

CuMat Hologram::createTukeyWindow(int n, double r)
{
    // ratio r should be between 0 and 1
    // Other values will return matrix of ones
    if (r <= 0.0)
    {
        cv::Mat mat = cv::Mat::ones(cv::Size(n, n), CV_32F);
        CuMat result;
        result.setMatData(mat);
        return result;
    }
    else if (r >= 1.0)
    {
        cv::Mat mat = cv::Mat::ones(cv::Size(n, n), CV_32F);
        CuMat result;
        result.setMatData(mat);
        return result;
    }
    else // This is the primary case
    {
        double period = r / 2.0;

        // Replicate linspace behavior
        cv::Mat t(n, 1, CV_32F);
        float tval = 0;
        float interval = 1.0 / (n-1);
        for (int i = 0; i < n; ++i)
        {
            t.at<float>(i, 0) = tval;
            tval += interval;
        }

        // Create the window in three section (defined by tl, th)
        double tl = floor(period * (n - 1)) + 1;
        double th = n - tl + 1;
        cv::Mat w(n, 1, CV_32F);
        for (int i = 0; i < tl; ++i)
        {
            w.at<float>(i, 0) = (1 + cos(M_PI / period * (t.at<float>(i, 0) - period))) / 2.0;
        }
        for (int i = tl; i < th; ++i)
        {
            w.at<float>(i, 0) = 1.0;
        }
        for (int i = th; i < n; ++i)
        {
            w.at<float>(i, 0) = (1 + cos(M_PI / period * (t.at<float>(i, 0) - 1.0 + period))) / 2.0;
        }

        // Convert from a 1D to a 2D window
        cv::Mat wt;
        transpose(w, wt);
        cv::Mat win2d = w * wt;
        CuMat window;
        window.setMatData(win2d);

        return window;
    }
}

CuMat Hologram::createGaussianWindow(int L, float a)
{
    // Exclude error checking present in Matlab version for simplicity
    cv::Mat w = cv::Mat::zeros(cv::Size(1, L), CV_32F);
    float N = L - 1;
    for (int i = 0; i <= N; ++i)
    {
        float n = i - N / 2;
        w.at<float>(i, 0) = exp(-0.5 * pow((a*n / (N / 2)), 2));
    }

    cv::Mat wt;
    transpose(w, wt);
    CuMat win;
    win.setMatData(w * wt);
    return win;
}
