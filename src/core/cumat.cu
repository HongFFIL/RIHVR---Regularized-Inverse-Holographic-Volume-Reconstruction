#include "cumat.h"  // class implemented


#if defined(_MSC_VER)
    #include <float.h>  // Needed for _set_output_format
#endif


using namespace umnholo;

/////////////////////////////// KERNELS //////////////////////////////////////

__global__ void minimization_kernel(float* src1_d, float* src2_d, float* dst_d, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        dst_d[idx] = (src1_d[idx] < src2_d[idx]) ? src1_d[idx] : src2_d[idx];
    }

    return;
}

__global__ void maximization_kernel(float* src1_d, float* src2_d, float* dst_d, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        dst_d[idx] = (src1_d[idx] > src2_d[idx]) ? src1_d[idx] : src2_d[idx];
    }

    return;
}

__global__ void maximization_arg_kernel(
        float* src1_d,
        float* src2_d,
        float* dst_d,
        float argval,
        float* arg_d,
        int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        dst_d[idx] = (src1_d[idx] > src2_d[idx]) ? src1_d[idx] : src2_d[idx];
        arg_d[idx] = (src1_d[idx] > src2_d[idx]) ? arg_d[idx] : argval;
    }

    return;
}

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

CuMat::CuMat()
{
    stored_mat = false;
    stored_cu = false;
    is_allocated = false;
    is_zero = false;
    data_size = 0;
    rows = 0;
    cols = 0;
    depth = 0;
    data_type = 0;
    elem_size = 0;
    allocated_size = 0;
    
    is_allocated_buffer_64_d = false;
    buffer_64_d = NULL;
    
    identifier = -1;

    cudaMemGetInfo(&available_device_mem, NULL);
    //printf("called cudaMemGetInfo (CuMat::CuMat)\n");
    return;
}

void CuMat::destroy()
{
    //CHECK_MEMORY("CuMat:destroy begin");
    if (is_allocated)
    {
        cudaFree(cu_data);
        cudaError err = cudaGetLastError();
        if (err != CUDA_SUCCESS)
        {
            // The invalid device pointer error means the data is not allocated
            if (err != cudaErrorInvalidDevicePointer)
            {
                CUDA_SAFE_CALL(err);
            }
        }
    }
    mat_data.deallocate();

    allocated_size = 0;
    is_allocated = false;
    stored_cu = false;
    stored_mat = false;
    
    if (is_allocated_buffer_64_d)
    {
        cudaError_t err = cudaFree(buffer_64_d);
        if (err == cudaErrorInvalidDevicePointer)
        {
            // No need to do much, just means data already freed
            // Remove the error so won't be caught later
            err = cudaGetLastError();
        }
        buffer_64_d = NULL;
        is_allocated_buffer_64_d = false;
    }

    //CHECK_MEMORY("CuMat:destroy end");
    return;
}

//============================= OPERATORS ====================================

/** Copy constructor to avoid needlessly losing allocated data **/
CuMat CuMat::operator=(CuMat rhs)
{
    CHECK_FOR_ERROR("begin CuMat::=");
    // Plain data
    this->data_size = rhs.data_size;
    this->rows = rhs.rows;
    this->cols = rhs.cols;
    this->depth = rhs.depth;
    this->elem_size = rhs.elem_size;
    this->data_type = rhs.data_type;

    // Handle cases where data is stored differently on each object
    // Each object has 4 possible states: cu only, mat only, both, neither
    if (this->is_allocated)
    {
        if (rhs.is_allocated)
        {
            if (this->allocated_size >= rhs.allocated_size)
            {
                // Keep this size the same
                if (rhs.stored_cu)
                {
                    cudaMemcpy(this->cu_data, rhs.cu_data, rhs.allocated_size, cudaMemcpyDeviceToDevice);
                    this->stored_cu = rhs.stored_cu;
                }
                this->stored_cu = rhs.stored_cu;
            }
            else
            {
                // This can be very dangerous behavior
                // If using constructor to initialize an empty matrix, this 
                // works great. However, if that initialization occurs in a
                // loop, it will cause the source data to be overwritten
                // This can be prevented by always initializing your CuMat
                cudaFree(this->cu_data);
                this->cu_data = rhs.cu_data;
                this->allocated_size = rhs.allocated_size;
                this->is_allocated = rhs.is_allocated;
                this->stored_cu = rhs.stored_cu;
            }
        }
        else
        {
            this->stored_cu = false;
        }
    }
    else
    {
        this->is_allocated = rhs.is_allocated;
        this->allocated_size = rhs.allocated_size;
        this->cu_data = rhs.cu_data;
        this->stored_cu = rhs.stored_cu;
    }

    //this->mat_data.deallocate();
    this->mat_data = rhs.mat_data;
    this->stored_mat = rhs.stored_mat;
    
    if (rhs.is_allocated_buffer_64_d && !this->is_allocated_buffer_64_d)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_64_d, 64));
        CUDA_SAFE_CALL(cudaMemset(buffer_64_d, 0, 64));
        this->is_allocated_buffer_64_d = true;
    }
    else
    {
        this->buffer_64_d = NULL;
        this->is_allocated_buffer_64_d = false;
    }

    CHECK_FOR_ERROR("end CuMat::=");
    return *this;
}

//============================= OPERATIONS ===================================

void CuMat::convertTo(int new_type, bool duplicate)
{
    if (data_type == new_type)
        return;

    int old_channels = 1 + (data_type >> CV_CN_SHIFT);
    int new_channels = 1 + (new_type >> CV_CN_SHIFT);
    this->getMatData();

    if (old_channels == new_channels)
    {
        mat_data.convertTo(mat_data, new_type);
        data_type = new_type;
    }
    else if (new_channels > old_channels)
    {
        if (new_channels != 2) throw HOLO_ERROR_UNKNOWN_TYPE;
        int channel_depth = new_type & CV_MAT_DEPTH_MASK;

        cv::Mat planes[2] = { mat_data, cv::Mat::zeros(mat_data.size(), channel_depth) };
        if (duplicate) planes[1] = mat_data;
        cv::Mat complex;
        cv::merge(planes, new_channels, complex);
        this->setMatData(complex);
    }
    else
    {
        // Do not do anything since behavior here could be unexpected
        throw HOLO_ERROR_UNKNOWN_TYPE;
    }

    return;
}

void CuMat::allocateCuData()
{
    CHECK_FOR_ERROR("begin CuMat::allocateCuData");
    if (data_size < available_device_mem)
    {
        if (!is_allocated || (allocated_size < data_size))
        {
            if (allocated_size > 0 && is_allocated)
            {
                cudaFree(cu_data);
            }
            
            cudaMalloc((void**)&cu_data, data_size);
            allocated_size = data_size;
            available_device_mem -= data_size;

            // Data may now be modified by external functions
            stored_mat = false;
            stored_cu = true;
            is_allocated = true;
        }
    }
    else
    {
        throw HOLO_ERROR_OUT_OF_MEMORY;
    }
    
    CHECK_FOR_ERROR("end CuMat::allocateCuData");
    return;
}

void CuMat::allocateCuData(size_t width, size_t height, size_t depth, size_t elem_size)
{
    this->elem_size = elem_size;
    this->rows = height;
    this->cols = width;
    this->depth = depth;
    this->data_size = width * height * depth * elem_size;

    switch (elem_size)
    {
    case 1:
    {
        data_type = CV_8U;
        break;
    }
    case 4:
    {
        data_type = CV_32F;
        break;
    }
    case 8:
    {
        data_type = CV_32FC2;
        break;
    }
    default:
    {
        data_type = CV_8U;
        break;
    }
    }

    allocateCuData();

    return;
}

__global__ void copyReal_kernel(float* out_d, float2* in_d, int n)
{
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    
    int idx = tid + bid*blockDim.x*blockDim.y;
    if (idx < n)
    {
        out_d[idx] = in_d[idx].x;
    }

    return;
}

void CuMat::makeReal()
{
    CHECK_FOR_ERROR("begin CuMat::makeReal");

    if (this->data_type == CV_32F) return;
    if (this->data_type != CV_32FC2) throw HOLO_ERROR_UNKNOWN_TYPE;

    if (stored_mat)
    {
        cv::Mat complex_data[2];
        cv::split(mat_data, complex_data);
        mat_data = complex_data[0];
        data_type = CV_32F;
        elem_size = mat_data.elemSize();
        return;
    }
    if (stored_cu)
    {
        int num_elements = rows*cols*depth;

        float2* current_data = (float2*)cu_data;
        float* new_data = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&new_data, num_elements*sizeof(float)) );
        CHECK_FOR_ERROR("after cudaMalloc");

        int tile_size = 16;
        dim3 gridDim(rows/tile_size, cols/tile_size, depth);
        dim3 blockDim(tile_size, tile_size, 1);
        copyReal_kernel<<<gridDim, blockDim>>>(new_data, current_data, num_elements);
        CHECK_FOR_ERROR("after copyReal_kernel");
        cudaMemcpy(current_data, new_data, num_elements*sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(new_data);
        CHECK_FOR_ERROR("CuMat::makeReal");

        data_type = CV_32F;
        elem_size = 4;
    }

    CHECK_FOR_ERROR("CuMat::makeReal");
    return;
}

void CuMat::save(char* filename, CuMatFileMode format, bool limit_size)
{
    if (format == FILEMODE_BINARY)
    {
        throw HOLO_ERROR_INVALID_ARGUMENT;
        FILE* fid = fopen(filename, "w");
        if (fid == NULL) throw HOLO_ERROR_BAD_FILENAME;

        // Write a header with data desciptions
        fwrite((void*)&data_size, sizeof(size_t), 1, fid);
        fwrite((void*)&rows, sizeof(size_t), 1, fid);
        fwrite((void*)&cols, sizeof(size_t), 1, fid);
        fwrite((void*)&depth, sizeof(size_t), 1, fid);
        fwrite((void*)&data_type, sizeof(int), 1, fid);
        fwrite((void*)&elem_size, sizeof(size_t), 1, fid);

        size_t numel = rows*cols*depth;
        size_t written = 0;
        if (mat_data.isContinuous())
        {
             size_t w = fwrite((void*)this->mat_data.data, elem_size, numel, fid);
             written += w;
        }
        else
        {
            for (int d = 0; d < depth; ++d)
            {
                for (int r = 0; r < rows; ++r)
                {
                    size_t w = fwrite((void*)this->mat_data.ptr(r, 0, d), elem_size, cols, fid);
                    written += w;
                }
            }
        }
        fclose(fid);

        if (written != numel) throw HOLO_ERROR_MISSING_DATA;
    }
    else if (format == FILEMODE_YML)
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "mat_data" << mat_data;
    }
    else if (format == FILEMODE_ASCII)
    {
        FILE* fid = fopen(filename, "w");
        if (fid == NULL) throw HOLO_ERROR_BAD_FILENAME;

        cv::Mat data;
        if (depth > 1)
        {
            std::cout << "Warning: ascii output not supported for 3D array" << std::endl;
        }
        data = getReal().getMatData();
        
        if (data.type() != CV_32F)
        {
            data.convertTo(data, CV_32F);
        }

        int row_count = data.rows;
        int col_count = data.cols;
        if (limit_size)
        {
			if (data.rows > MAX_SAVED_ROWS) row_count = MAX_SAVED_ROWS;
			if (data.cols > MAX_SAVED_ROWS) col_count = MAX_SAVED_ROWS;
		}

//#if defined(WIN32) || defined(_WIN32)
//        _set_output_format(_TWO_DIGIT_EXPONENT);
//#endif

#if defined(_MSC_VER) && !defined(__CUDACC__)
    _set_output_format(_TWO_DIGIT_EXPONENT);
#endif
        for (int r = 0; r < row_count; ++r)
        {
            for (int c = 0; c < col_count; ++c)
            {
                fprintf(fid, "   %0.7E", data.at<float>(r, c));
            }
            fprintf(fid, "\n");
        }

        fclose(fid);
    }
    else
    {
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    return;
}

void CuMat::load(char* filename, CuMatFileMode format)
{
    if (format == FILEMODE_BINARY)
    {
        throw HOLO_ERROR_INVALID_ARGUMENT;
        FILE* fid = fopen(filename, "r");
        if (fid == NULL) throw HOLO_ERROR_BAD_FILENAME;

        // Read the header information
        fread((void*)&(this->data_size), sizeof(size_t), 1, fid);
        fread((void*)&(this->rows), sizeof(size_t), 1, fid);
        fread((void*)&(this->cols), sizeof(size_t), 1, fid);
        fread((void*)&(this->depth), sizeof(size_t), 1, fid);
        fread((void*)&(this->data_type), sizeof(int), 1, fid);
        fread((void*)&(this->elem_size), sizeof(size_t), 1, fid);

        size_t numel = rows*cols*depth;
        cv::Mat temp;
        if (depth > 1)
        {
            //int dims[3] = { rows, cols, depth };
			int dims[3] = {
							static_cast<int>(rows),
							static_cast<int>(cols),
							static_cast<int>(depth)
							};
            temp = cv::Mat(3, dims, data_type);
        }
        else
        {
            temp = cv::Mat(rows, cols, data_type);
        }
        fread((void*)temp.data, elem_size, numel, fid);

        this->setMatData(temp);
    }
    else if (format == FILEMODE_YML)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        fs["mat_data"] >> mat_data;
        setMatData(mat_data);
    }
    else if (format == FILEMODE_ASCII)
    {
        FILE* fid = fopen(filename, "r");
        if (fid == NULL)
        {
            std::cout << "Unable to find file: " << filename << std::endl;
            throw HOLO_ERROR_BAD_FILENAME;
        }

        char temp_str[MAX_SAVED_ROWS*SAVED_NUM_CHARS];

        // Count columns in file
        int num_char = 0;
        while ((fgetc(fid) != '\n') && !feof(fid)) num_char++;
        int num_cols = num_char / SAVED_NUM_CHARS;
        rewind(fid);
        
        if (num_cols > MAX_SAVED_ROWS)
        {
            std::cout << "Unable to read files with more than " << MAX_SAVED_ROWS << " columns" << std::endl;
            throw HOLO_ERROR_INVALID_FILE;
        }

        // Count rows
        int num_rows = 0;
        while (!feof(fid))
        {
            fgets(temp_str, MAX_SAVED_ROWS*SAVED_NUM_CHARS, fid);
            //double temp;
            //if (num_rows < 200) {
            //    sscanf(temp_str, "   %e", &temp);
                //printf("Row %d: <%s>\n", num_rows, temp_str);
            //}
            num_rows++;
        }
        num_rows--;
        rewind(fid);

        cv::Mat temp_mat(num_rows, num_cols, CV_32F);
        for (int r = 0; r < num_rows; ++r)
        {
            fgets(temp_str, MAX_SAVED_ROWS*SAVED_NUM_CHARS, fid);
            for (int c = 0; c < num_cols; ++c)
            {
                sscanf(temp_str+SAVED_NUM_CHARS*c, "   %e", &(temp_mat.at<float>(r, c)));
            }
        }

        this->setMatData(temp_mat);
        fclose(fid);
    }
    else if (format == FILEMODE_CSV)
    {
        // from http://answers.opencv.org/question/55210/reading-csv-file-in-opencv/
        std::ifstream input_file(filename);
        assert(input_file.is_open());
        std::string current_line;
        std::vector< std::vector<double> > all_data;
        
        // Start reading lines as long as there are lines in the file
        int linenum = 0;
        while (getline(input_file, current_line))
        {
            // Now inside each line we need to seperate the cols
            std::vector<double> values;
            std::stringstream temp(current_line);
            std::string single_value;
            while (getline(temp,single_value,','))
            {
                // convert the string element to a double value
                values.push_back(atof(single_value.c_str()));
            }
            // add the row to the complete data vector
            all_data.push_back(values);
        }
        input_file.close();

        // Now add all the data into a Mat element
        cv::Mat temp_mat = cv::Mat::zeros(all_data.size(), all_data[0].size(), CV_32F);
        // Loop over vectors and add the data
        for (int row = 0; row < all_data.size(); ++row)
        {
            for (int col= 0; col< all_data[0].size(); ++col)
            {
                temp_mat.at<float>(row,col) = all_data[row][col];
            }
        }
        
        this->setMatData(temp_mat);
    }
    else
    {
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    return;
}

__global__ void thresholdToZero_kernel(float* data, float thresh, int size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int width = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;
    int idx = x + y * width + z * width * height;

    data[idx] = (data[idx] < thresh) ? 0.0 : data[idx];
}

__global__ void thresholdBinary_kernel(float* data, float thresh, float set_value, int size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int width = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;
    int idx = x + y * width + z * width * height;

    data[idx] = (data[idx] > thresh) ? set_value : 0.0;
}

__global__ void thresholdBinaryInv_kernel(float* data, float thresh, float set_value, int size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int width = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;
    int idx = x + y * width + z * width * height;

    data[idx] = (data[idx] < thresh) ? set_value : 0.0;
}

/** If device data, use specialized kernel, otherwise use OpenCV function **/
void CuMat::threshold(double thresh, int type, float set_value)
{
    if (this->stored_mat)
    {
        cv::threshold(mat_data, mat_data, thresh, set_value, type);

    }
    else if (this->stored_cu)
    {
        if (this->data_type != CV_32F)
        {
            std::cout << "CuMat::threshold: Error: invalid state" << std::endl;
            std::cout << "Data type is " << this->data_type << " should be " << CV_32F << std::endl;
            throw HOLO_ERROR_INVALID_STATE;
        }

        dim3 blockDim(16, 16, 1);
        dim3 gridDim(ceil(cols / 16), ceil(rows / 16), depth);
        switch (type)
        {
        case cv::THRESH_TOZERO:
        {
            thresholdToZero_kernel<<<gridDim, blockDim>>>((float*)cu_data, thresh, rows*cols*depth);
            this->setCuData(cu_data);
            cudaDeviceSynchronize();
            break;
        }
        case cv::THRESH_BINARY:
        {
            thresholdBinary_kernel<<<gridDim, blockDim>>>((float*)cu_data, thresh, set_value, rows*cols*depth);
            this->setCuData(cu_data);
            cudaDeviceSynchronize();
            break;
        }
        case cv::THRESH_BINARY_INV:
        {
            thresholdBinaryInv_kernel<<<gridDim, blockDim>>>((float*)cu_data, thresh, set_value, rows*cols*depth);
            this->setCuData(cu_data);
            cudaDeviceSynchronize();
            break;
        }
        default:
        {
            std::cout << "CuMat::threshold: Error: invalid argument" << std::endl;
            throw HOLO_ERROR_INVALID_ARGUMENT;
        }
        }
    }
    else throw HOLO_ERROR_INVALID_STATE;

    CHECK_FOR_ERROR("CuMat::threshold");
    return;
}

void CuMat::simulateUint8Cast()
{
    cv::Mat data = this->getMatData();
    if (data.type() != CV_32F)
    {
        std::cout << "CuMat::simulateUint8Cast Error" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    data = data * 255;
    data.convertTo(data, CV_8U);
    data.convertTo(data, CV_32F);
    data = data / 255.0;
    this->setMatData(data);
    
    return;
}

void CuMat::dilateDisk(int radius)
{
    mat_data = getMatData();
    cv::Mat element;
    if (radius == 2)
    {
        int size = 5;
        element = cv::Mat::zeros(size, size, CV_8U);
        element.at<uint8_t>(0, 2) = 1;
        element.at<uint8_t>(1, 1) = 1;
        element.at<uint8_t>(1, 2) = 1;
        element.at<uint8_t>(1, 3) = 1;
        element.at<uint8_t>(2, 0) = 1;
        element.at<uint8_t>(2, 1) = 1;
        element.at<uint8_t>(2, 2) = 1;
        element.at<uint8_t>(2, 3) = 1;
        element.at<uint8_t>(2, 4) = 1;
        element.at<uint8_t>(3, 1) = 1;
        element.at<uint8_t>(3, 2) = 1;
        element.at<uint8_t>(3, 3) = 1;
        element.at<uint8_t>(4, 2) = 1;
    }
    else
    {
        int size = 2*radius -1;
        element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size+2));
        element = element(cv::Rect(0, 1, size, size));
    }

    cv::dilate(mat_data, mat_data, element);
    
    this->setMatData(mat_data);
}

void CuMat::morphOpening(CuMat strel, CuMat buffer)
{
    // OpenCV method:
    ///*
    cv::Mat opening_element = strel.getMatData();
    opening_element.convertTo(opening_element, CV_8U);
    cv::Mat src = this->getMatData();
    src.convertTo(src, CV_8U);
    cv::Mat dst = src.clone();
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, opening_element);
    //dst.convertTo(dst, CV_32F);
    this->setMatData(dst);
    //*/

    // NPPI Method
    /*
    Npp32f* src = (Npp32f*)this->getCuData();
    Npp32s step = cols*4;
    NppiSize size = {cols, rows};
    NppiPoint src_offset = {0, 0};
    Npp32f* dst = (Npp32f*)buffer.getCuData();
    Npp8u* mask = (Npp8u*)strel.getCuData();
    NppiSize mask_size = {strel.cols, strel.rows};
    NppiPoint anchor = {mask_size.width/2, mask_size.height/2};
    NppiBorderType border = NPP_BORDER_REPLICATE;

    NppStatus err = NPP_NO_ERROR;
    err = nppiErodeBorder_32f_C1R(src, step, size, src_offset, 
                                 dst, step, size,
                                 mask, mask_size, anchor, border);
    if (err != NPP_SUCCESS)
    {
        std::cout << "CuMat::morphOpening: Error: nppiErodeBorder returned " << err << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    err = nppiDilateBorder_32f_C1R(dst, step, size, src_offset, 
                                 src, step, size,
                                 mask, mask_size, anchor, border);
    if (err != NPP_SUCCESS)
    {
        std::cout << "CuMat::morphOpening: Error: nppiErodeBorder returned " << err << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }

    this->setCuData(src);
    CHECK_FOR_ERROR("CuMat::morphOpening");
    //*/
}

void CuMat::maskMean(CuMat mask)
{
    mat_data = getMatData();
    cv::Mat mask_data = mask.getMatData();
    
    cv::Scalar ave = mean(mat_data);
    mat_data.setTo(ave, mask_data);
    this->setMatData(mat_data);
}

void CuMat::setToMean()
{
    mat_data = getMatData();
    
    cv::Scalar ave = mean(mat_data);
    mat_data.setTo(ave);
    this->setMatData(mat_data);
}

/*
template<class T>
void CuMat::minMax(T* min, T* max)
{
    // Check that class matches stored data
    if (sizeof(T) != elem_size)
    {
        std::cout << "CuMat::min Error: Requested class must matched stored data" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    NppStatus err1, err2;
    
    // Compute min and max using NPPI functions
    NppiSize roi = {cols, rows*depth};
    Npp8u* buffer;
    int buffer_size_h = 0;
    T* src = this->getCuData();
    int src_step = cols*sizeof(T);
    
    switch (sizeof(T))
    {
        case 8:
            err1 = nppiMinMaxGetBufferHostSize_8u_C1R(roi, &buffer_size_h);
            cudaMalloc((void**)&buffer, buffer_size_h);
            err2 = nppiMinMax_8u_C1R(src, src_step, roi, min, max, buffer);
            break;
        case 16:
            err1 = nppiMinMaxGetBufferHostSize_16s_C1R(roi, &buffer_size_h);
            cudaMalloc((void**)&buffer, buffer_size_h);
            err2 = nppiMinMax_16s_C1R(src, src_step, roi, min, max, buffer);
            break;
        case 32:
            err1 = nppiMinMaxGetBufferHostSize_32f_C1R(roi, &buffer_size_h);
            cudaMalloc((void**)&buffer, buffer_size_h);
            err2 = nppiMinMax_32f_C1R(src, src_step, roi, min, max, buffer);
            break;
        default:
            std::cout << "CuMat::min data of size " << sizeof(T) << " is not supported" << std::endl;
            throw HOLO_ERROR_INVALID_ARGUMENT;
            break;
    }
    
    if ((err1 != NPP_SUCCESS) || (err2 != NPP_SUCCESS))
    {
        std::cout << "CuMat::minMax Error Nppi functions with error codes";
        std::cout << err1 << " and " << err2 << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    
    CHECK_FOR_ERROR("CuMat::minMax");
}
*/

// These implementations are increadibly inefficient. Will need to 
// improve if going to be used many times
__global__ void nnz_kernel(float* data, int* nnz, size_t numel)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
    {
        if (data[idx])
        {
            atomicAdd(nnz, 1);
        }
    }

    return;
}
__global__ void nnz_kernel(float2* data, int* nnz, size_t numel)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
    {
        if ((data[idx].x) || (data[idx].y))
        {
            atomicAdd(nnz, 1);
        }
    }

    return;
}

size_t CuMat::countNonZeros()
{
    if (stored_mat)
    {
        if (data_type == CV_32FC2)
        {
            cv::Mat planes[2];
            cv::split(mat_data, planes);
            int nnz = 0;
            nnz += cv::countNonZero(planes[0]);
            nnz += cv::countNonZero(planes[1]);
            return nnz;
        }
        else
        {
            return cv::countNonZero(mat_data); // This may fail if multi-channel array
        }
    }
    
    int nnz = 0;
    size_t numel = rows*cols*depth;
    size_t block_dim = 256;
    size_t grid_dim = ceil((float)numel / (float)block_dim);
    
    int* nnz_d;
    if (~is_allocated_buffer_64_d)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_64_d, 64));
        is_allocated_buffer_64_d = true;
    }
    nnz_d = (int*)buffer_64_d;
    cudaMemset(nnz_d, 0, 1*sizeof(int));
    
    switch (elem_size)
    {
    case 4:
    {
        float* data_d = (float*)this->getCuData();
        nnz_kernel<<<grid_dim, block_dim>>>(data_d, nnz_d, numel);
        break;
    }
    case 8:
    {
        float2* data_d = (float2*)this->getCuData();
        nnz_kernel<<<grid_dim, block_dim>>>(data_d, nnz_d, numel);
        break;
    }
    default:
    {
        std::cout << "CuMat::countNonZeros: Unable to count data with "
            << "element size " << elem_size << " bytes" << std::endl;
        return 0;
    }
    }
    
    cudaMemcpy(&nnz, nnz_d, 1*sizeof(int), cudaMemcpyDeviceToHost);
    
    
    CHECK_FOR_ERROR("end CuMat::countNonZeros");
    
    return nnz;
}

//============================= ACCESS     ===================================

cv::Mat CuMat::getMatData()
{
    if (!stored_mat && !stored_cu && data_size == 0)
    {
        std::cout << "Attempting to access Mat data of empty CuMat" << std::endl;
        throw HOLO_ERROR_MISSING_DATA;
    }
    
    if (stored_mat)
    {
        //printf("CuMat::getMatData data already stored\n");
        return mat_data;
    }
    
    if (mat_data.data == NULL)
    {
        //CHECK_MEMORY("CuMat:getMatData before allocating");
        if (depth > 1)
        {
            //int dims[3] = { rows, cols, depth };
	        //int dims[3] = { rows, cols, depth };
			int dims[3] = {
							static_cast<int>(rows),
							static_cast<int>(cols),
							static_cast<int>(depth)
							};
            mat_data = cv::Mat::zeros(3, dims, data_type);
        }
        else
        {
            mat_data = cv::Mat::zeros(rows, cols, data_type);
        }
        //printf("CuMat::getMatData created new Mat\n");
    }

    if (stored_cu)
    {
        CUDA_SAFE_CALL( cudaMemcpy(mat_data.data, cu_data, data_size, cudaMemcpyDeviceToHost) );
        stored_mat = true;
        //printf("CuMat::getMatData copied data from cu_data\n");
        //CHECK_MEMORY("CuMat:getMatData after copy");
    }

    CHECK_FOR_ERROR("getMatData");
    return mat_data;
}

void CuMat::setMatData(cv::InputArray input)
{
    CHECK_FOR_ERROR("begin CuMat::setMatData");
    is_zero = false;
    mat_data = input.getMat();
    if (stored_cu)
    {
        //std::cout << "Warning: CuMat had device data stored that was thrown out" << std::endl;
        stored_cu = false;
    }
    stored_mat = true;
    rows = mat_data.rows;
    cols = mat_data.cols;
    data_size = mat_data.total() * mat_data.elemSize();
    data_type = mat_data.type();
    elem_size = mat_data.elemSize();

    if (mat_data.dims == 3)
    {
        depth = mat_data.size[2];
    }
    else
    {
        depth = 1;
    }
    CHECK_FOR_ERROR("end CuMat::setMatData");
    return;
}

void* CuMat::getCuData()
{
    if (!stored_mat && !stored_cu && data_size == 0)
    {
        std::cout << "Attempting to access cu data of empty CuMat" << std::endl;
        throw HOLO_ERROR_MISSING_DATA;
    }
    
    is_zero = false;
    if (stored_cu) return cu_data;

    if (stored_mat)
    {
        if (allocated_size < data_size)
        {
            cudaMemGetInfo(&available_device_mem, NULL);
            //printf("Called cudaMemGetInfo (CuMat::getCuData)\n");
            if (data_size < available_device_mem)
            {
                cudaError err = cudaMalloc((void**)&cu_data, data_size);
                if (err != cudaSuccess) throw HOLO_ERROR_OUT_OF_MEMORY;
                
                allocated_size = data_size;
                uchar* raw = mat_data.data;
                cudaMemcpy(cu_data, raw, data_size, cudaMemcpyHostToDevice);
                available_device_mem -= data_size;

                // Data may now be modified by external functions
                stored_mat = false;
                stored_cu = true;
                is_allocated = true;

                return cu_data;
            }
            else
            {
                throw HOLO_ERROR_OUT_OF_MEMORY;
                return NULL;
            }
        }
        else
        {
            uchar* raw = mat_data.data;
            cudaMemcpy(cu_data, raw, data_size, cudaMemcpyHostToDevice);

            // Data may now be modified by external functions
            stored_mat = false;
            stored_cu = true;
            is_allocated = true;

            return cu_data;
        }
    }
    else if (allocated_size < data_size)
    {
        if (data_size < available_device_mem)
        {
            cudaMalloc((void**)&cu_data, data_size);
            allocated_size = data_size;
            available_device_mem -= data_size;

            // Data may now be modified by external functions
            stored_mat = false;
            stored_cu = true;
            is_allocated = true;

            return cu_data;
        }
        else
        {
            throw HOLO_ERROR_OUT_OF_MEMORY;
            return NULL;
        }
    }

    //CHECK_FOR_ERROR("getCuData");
    return cu_data;
}

void CuMat::setCuData(void* input)
{
    cu_data = input;
    if (stored_mat) stored_mat = false;
    stored_cu = true;
    is_allocated = true;
    is_zero = false;
    return;
}

void CuMat::setCuData(void* input, size_t size)
{
    setCuData(input);
    allocated_size = size;

    return;
}

void CuMat::setCuData(void* input, size_t r, size_t c, int type)
{
    setCuData(input);
    data_type = type;
    
    switch (type)
    {
        case CV_8U:
        {
            elem_size = 1;
            break;
        }
        case CV_32F:
        {
            elem_size = 4;
            break;
        }
        case CV_32FC2:
        {
            elem_size = 8;
            break;
        }
        default:
        {
            std::cout << "CuMat::setCuData: Error: Unknown Type: " 
                << elem_size << std::endl;
            throw HOLO_ERROR_UNKNOWN_TYPE;
        }
    }

    if (depth == 0) depth = 1;
    setRows(r);
    setCols(c);

    allocated_size = data_size;

    return;
}

void CuMat::setCuData(void* input, size_t r, size_t c, size_t d, int type)
{
    setCuData(input, r, c, type);
    setDepth(d);

    allocated_size = data_size;

    return;
}

void CuMat::setRows(size_t r)
{
    rows = r;
    data_size = rows * cols * depth * elem_size;
}


void CuMat::setCols(size_t c)
{
    cols = c;
    data_size = rows * cols * depth * elem_size;
}


void CuMat::setDepth(size_t d)
{
    depth = d;
    data_size = rows * cols * depth * elem_size;
}

void CuMat::setElemSize(size_t es)
{
    elem_size = es;
    data_size = rows * cols * depth * elem_size;
    
    switch (elem_size)
    {
    case 1:
    {
        data_type = CV_8U;
        break;
    }
    case 4:
    {
        data_type = CV_32F;
        break;
    }
    case 8:
    {
        data_type = CV_32FC2;
        break;
    }
    default:
    {
        std::cout << "CuMat::setElemSize Error: Unknown data type" << std::endl;
        std::cout << "Input size is " << es << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
}

void CuMat::setDataType(int type)
{
    data_type = type;
    
    switch (type)
    {
    case CV_8U:
    {
        elem_size = 1;
        break;
    }
    case CV_32F:
    {
        elem_size = 4;
        break;
    }
    case CV_32FC2:
    {
        elem_size = 8;
        break;
    }
    default:
    {
        std::cout << "CuMat::setDataType Error: Unknown data type" << std::endl;
        std::cout << "Input type is " << type << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    data_size = rows * cols * depth * elem_size;
    return;
}

CuMat CuMat::getReal()
{
    //CHECK_MEMORY("CuMat:getReal begin");
    mat_data = this->getMatData();
    if (mat_data.channels() == 1) return *this;
    if (mat_data.channels() != 2) throw HOLO_ERROR_UNKNOWN_TYPE;

    //CHECK_MEMORY("CuMat:getReal before split");
    cv::Mat planes[2];
    //CHECK_MEMORY("CuMat:getReal after create planes");
    cv::split(mat_data, planes);
    //CHECK_MEMORY("CuMat:getReal after split");

    CuMat result;
    result.setMatData(planes[0]);
    
    //CHECK_MEMORY("CuMat:getReal before releases");
    //mat_data.release();
    // Delete the imaginary part
    /*planes[1].release();
    planes[1].release();
    planes[1].release();
    planes[1].release();
    planes[0].release();
    planes[0].release();
    planes[0].release();
    planes[0].release();
    //planes[1].deallocate();*/
    //printf("CuMat::getReal deallocated planes[1]\n");
    //printf("refcount for mat_data = %d, planes[0] = %d, planes[1] = %d\n",...
    //    *(mat_data.refcount), *(planes[0].refcount), *(planes[1].refcount));
    //CHECK_MEMORY("CuMat:getReal end");

    return result;
}

CuMat CuMat::getImag()
{
    if (this->getMatData().channels() == 1) return *this;
    if (this->getMatData().channels() != 2) throw HOLO_ERROR_UNKNOWN_TYPE;

    cv::Mat planes[2];
    cv::split(this->getMatData(), planes);

    CuMat result;
    result.setMatData(planes[1]);

    return result;
}

CuMat CuMat::getMag()
{
    if (this->getMatData().channels() == 1) return *this;
    if (this->getMatData().channels() != 2) throw HOLO_ERROR_UNKNOWN_TYPE;

    cv::Mat planes[2];
    cv::split(this->getMatData(), planes);
    
    cv::magnitude(planes[0], planes[1], planes[0]);

    CuMat result;
    result.setMatData(planes[0]);

    return result;
}

CuMat CuMat::getPhase()
{
    if (this->getMatData().channels() == 1) return *this;
    if (this->getMatData().channels() != 2) throw HOLO_ERROR_UNKNOWN_TYPE;

    cv::Mat planes[2];
    cv::split(this->getMatData(), planes);
    
    cv::phase(planes[0], planes[1], planes[0]);

    CuMat result;
    result.setMatData(planes[0]);

    return result;
}

//============================= INQUIRY    ===================================

bool CuMat::hasNan()
{
    if ((elem_size != 4) && (elem_size != 8))
    {
        std::cout << "CuMat::hasNan: Data must be either float or float2" << std::endl;
        throw HOLO_ERROR_UNKNOWN_TYPE;
    }
    size_t full_size = rows*cols*depth*elem_size;
    float* temp_h = (float*)malloc(full_size);
    float* temp_d = (float*)this->getCuData();
    cudaMemcpy(temp_h, temp_d, full_size, cudaMemcpyDeviceToHost);
    size_t numel = rows*cols*depth*(elem_size / 4);
    for (size_t idx = 0; idx < numel; ++idx)
    {
        if (isnan(temp_h[idx])) return true;
    }
    
    free(temp_h);
    
    return false;
}

/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////


/////////////////////////////// OVERLOADS  ///////////////////////////////////

void umnholo::min(CuMat src1, CuMat src2, CuMat dst)
{
    if (src1.getDataType() != src2.getDataType()) throw HOLO_ERROR_INVALID_ARGUMENT;

    if (src1.isStoredCu() && src2.isStoredCu())
    {
        if (src1.getDataType() != CV_32F) throw HOLO_ERROR_UNKNOWN_TYPE;

        float* src1_d = (float*)src1.getCuData();
        float* src2_d = (float*)src2.getCuData();
        float* dst_d = (float*)dst.getCuData();

        int num_elements = src1.getWidth() * src1.getHeight() * src1.getDepth();

        minimization_kernel<<<1024, num_elements/1024 + 1>>>(src1_d, src2_d, dst_d, num_elements);
    }
    else
    {
        cv::Mat src1_mat = src1.getMatData();
        cv::Mat src2_mat = src2.getMatData();
        cv::Mat dst_mat = dst.getMatData();
        cv::min(src1_mat, src2_mat, dst_mat);

        dst.setMatData(dst_mat);
    }

    return;
}

void umnholo::max(CuMat src1, CuMat src2, CuMat dst)
{
    if (src1.getDataType() != src2.getDataType()) throw HOLO_ERROR_INVALID_ARGUMENT;

    if (src1.isStoredCu() && src2.isStoredCu())
    {
        if (src1.getDataType() != CV_32F) throw HOLO_ERROR_UNKNOWN_TYPE;

        float* src1_d = (float*)src1.getCuData();
        float* src2_d = (float*)src2.getCuData();
        float* dst_d = (float*)dst.getCuData();

        int num_elements = src1.getWidth() * src1.getHeight() * src1.getDepth();

        maximization_kernel<<<1024, num_elements/1024 + 1>>>(src1_d, src2_d, dst_d, num_elements);
    }
    else
    {
        cv::Mat src1_mat = src1.getMatData();
        cv::Mat src2_mat = src2.getMatData();
        cv::Mat dst_mat = dst.getMatData();
        cv::max(src1_mat, src2_mat, dst_mat);

        dst.setMatData(dst_mat);
    }

    return;
}

void umnholo::argmax(CuMat src1, CuMat src2, CuMat dst, float id2, CuMat arg)
{
    CHECK_FOR_ERROR("begin umnholo::argmax");
    if (src1.getDataType() != src2.getDataType()) throw HOLO_ERROR_INVALID_ARGUMENT;
    
    if (src1.getDataType() != CV_32F) throw HOLO_ERROR_UNKNOWN_TYPE;
    
    float* src1_d = (float*)src1.getCuData();
    float* src2_d = (float*)src2.getCuData();
    float* dst_d = (float*)dst.getCuData();
    float* arg_d = (float*)arg.getCuData();
    
    int num_elements = src1.getWidth() * src1.getHeight() * src1.getDepth();
    int threads_per_block = 1024;
    int num_blocks = ceil((float)num_elements / (float)threads_per_block);
    
    maximization_arg_kernel<<<num_blocks, threads_per_block>>>
        (src1_d, src2_d, dst_d, id2, arg_d, num_elements);
    
    dst.setCuData((void*)dst_d);
    arg.setCuData((void*)arg_d);
    
    cv::Mat dst_mat = dst.getMatData();
    cv::Mat arg_mat = arg.getMatData();
    
    CHECK_FOR_ERROR("end umnholo::argmax");
}

__global__ void multiplication_kernel(float2* src1, float2* src2, float2* dst, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx].x = src1[idx].x * src2[idx].x;
        dst[idx].y = src1[idx].y * src2[idx].y;
    }

    return;
}

void umnholo::mult(CuMat &src1, CuMat &src2, CuMat &dst)
{
    if (src1.getDataType() != src2.getDataType()) throw HOLO_ERROR_INVALID_ARGUMENT;

    if (src1.isStoredCu() || src2.isStoredCu())
    {
        if (src1.getDataType() != CV_32FC2) throw HOLO_ERROR_UNKNOWN_TYPE;

        float2* src1_d = (float2*)src1.getCuData();
        float2* src2_d = (float2*)src2.getCuData();
        float2* dst_d = (float2*)dst.getCuData();

        int num_elements = src1.getWidth() * src1.getHeight() * src1.getDepth();

        multiplication_kernel<<<1024, num_elements/1024 + 1>>>(src1_d, src2_d, dst_d, num_elements);
    }
    else
    {
        cv::Mat src1_mat = src1.getMatData();
        cv::Mat src2_mat = src2.getMatData();
        cv::Mat dst_mat = dst.getMatData();
        cv::multiply(src1_mat, src2_mat, dst_mat);

        dst.setMatData(dst_mat);
    }

    return;
}

__global__ void scalar_multiplication_kernel
    (float2* src, float scalar, float2* dst, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx].x = src[idx].x * scalar;
        dst[idx].y = src[idx].y * scalar;
    }

    return;
}

void umnholo::mult(CuMat &src, float scalar, CuMat &dst)
{
    CHECK_FOR_ERROR("begin umnholo::mult scalar");
    if (src.isStoredCu())
    {
        if (src.getDataType() != CV_32FC2) throw HOLO_ERROR_UNKNOWN_TYPE;

        float2* src_d = (float2*)src.getCuData();
        float2* dst_d = (float2*)dst.getCuData();

        int num_elements = src.getWidth() * src.getHeight() * src.getDepth();
        
        size_t block_dim = 1024;
        size_t grid_dim = ceil((double)num_elements / (double)block_dim);
        scalar_multiplication_kernel<<<grid_dim, block_dim>>>
            (src_d, scalar, dst_d, num_elements);
    }
    else
    {
        cv::Mat src_mat = src.getMatData();
        cv::Mat dst_mat = dst.getMatData();
        dst_mat = src_mat * scalar;

        dst.setMatData(dst_mat);
    }

    CHECK_FOR_ERROR("end umnholo::mult scalar");
    return;
}

void umnholo::sum(CuMat &src1, CuMat &src2, CuMat &dst)
{
    if (src1.getDataType() != src2.getDataType())
    {
        std::cout << "cumat sum: Error inputs must be same type" << std::endl;
        std::cout << "  types are " << src1.getDataType() 
                  << " and " << src2.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    cv::Mat src1_mat = src1.getMatData();
    cv::Mat src2_mat = src2.getMatData();
    cv::Mat dst_mat = dst.getMatData();
    cv::add(src1_mat, src2_mat, dst_mat);

    dst.setMatData(dst_mat);
}

__global__ void subtraction_kernel(float2* src1, float2* src2, float2* dst, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx].x = src1[idx].x - src2[idx].x;
        dst[idx].y = src1[idx].y - src2[idx].y;
    }

    return;
}

void umnholo::subtract(CuMat &src1, CuMat &src2, CuMat &dst)
{
    CHECK_FOR_ERROR("begin umnholo::subtract");
    
    if (src1.getDataType() != src2.getDataType())
    {
        std::cout << "cumat subtract: Error inputs must be same type" << std::endl;
        std::cout << "  types are " << src1.getDataType() 
                  << " and " << src2.getDataType() << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (src1.isStoredCu() || src2.isStoredCu())
    {
        if (src1.getDataType() != CV_32FC2) throw HOLO_ERROR_UNKNOWN_TYPE;

        float2* src1_d = (float2*)src1.getCuData();
        float2* src2_d = (float2*)src2.getCuData();
        float2* dst_d = (float2*)dst.getCuData();

        int num_elements = src1.getWidth() * src1.getHeight() * src1.getDepth();
        if (!dst.isAllocated())
        {
            dst.allocateCuData(src1.getWidth(), src1.getHeight(), 
                src1.getDepth(), src1.getElemSize());
            dst_d = (float2*)dst.getCuData();
        }

        size_t block_dim = 1024;
        size_t grid_dim = num_elements/block_dim + 1;
        subtraction_kernel<<<grid_dim, block_dim>>>(src1_d, src2_d, dst_d, num_elements);
        
        dst.setCuData(dst_d);
    }
    else
    {
        cv::Mat src1_mat = src1.getMatData();
        cv::Mat src2_mat = src2.getMatData();
        cv::Mat dst_mat = dst.getMatData();
        cv::subtract(src1_mat, src2_mat, dst_mat);

        dst.setMatData(dst_mat);
    }
    
    CHECK_FOR_ERROR("end umnholo::subtract");
}
