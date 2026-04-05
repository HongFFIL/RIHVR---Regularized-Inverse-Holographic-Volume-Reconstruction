#include "sparse_volume.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

SparseVolume::SparseVolume()
{
    DECLARE_TIMING(SPV_CONSTRUCTOR);
    START_TIMING(SPV_CONSTRUCTOR);
    width = 0;
    height = 0;
    depth = 0;
    //num_voxels = 0;
    num_planes = 0;
    plane_subsampling = 1;
    ignore_subsample_checks = false;
    voxel_size.x = 1;
    voxel_size.y = 1;
    voxel_size.z = 1;
    is_allocated_buffer_64_d = false;
    SAVE_TIMING(SPV_CONSTRUCTOR);
}

void SparseVolume::initialize(Hologram holo, int subsampling)
{
    DECLARE_TIMING(SPV_CONSTRUCTOR);
    START_TIMING(SPV_CONSTRUCTOR);
    this->params = holo.getParams();
    width = holo.getWidth();
    height = holo.getHeight();
    depth = params.num_planes;
    num_planes = params.num_planes;
    plane_subsampling = subsampling;
    //plane_data_h = (float2*)malloc(width*height*sizeof(float2));
    
    coo_idx = (size_t**)malloc(num_planes * sizeof(size_t*));
    coo_value = (float2**)malloc(num_planes * sizeof(float2*));
    plane_nnz = (size_t*)malloc(num_planes * sizeof(size_t));
    plane_allocated = (size_t*)malloc(num_planes * sizeof(size_t));
    
    for (int z = 0; z < num_planes; ++z)
    {
        plane_nnz[z] = 0;
        plane_allocated[z] = 0;
    }
    
    //data.resize(num_planes);
    CHECK_FOR_ERROR("SparseVolume::initialize");
    SAVE_TIMING(SPV_CONSTRUCTOR);
}

void SparseVolume::initialize(SparseVolume volume)
{
    DECLARE_TIMING(SPV_CONSTRUCTOR);
    START_TIMING(SPV_CONSTRUCTOR);
    this->params = volume.getParams();
    width = volume.width;
    height = volume.height;
    depth = params.num_planes;
    num_planes = params.num_planes;
    plane_subsampling = volume.plane_subsampling;
    
    coo_idx = (size_t**)malloc(num_planes * sizeof(size_t*));
    coo_value = (float2**)malloc(num_planes * sizeof(float2*));
    plane_nnz = (size_t*)malloc(num_planes * sizeof(size_t));
    plane_allocated = (size_t*)malloc(num_planes * sizeof(size_t));
    
    for (int z = 0; z < num_planes; ++z)
    {
        plane_nnz[z] = 0;
        plane_allocated[z] = 0;
    }
    
    CHECK_FOR_ERROR("SparseVolume::initialize");
    SAVE_TIMING(SPV_CONSTRUCTOR);
}

void SparseVolume::erase()
{
    for (int z = 0; z < num_planes; ++z)
    {
        if (plane_allocated[z] > 0)
        {
            cudaFree(coo_idx[z]);
            cudaFree(coo_value[z]);
        }
    }
    
    for (int z = 0; z < num_planes; ++z)
    {
        plane_nnz[z] = 0;
        plane_allocated[z] = 0;
    }
    CHECK_FOR_ERROR("end SparseVolume::erase");
}

void SparseVolume::destroy()
{
    CHECK_FOR_ERROR("begin SparseVolume::destroy");
    for (int z = 0; z < num_planes; ++z)
    {
        if (plane_allocated[z] > 0)
        {
            cudaFree(coo_idx[z]);
            cudaFree(coo_value[z]);
        }
    }
    free(coo_idx);
    free(coo_value);
    free(plane_nnz);
    free(plane_allocated);
    //free(plane_data_h);
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
    CHECK_FOR_ERROR("end SparseVolume::destroy");
}

//============================= OPERATORS ====================================

SparseVolume& SparseVolume::operator=(const SparseVolume& other)
{
    CHECK_FOR_ERROR("before SparseVolume::operator=");
    // Clear out LHS data
    //this->data.clear();
    //this->plane_list.clear();
    
    if (this->num_planes != other.num_planes)
    {
        std::cout << "Error in SparseVolume::operator=" << std::endl;
        std::cout << "number of planes must be equal" << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    this->params = other.params;
    //this->data = other.data;
    //this->plane_list = other.plane_list;
    this->width = other.width;
    this->height = other.height;
    this->depth = other.depth;
    //this->num_voxels = other.num_voxels;
    this->num_planes = other.num_planes;
    this->voxel_size = other.voxel_size;
    this->plane_subsampling = other.plane_subsampling;
    
    for (int z = 0; z < num_planes; z+=plane_subsampling)
    {
        if (other.plane_allocated[z] > this->plane_allocated[z])
        {
            size_t new_size = other.plane_allocated[z];
            if (this->plane_allocated[z] > 0)
            {
                //printf("plane %d: allocated = %d, nnz = %d\n", z, plane_allocated[z], plane_nnz[z]);
                CUDA_SAFE_CALL(cudaFree(this->coo_idx[z]));
                CUDA_SAFE_CALL(cudaFree(this->coo_value[z]));
            }
            CUDA_SAFE_CALL(cudaMalloc((void**)&this->coo_idx[z], new_size*sizeof(size_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&this->coo_value[z], new_size*sizeof(float2)));
            this->plane_allocated[z] = new_size;
        }
        
        cudaMemset(this->coo_idx[z], 0, this->plane_allocated[z]*sizeof(size_t));
        cudaMemset(this->coo_value[z], 0, this->plane_allocated[z]*sizeof(size_t));
        
        CUDA_SAFE_CALL(cudaMemcpy(this->coo_idx[z], other.coo_idx[z],
            other.plane_nnz[z]*sizeof(size_t), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(this->coo_value[z], other.coo_value[z],
            other.plane_nnz[z]*sizeof(size_t), cudaMemcpyDeviceToDevice));
        this->plane_nnz[z] = other.plane_nnz[z];
    }
    
    // Buffer memory is shared. This is potentially somewhat dangerous
    this->is_allocated_buffer_64_d = other.is_allocated_buffer_64_d;
    this->buffer_64_d = other.buffer_64_d;
    
    CHECK_FOR_ERROR("after SparseVolume::operator=");
    return *this;
}// =

//============================= OPERATIONS ===================================

void SparseVolume::calcL1NormComplex(double* norm)
{
    //DECLARE_TIMING(SP_VOL_CALC_L1NORM);
    //START_TIMING(SP_VOL_CALC_L1NORM);
    // TODO: Check whethe this can be done faster on GPU
    // For now, believe that conversion woud take too long.
    *norm = 0;
    
    size_t max_nnz = 0;
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        max_nnz = std::max(max_nnz, plane_nnz[zid]);
    }
    size_t* coo_idx_h = (size_t*)malloc(max_nnz*sizeof(size_t));
    float2* coo_value_h = (float2*)malloc(max_nnz*sizeof(float2));
    
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        cudaMemcpy(coo_idx_h, coo_idx[zid], plane_nnz[zid]*sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(coo_value_h, coo_value[zid], plane_nnz[zid]*sizeof(float2), cudaMemcpyDeviceToHost);
        
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t index = coo_idx_h[idx];
            size_t xid = index % width;
            size_t yid = index / width;
            float2 value = coo_value_h[idx];
            // *norm += sqrt(value.x*value.x) + sqrt(value.y*value.y);
            *norm += sqrt(value.x*value.x + value.y*value.y);
        }
    }
    
    // Norm scale should not change with level of subsampling
    // Implicitly using nearest-neighbor interpolation
    *norm *= plane_subsampling;
    
    free(coo_idx_h);
    free(coo_value_h);
    
    //SAVE_TIMING(SP_VOL_CALC_L1NORM);
    return;
}

__global__ void tv_norm_kernel
    (float* out, float2* data, float2* data_prev,
     size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xm = (x>0)? x-1 : Nx-1;
        int ym = (y>0)? y-1 : Ny-1;
        int idx_xm = y*Nx + xm;
        int idx_ym = ym*Nx + x;
        
        // grad is the result of grad(y1)
        float2 grad_x, grad_y, grad_z;
        grad_x.x = data[idx_xm].x - data[idx].x;
        grad_y.x = data[idx_ym].x - data[idx].x; 
        grad_z.x = data_prev[idx].x - data[idx].x;
        grad_x.y = data[idx_xm].y - data[idx].y; 
        grad_y.y = data[idx_ym].y - data[idx].y; 
        grad_z.y = data_prev[idx].y - data[idx].y;
        
        float normalizer = grad_x.x*grad_x.x + grad_x.y*grad_x.y +
                           grad_y.x*grad_y.x + grad_y.y*grad_y.y +
                           grad_z.x*grad_z.x + grad_z.y*grad_z.y;
        out[idx] = sqrt(normalizer);
    }
}

template <unsigned int blockSize>
__global__ void tv_summation_kernel(float* data_d, float* buffer, int size)
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
        sdata[tid] += data_d[i];
        sdata[tid] += data_d[i+blockSize];
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

void SparseVolume::calcTVNorm(double* norm)
{
    CHECK_FOR_ERROR("begin SparseVolume::calcTVNorm");
    // Prep for tv_norm kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    // Prep for summation
    size_t plane_size = width*height;
    dim3 block_dim_sum(512, 1, 1);
    dim3 grid_dim_sum(ceil(plane_size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, grid_dim_sum.x * sizeof(float));
    float* buffer_h = (float*)malloc(grid_dim_sum.x * sizeof(float));
    float* summation_plane_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&summation_plane_d, width*height*2*sizeof(float)));
    *norm = 0.0;
    
    CuMat prev_plane;
    CuMat this_plane;
    this->getPlane(&prev_plane, depth-1);
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        this->getPlane(&this_plane, zid);
        
        float2* this_d = (float2*)this_plane.getCuData();
        float2* prev_d = (float2*)prev_plane.getCuData();
        
        // Calculates the gradient for each voxel
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm before tv_norm_kernel");
        tv_norm_kernel<<<grid_dim, block_dim>>>
            (summation_plane_d, this_d, prev_d, width, height);
        cudaDeviceSynchronize();
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm after tv_norm_kernel");
        
        // Sum the voxels together
        tv_summation_kernel<512><<<grid_dim_sum, block_dim_sum, smemSize>>>
            (summation_plane_d, buffer_d, plane_size);
        cudaDeviceSynchronize();
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm after tv_summation_kernel");
        cudaMemcpy(buffer_h, buffer_d, grid_dim_sum.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_dim_sum.x; ++i)
        {
            *norm += buffer_h[i];
        }
        
        prev_plane = this_plane;
    }
    
    // Norm scale should not change with level of subsampling
    // Implicitly using nearest-neighbor interpolation
    *norm *= plane_subsampling;
    
    prev_plane.destroy();
    this_plane.destroy();
    cudaFree(buffer_d);
    cudaFree(summation_plane_d);
    free(buffer_h);
    CHECK_FOR_ERROR("end SparseVolume::calcTVNorm");
}

__global__ void tv_norm_2d_kernel
    (float* out, float2* data,
     size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xm = (x>0)? x-1 : Nx-1;
        int ym = (y>0)? y-1 : Ny-1;
        int idx_xm = y*Nx + xm;
        int idx_ym = ym*Nx + x;
        
        // grad is the result of grad(y1)
        float2 grad_x, grad_y;
        grad_x.x = data[idx_xm].x - data[idx].x;
        grad_y.x = data[idx_ym].x - data[idx].x;
        grad_x.y = data[idx_xm].y - data[idx].y;
        grad_y.y = data[idx_ym].y - data[idx].y;
        
        float normalizer = grad_x.x*grad_x.x + grad_x.y*grad_x.y +
                           grad_y.x*grad_y.x + grad_y.y*grad_y.y;
        out[idx] = sqrt(normalizer);
    }
}

void SparseVolume::calcTVNorm2d(double* norm)
{
    CHECK_FOR_ERROR("begin SparseVolume::calcTVNorm2d");
    // Prep for tv_norm kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    // Prep for summation
    size_t plane_size = width*height;
    dim3 block_dim_sum(512, 1, 1);
    dim3 grid_dim_sum(ceil(plane_size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, grid_dim_sum.x * sizeof(float));
    float* buffer_h = (float*)malloc(grid_dim_sum.x * sizeof(float));
    float* summation_plane_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&summation_plane_d, width*height*2*sizeof(float)));
    *norm = 0.0;
    
    CuMat this_plane;
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        this->getPlane(&this_plane, zid);
        
        float2* this_d = (float2*)this_plane.getCuData();
        
        // Calculates the gradient for each voxel
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm2d before tv_norm_kernel");
        tv_norm_2d_kernel<<<grid_dim, block_dim>>>
            (summation_plane_d, this_d, width, height);
        cudaDeviceSynchronize();
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm2d after tv_norm_kernel");
        
        // Sum the voxels together
        tv_summation_kernel<512><<<grid_dim_sum, block_dim_sum, smemSize>>>
            (summation_plane_d, buffer_d, plane_size);
        cudaDeviceSynchronize();
        CHECK_FOR_ERROR("SparseVolume::calcTVNorm2d after tv_summation_kernel");
        cudaMemcpy(buffer_h, buffer_d, grid_dim_sum.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < grid_dim_sum.x; ++i)
        {
            *norm += buffer_h[i];
        }
    }
    
    // Norm scale should not change with level of subsampling
    // Implicitly using nearest-neighbor interpolation
    *norm *= plane_subsampling;
    
    this_plane.destroy();
    cudaFree(buffer_d);
    cudaFree(summation_plane_d);
    free(buffer_h);
    CHECK_FOR_ERROR("end SparseVolume::calcTVNorm2d");
}

size_t SparseVolume::countNonZeros()
{
    size_t nnz = 0;
    for (size_t plane = 0; plane < num_planes; plane+=plane_subsampling)
    {
        nnz += plane_nnz[plane];
    }
    
    // Implicitly using nearest-neighbor interpolation
    nnz *= plane_subsampling;
    
    return nnz;
}

void SparseVolume::savePlaneNonZeros(char* prefix)
{
    char filename[FILENAME_MAX];
    sprintf(filename, "%s/%s.csv", params.output_path, prefix);
    FILE* fid = NULL;
    fid = fopen(filename, "w");
    if (fid == NULL)
    {
        std::cout << "Unable to open file: " << filename << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }
    
    for (size_t plane = 0; plane < num_planes; plane+=plane_subsampling)
    {
        fprintf(fid, "%d, %d\n", plane, plane_nnz[plane]);
    }
    
    fclose(fid);
}

double SparseVolume::calcMaxIntensity()
{
    // TODO: Check whether this can be done faster on GPU
    // Is based off calcL1NormComplex so same trends should apply
    double max_intensity = 0.0;
    
    size_t max_nnz = 0;
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        max_nnz = std::max(max_nnz, plane_nnz[zid]);
    }
    size_t* coo_idx_h = (size_t*)malloc(max_nnz*sizeof(size_t));
    float2* coo_value_h = (float2*)malloc(max_nnz*sizeof(float2));
    
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        cudaMemcpy(coo_idx_h, coo_idx[zid], plane_nnz[zid]*sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(coo_value_h, coo_value[zid], plane_nnz[zid]*sizeof(float2), cudaMemcpyDeviceToHost);
        
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t index = coo_idx_h[idx];
            size_t xid = index % width;
            size_t yid = index / width;
            float2 value = coo_value_h[idx];
            double intensity = value.x*value.x + value.y*value.y;
            if (intensity > max_intensity) max_intensity = intensity;
        }
    }
    
    free(coo_idx_h);
    free(coo_value_h);
    
    return max_intensity;
}

void SparseVolume::saveData(char* prefix)
{
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
    
    size_t max_nnz = 0;
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        max_nnz = std::max(max_nnz, plane_nnz[zid]);
    }
    size_t* coo_idx_h = (size_t*)malloc(max_nnz*sizeof(size_t));
    float2* coo_value_h = (float2*)malloc(max_nnz*sizeof(float2));
    
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        cudaMemcpy(coo_idx_h, coo_idx[zid], plane_nnz[zid]*sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(coo_value_h, coo_value[zid], plane_nnz[zid]*sizeof(float2), cudaMemcpyDeviceToHost);
        
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t index = coo_idx_h[idx];
            size_t xid = index % width;
            size_t yid = index / width;
            float2 value = coo_value_h[idx];
            float intensity = value.x*value.x + value.y*value.y;
            fprintf(fid, "%d, %d, %d, %e\n", xid, yid, zid, intensity);
        }
    }
    
    free(coo_idx_h);
    free(coo_value_h);
    
    fclose(fid);
    CHECK_FOR_ERROR("end SparseVolume::saveData");
}

void SparseVolume::loadData(char* filename)
{
    // Check that the volume has been initialized
    if (width == 0)
    {
        std::cout << "SparseVolume was not initialized before loading data\n";
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    // Erase just in case there is some data present already
    this->erase();
    
    FILE* fid = NULL;
    fid = fopen(filename, "r");
    if (fid == NULL)
    {
        std::cout << "Unable to open file: " << filename << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }
    
    char* check_str = "x, y, z, value";
    char read_str[14];
    if (fgets(read_str, 15, fid) == NULL)
        printf("SparseVolume::loadData Unable to read first input line\n");
    std::cout << "Read first line data: " << read_str << std::endl;
    if (strcmp(check_str, read_str))
    {
        std::cout << "SparseVolume::loadData Error:" << std::endl;
        std::cout << "File <" << filename << "> is not correctly formatted\n";
        fclose(fid);
        
        std::cout << "strcmp returns " << strcmp(check_str, read_str) << std::endl;
        
        for (int i = 0; i < 14; ++i)
        {
            printf("char %d: [%c] vs [%c] = %d\n",
                i, check_str[i], read_str[i], check_str[i] == read_str[i]);
        }
        
        throw HOLO_ERROR_INVALID_FILE;
    }
    
    size_t max_nnz = 0;
    while (!feof(fid))
    {
        int xid, yid, zid;
        float intensity;
        fscanf(fid, "%d, %d, %d, %e\n", &xid, &yid, &zid, &intensity);
        
        plane_nnz[zid]++;
        if (plane_nnz[zid] > max_nnz) max_nnz = plane_nnz[zid];
        
        if ((xid > width) || (yid > height) || (zid > depth))
        {
            std::cout << "Error: mismatch between loaded data and expected volume size" << std::endl;
            std::cout << "Expected width = " << width
                << ", height = " << height
                << ", depth = " << depth << std::endl;
            std::cout << "Loaded voxel data at x = " << xid
                << ", y = " << yid << ", z = " << zid << std::endl;
            throw HOLO_ERROR_INVALID_DATA;
        }
    }
    
    // Prep for next read
    rewind(fid);
    fgets(read_str, 15, fid);
    
    size_t* coo_idx_h = (size_t*)malloc(max_nnz*sizeof(size_t));
    float2* coo_value_h = (float2*)malloc(max_nnz*sizeof(float2));
    
    // Begin reading planes one at a time
    for (size_t zid = 0; zid < num_planes; zid+=plane_subsampling)
    {
        // Erase prior data
        for (size_t i = 0; i < max_nnz; ++i)
        {
            coo_idx_h[i] = 0;
            coo_value_h[i].x = 0;
            coo_value_h[i].y = 0;
        }
        
        // Read each corresponding line and store the data
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            int xid, yid, zid2;
            float intensity;
            fscanf(fid, "%d, %d, %d, %e\n", &xid, &yid, &zid2, &intensity);
            
            // Be sure to use original complex representation
            coo_value_h[idx].x = sqrt(intensity);
            coo_value_h[idx].y = 0.0;
            
            int index = yid*width + xid;
            coo_idx_h[idx] = (size_t)index;
        }
        
        if (plane_allocated[zid] < plane_nnz[zid])
        {
            size_t min_bytes = plane_nnz[zid] * sizeof(float2);
            size_t new_size = 512*ceil(1 + (min_bytes / 512.0))/sizeof(float2); // round up a bit
            
            if (plane_allocated[zid] > 0)
            {
                cudaFree(coo_idx[zid]);
                cudaFree(coo_value[zid]);
            }
            CUDA_SAFE_CALL(cudaMalloc((void**)&coo_idx[zid], new_size*sizeof(size_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&coo_value[zid], new_size*sizeof(float2)));
            plane_allocated[zid] = new_size;
        }
        
        cudaMemcpy(coo_idx[zid], coo_idx_h, plane_nnz[zid]*sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(coo_value[zid], coo_value_h, plane_nnz[zid]*sizeof(float2), cudaMemcpyHostToDevice);
    }
    
    free(coo_idx_h);
    free(coo_value_h);
    
    fclose(fid);
}

void SparseVolume::saveProjections(char* prefix, CmbMethod method, bool prefix_as_suffix)
{
    int num_subsample_planes = ceil((double)depth / (double)plane_subsampling);// + 1;
    
    // Initialize the cmb images
    cv::Mat xycmb = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat xzcmb = cv::Mat::zeros(depth,  width, CV_32F);
    cv::Mat yzcmb = cv::Mat::zeros(height, depth, CV_32F);
    cv::Mat sub_xzcmb = cv::Mat::zeros(num_subsample_planes, width, CV_32F);
    cv::Mat sub_yzcmb = cv::Mat::zeros(height, num_subsample_planes, CV_32F);
    if (method == MIN_CMB)
    {
        xycmb = 1e6 * cv::Mat::ones(height, width, CV_32F);
    }
    
    cv::Mat xzvec = cv::Mat::zeros(1, width, CV_32F);
    cv::Mat yzvec = cv::Mat::zeros(height, 1, CV_32F);
    
    CuMat single_plane;
    single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    size_t plane_size = width * height;

    cv::Mat plane;
    
    // Compute global min and max for normalization
    // TODO: This should be a separate method
    double global_min = 0;
    double global_max = 0;
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        this->getPlane(&single_plane, zidx);
        plane = single_plane.getMag().getMatData();
        
        double single_min = 0;
        double single_max = 0;
        minMaxLoc(plane, &single_min, &single_max);
        
        if (zidx == 0) {global_min = single_min; global_max = single_max;}
        global_min = (single_min < global_min)? single_min : global_min;
        global_max = (single_max > global_max)? single_max : global_max;
        this->unGetPlane(&single_plane, zidx);
    }
    
    printf("saveProjections global min = %f, max = %f\n", global_min, global_max);
    
    // Compute the combined images
    // Start at zidx = 1 to exclude noisy first plane
    int sub_idx = 0;
    for (int zidx = 0; zidx < depth; zidx+=plane_subsampling)
    //for (int zidx = 1; zidx < params.num_planes; ++zidx)
    {
        //this->getPlaneInterpolated(&single_plane, zidx);
        this->getPlane(&single_plane, zidx);
        plane = single_plane.getMag().getMatData();
        
        plane = (plane - global_min) / (global_max - global_min);
        
        if (method == MAX_CMB)
        {
            cv::reduce(plane, xzvec, 0, cv::REDUCE_MAX);
            cv::reduce(plane, yzvec, 1, cv::REDUCE_MAX);
            if (zidx > 0) cv::max(plane, xycmb, xycmb);
        }
        else if (method == MIN_CMB)
        {
            cv::reduce(plane, xzvec, 0, cv::REDUCE_MIN);
            cv::reduce(plane, yzvec, 1, cv::REDUCE_MIN);
            if (zidx > 0) cv::min(plane, xycmb, xycmb);
        }
        else throw HOLO_ERROR_INVALID_ARGUMENT;
        
        if (plane_subsampling > 1)
        {
            xzvec.copyTo(sub_xzcmb(cv::Rect(0, sub_idx, width, 1)));
            yzvec.copyTo(sub_yzcmb(cv::Rect(sub_idx, 0, 1, height)));
        }
        else
        {
            xzvec.copyTo(xzcmb(cv::Rect(0, zidx, width, 1)));
            yzvec.copyTo(yzcmb(cv::Rect(zidx, 0, 1, height)));
        }
        
        this->unGetPlane(&single_plane, zidx);
        sub_idx++;
    }
    
    if (plane_subsampling > 1)
    {
        cv::resize(sub_xzcmb, xzcmb, cv::Size(width, depth));
        cv::resize(sub_yzcmb, yzcmb, cv::Size(depth, height));
    }
    
    xycmb.convertTo(xycmb, CV_8U, 255);
    xzcmb.convertTo(xzcmb, CV_8U, 255);
    yzcmb.convertTo(yzcmb, CV_8U, 255);
    sub_xzcmb.convertTo(sub_xzcmb, CV_8U, 255);
    sub_yzcmb.convertTo(sub_yzcmb, CV_8U, 255);
    
    // Save the images
    char filename[FILENAME_MAX];
    sprintf(filename, "%s%s_xy.tif", params.output_path, prefix);
    if (prefix_as_suffix) sprintf(filename, "%sxy_%s.tif", params.output_path, prefix);
    bool result1 = cv::imwrite(filename, xycmb);
    sprintf(filename, "%s%s_xz.tif", params.output_path, prefix);
    if (prefix_as_suffix) sprintf(filename, "%sxz_%s.tif", params.output_path, prefix);
    bool result2 = cv::imwrite(filename, xzcmb);
    sprintf(filename, "%s%s_yz.tif", params.output_path, prefix);
    if (prefix_as_suffix) sprintf(filename, "%syz_%s.tif", params.output_path, prefix);
    bool result3 = cv::imwrite(filename, yzcmb);
    if (!result1 || !result2 || !result3)
    {
        std::cerr << "OpticalField::saveProjections: imwrite failed" << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    
    if (plane_subsampling > 1)
    {
        printf("plane_subsampling = %d, saving subsampled images\n", plane_subsampling);
        sprintf(filename, "%s%s_xz_sub.tif", params.output_path, prefix);
        if (prefix_as_suffix) sprintf(filename, "%sxz_sub_%s.tif", params.output_path, prefix);
        cv::imwrite(filename, sub_xzcmb);
        sprintf(filename, "%s%s_yz_sub.tif", params.output_path, prefix);
        if (prefix_as_suffix) sprintf(filename, "%syz_sub_%s.tif", params.output_path, prefix);
        cv::imwrite(filename, sub_yzcmb);
    }
    
    single_plane.destroy();
    
    return;
}

void SparseVolume::calcProjections(
        CuMat* xycmb,
        CuMat* xzcmb,
        CuMat* yzcmb,
        CmbMethod method)
{
    bool use_xy = xycmb != NULL;
    bool use_xz = xzcmb != NULL;
    bool use_yz = yzcmb != NULL;
    // Computations of all three at once are fairly cheap
    // Will compute all and select data for output at end
    
    int num_subsample_planes = ceil((double)depth / (double)plane_subsampling);
    
    // Initialize the cmb images
    cv::Mat xycmb_mat = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat xzcmb_mat = cv::Mat::zeros(depth, width, CV_32F);
    cv::Mat yzcmb_mat = cv::Mat::zeros(height, depth, CV_32F);
    cv::Mat sub_xzcmb = cv::Mat::zeros(num_subsample_planes, width, CV_32F);
    cv::Mat sub_yzcmb = cv::Mat::zeros(height, num_subsample_planes, CV_32F);
    if (method == MIN_CMB)
    {
        xycmb_mat = 1e6 * cv::Mat::ones(height, width, CV_32F);
    }
    cv::Mat xzvec = cv::Mat::zeros(1, width, CV_32F);
    cv::Mat yzvec = cv::Mat::zeros(height, 1, CV_32F);
    
    CuMat single_plane;
    single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    size_t plane_size = width * height;

    cv::Mat plane;
    
    // Compute the combined images
    // Start at zidx = 1 to exclude noisy first plane
    int sub_idx = 0;
    for (int zidx = 0; zidx < depth; zidx+=plane_subsampling)
    {
        this->getPlane(&single_plane, zidx);
        plane = single_plane.getMag().getMatData();
        
        if (method == MAX_CMB)
        {
            cv::reduce(plane, xzvec, 0, cv::REDUCE_MAX);
            cv::reduce(plane, yzvec, 1, cv::REDUCE_MAX);
            if (zidx > 0) cv::max(plane, xycmb_mat, xycmb_mat);
        }
        else if (method == MIN_CMB)
        {
            cv::reduce(plane, xzvec, 0, cv::REDUCE_MIN);
            cv::reduce(plane, yzvec, 1, cv::REDUCE_MIN);
            if (zidx > 0) cv::min(plane, xycmb_mat, xycmb_mat);
        }
        else throw HOLO_ERROR_INVALID_ARGUMENT;
        
        if (plane_subsampling > 1)
        {
            xzvec.copyTo(sub_xzcmb(cv::Rect(0, sub_idx, width, 1)));
            yzvec.copyTo(sub_yzcmb(cv::Rect(sub_idx, 0, 1, height)));
        }
        else
        {
            xzvec.copyTo(xzcmb_mat(cv::Rect(0, zidx, width, 1)));
            yzvec.copyTo(yzcmb_mat(cv::Rect(zidx, 0, 1, height)));
        }
        
        this->unGetPlane(&single_plane, zidx);
        sub_idx++;
    }
    
    if (plane_subsampling > 1)
    {
        cv::resize(sub_xzcmb, xzcmb_mat, cv::Size(width, depth));
        cv::resize(sub_yzcmb, yzcmb_mat, cv::Size(depth, height));
    }
    
    // Return only the desired data
    if (use_xy) xycmb->setMatData(xycmb_mat);
    if (use_xz) xzcmb->setMatData(xzcmb_mat);
    if (use_yz) yzcmb->setMatData(yzcmb_mat);
    
    single_plane.destroy();
    
    return;
}

//============================= ACCESS     ===================================

double SparseVolume::getSparsity()
{
    size_t nnz = this->countNonZeros();
    size_t volume = width*height*depth;
    
    return (double)nnz / (double)volume;
}

size_t SparseVolume::getPlaneNnz(size_t plane_idx)
{
    if (plane_idx > num_planes)
    {
        std::cout << "SparseVolume::getPlaneNnz: Plane out of bounds" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (plane_idx % plane_subsampling == 0)
    {
        return plane_nnz[plane_idx];
    }
    
    // else use nearest neighbor
    int remain = plane_idx % plane_subsampling;
    plane_idx = plane_idx - remain;
    
    return plane_nnz[plane_idx];
}

__global__ void get_nonzeros_kernel
        (float2* out_data, size_t* coo_idx, float2* coo_value, size_t nnz)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz)
    {
        size_t i = coo_idx[idx];
        out_data[i] = coo_value[idx];
        //out_data[idx].x = i;//16.0;
        //out_data[idx].y = 3.14159;
    }
}

void SparseVolume::getPlane(CuMat* plane, size_t plane_idx)
{
    CHECK_FOR_ERROR("before SparseVolume::getPlane");
    DECLARE_TIMING(SP_VOL_GET_PLANE);
    START_TIMING(SP_VOL_GET_PLANE);
    
    if (plane_idx >= num_planes)
    {
        std::cout << "SparseVolume::getPlane: Plane out of bounds" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (plane_idx % plane_subsampling != 0)
    {
        // Plane is not one of the one designated for subsampling access
        std::cout << "SparseVolume::getPlane: Attempt to access plane "
            << plane_idx << " which is not designated for subsampling" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    // Check before getCuData or will be false
    bool plane_is_zero = plane->isZero();
    
    //DECLARE_TIMING(SP_VOL_GET_PLANE_ALLOCATE);
    //START_TIMING(SP_VOL_GET_PLANE_ALLOCATE);
    // Wipe both host and device memories
    size_t size = width*height*sizeof(float2);
    plane->allocateCuData(width, height, 1, sizeof(float2));
    float2* plane_data_d = (float2*)plane->getCuData();
    //SAVE_TIMING(SP_VOL_GET_PLANE_ALLOCATE);
    //plane_data_h = (float2*)malloc(width*height*sizeof(float2));
    if (!plane_is_zero)
    {
        DECLARE_TIMING(SP_VOL_GET_PLANE_ZERO);
        START_TIMING(SP_VOL_GET_PLANE_ZERO);
        cudaMemset(plane_data_d, 0, size);
        SAVE_TIMING(SP_VOL_GET_PLANE_ZERO);
    }
    //DECLARE_TIMING(SP_VOL_GET_PLANE_COPY_ZERO);
    //START_TIMING(SP_VOL_GET_PLANE_COPY_ZERO);
    //cudaMemcpy(plane_data_h, plane_data_d, size, cudaMemcpyDeviceToHost);
    //SAVE_TIMING(SP_VOL_GET_PLANE_COPY_ZERO);
    
    if (plane_nnz[plane_idx] == 0)
    {
        DECLARE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_C);
        START_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_C);
        plane->setCuData(plane_data_d);
        SAVE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_C);
        plane->identifier = plane_idx;
        return;
    }
    
    /*size_t* temp_coo_idx_h = (size_t*)malloc(plane_nnz[plane_idx]*sizeof(size_t));
    cudaMemcpy(temp_coo_idx_h, coo_idx[plane_idx], plane_nnz[plane_idx]*sizeof(size_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < plane_nnz[plane_idx]; ++i)
    {
        std::cout << "nnz idx " << i << ": coo_idx = " << temp_coo_idx_h[i] << std::endl;
    }
    free(temp_coo_idx_h);*/
    
    DECLARE_TIMING(SP_VOL_GET_PLANE_GPU);
    START_TIMING(SP_VOL_GET_PLANE_GPU);
    CHECK_FOR_ERROR("SparseVolume::getPlane before kernel");
    //std::cout << "plane_allocated[" << plane_idx << "] = " << plane_allocated[plane_idx] << std::endl;
    size_t block_dim = 256;
    size_t grid_dim = ceil((double)plane_nnz[plane_idx]/(double)block_dim);
    get_nonzeros_kernel<<<grid_dim, block_dim>>>
        (plane_data_d, coo_idx[plane_idx], coo_value[plane_idx], plane_nnz[plane_idx]);
    cudaDeviceSynchronize();
    CHECK_FOR_ERROR("SparseVolume::getPlane after kernel");
    SAVE_TIMING(SP_VOL_GET_PLANE_GPU);
    
    //DECLARE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_A);
    //START_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_A);
    //cudaMemcpy(plane_data_d, plane_data_h, size, cudaMemcpyHostToDevice);
    //SAVE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_A);
    //DECLARE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_B);
    //START_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_B);
    plane->setCuData(plane_data_d);
    //SAVE_TIMING(SP_VOL_GET_PLANE_CUMEMCPY_B);
    
    /*if (plane_idx == 0)
    {
        float2* plane_data_h = (float2*)malloc(width*height*sizeof(float2));
        cudaMemcpy(plane_data_h, plane_data_d, size, cudaMemcpyDeviceToHost);
        printf("SparseVolume::getPlane: plane 0:\n");
        bool found_nan = false;
        int num_nan = 0;
        for (int i = 0; i < width*height; ++i)
        {
            if (isnan(plane_data_h[i].x) || isnan(plane_data_h[i].y))
            {
                found_nan = true;
                num_nan++;
            }
        }
        printf("found_nan = %d, count = %d\n", found_nan, num_nan);
        
        std::cout << "data:" << std::endl;
        std::cout << plane->getMatData()(cv::Rect(0,0,4,4)) << std::endl;
        free(plane_data_h);
    }//*/
    
    //free(plane_data_h);
    
    plane->identifier = plane_idx;
    
    SAVE_TIMING(SP_VOL_GET_PLANE);
    CHECK_FOR_ERROR("after SparseVolume::getPlane");
    return;
}

__global__ void unget_nonzeros_kernel
        (float2* out_data, size_t* coo_idx, float2* coo_value, size_t nnz)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz)
    {
        size_t i = coo_idx[idx];
        // Use of - is only difference from getPlane
        out_data[i].x -= coo_value[idx].x;
        out_data[i].y -= coo_value[idx].y;
    }
}

void SparseVolume::unGetPlane(CuMat* plane, size_t plane_idx)
{
    CHECK_FOR_ERROR("before SparseVolume::unGetPlane");
    DECLARE_TIMING(SP_VOL_UNGET_PLANE);
    START_TIMING(SP_VOL_UNGET_PLANE);
    
    if(plane->identifier != plane_idx)
    {
        printf("Warning! Attempting to unGet plane %d but previous plane was %d\n",
            plane_idx, plane->identifier);
    }
    assert(plane->identifier == plane_idx);
    plane->identifier = -1; // barrier agains successive calls
    
    if (plane_idx >= num_planes)
    {
        std::cout << "SparseVolume::unGetPlane: Plane out of bounds" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (plane_idx % plane_subsampling != 0)
    {
        // Plane is not one of the one designated for subsampling access
        std::cout << "SparseVolume::unGetPlane: Attempt to access plane "
            << plane_idx << " which is not designated for subsampling" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    size_t size = width*height*sizeof(float2);
    plane->allocateCuData(width, height, 1, sizeof(float2));
    float2* plane_data_d = (float2*)plane->getCuData();
    //cudaMemset(plane_data_d, 0, size);
    
    if (plane_nnz[plane_idx] == 0)
    {
        return;
    }
    
    CHECK_FOR_ERROR("SparseVolume::unGetPlane before kernel");
    size_t block_dim = 256;
    size_t grid_dim = ceil((double)plane_nnz[plane_idx]/(double)block_dim);
    unget_nonzeros_kernel<<<grid_dim, block_dim>>>
        (plane_data_d, coo_idx[plane_idx], coo_value[plane_idx], plane_nnz[plane_idx]);
    cudaDeviceSynchronize();
    CHECK_FOR_ERROR("SparseVolume::unGetPlane after kernel");
    
    plane->setCuData(plane_data_d);
    
    plane->declareZero();
    
    SAVE_TIMING(SP_VOL_UNGET_PLANE);
    CHECK_FOR_ERROR("after SparseVolume::unGetPlane");
    return;
}

__global__ void set_nonzeros_kernel
        (size_t* coo_idx, float2* coo_value, float2* in_data, int* nnz, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if ((in_data[idx].x) || (in_data[idx].y))
        {
            int old_nnz = atomicAdd(nnz, 1);
            //(old_nnz < 1000)? coo_idx[old_nnz] = 1 : coo_idx[0] = idx;
            //coo_idx[0] = 1;
            //coo_idx[10] = 1;
            coo_idx[old_nnz] = idx;
            coo_value[old_nnz] = in_data[idx];
            //int temp_idx = idx;
            //printf("set old_nnz = %d, idx = %d\n", old_nnz, temp_idx);
        }
    }
}

void SparseVolume::setPlane(CuMat plane, size_t plane_idx)
{
    CHECK_FOR_ERROR("before SparseVolume::setPlane");
    DECLARE_TIMING(SP_VOL_SET_PLANE);
    START_TIMING(SP_VOL_SET_PLANE);
    
    if (!ignore_subsample_checks &&(plane_idx % plane_subsampling != 0))
    {
        // Plane is not one of the one designated for subsampling access
        std::cout << "SparseVolume::setPlane: Attempt to access plane "
            << plane_idx << " which is not designated for subsampling" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    /*cusparseStatus_t status;
    status = cusparseCdense2csr(
        sparse_handle,
        height, depth,
        sparse_descriptor,
        lda, nnz_per_row,
        csr_value_d, csr_row_d, csr_col_d
    );
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "Error in SparseVolume::setPlane" << std::endl
            << "cusparseCdense2csr returned: " << status << std::endl;
        throw HOLO_ERROR_UNKNOWN_ERROR;
    }*/
    
    plane.allocateCuData(width, height, 1, sizeof(float2));
    float2* plane_data_d = (float2*)plane.getCuData();
    //plane_data_h = (float2*)malloc(width*height*sizeof(float2));
    size_t size = width*height*sizeof(float2);
    //cudaMemcpy(plane_data_h, plane_data_d, size, cudaMemcpyDeviceToHost);
    
    CHECK_FOR_ERROR("SparseVolume::setPlane before countNonZeros");
    DECLARE_TIMING(SP_VOL_SET_PLANE_COUNT_NNZ);
    START_TIMING(SP_VOL_SET_PLANE_COUNT_NNZ);
    size_t nnz = plane.countNonZeros();
    SAVE_TIMING(SP_VOL_SET_PLANE_COUNT_NNZ);
    //std::cout << "plane_allocated[" << plane_idx << "] = " << nnz << std::endl;
    if (plane_allocated[plane_idx] < nnz)
    {
        DECLARE_TIMING(SP_VOL_SET_PLANE_ALLOCATIONS);
        START_TIMING(SP_VOL_SET_PLANE_ALLOCATIONS);
        //printf("Allocating device data\n");
        CHECK_FOR_ERROR("SparseVolume::setPlane before reallocations");
        size_t min_bytes = nnz * sizeof(float2);
        size_t new_size = 512*ceil(1 + (min_bytes / 512.0))/sizeof(float2); // round up a bit
        
        if (plane_allocated[plane_idx] > 0)
        {
            cudaFree(coo_idx[plane_idx]);
            cudaFree(coo_value[plane_idx]);
        }
        CHECK_FOR_ERROR("SparseVolume::setPlane after frees");
        CUDA_SAFE_CALL(cudaMalloc((void**)&coo_idx[plane_idx], new_size*sizeof(size_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&coo_value[plane_idx], new_size*sizeof(float2)));
        //std::cout << "allocated room for " << new_size << " elements" << std::endl;
        plane_allocated[plane_idx] = new_size;
        CHECK_FOR_ERROR("SparseVolume::setPlane after reallocations");
        
        //std::cout << "test if space was actually allocated" << std::endl;
        //size_t temp_idx_h[100];
        //CUDA_SAFE_CALL(cudaMemcpy(coo_idx[plane_idx], temp_idx_h, 100*sizeof(size_t), cudaMemcpyHostToDevice));
        //CHECK_FOR_ERROR("allocation test");
        //std::cout << "test was successful" << std::endl;
        SAVE_TIMING(SP_VOL_SET_PLANE_ALLOCATIONS);
    }
    //std::cout << "plane_allocated[" << plane_idx << "] = " << nnz << std::endl;
    
    DECLARE_TIMING(SP_VOL_SET_PLANE_TEMP_MALLOC);
    START_TIMING(SP_VOL_SET_PLANE_TEMP_MALLOC);
    int* nnz_d;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&nnz_d, 1*sizeof(int)));
    if (!is_allocated_buffer_64_d)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_64_d, 8*sizeof(float)));
        is_allocated_buffer_64_d = true;
    }
    nnz_d = (int*)buffer_64_d;
    SAVE_TIMING(SP_VOL_SET_PLANE_TEMP_MALLOC);
    DECLARE_TIMING(SP_VOL_SET_PLANE_TEMP_MEMSET);
    START_TIMING(SP_VOL_SET_PLANE_TEMP_MEMSET);
    CUDA_SAFE_CALL(cudaMemset(nnz_d, 0, 1*sizeof(int)));
    SAVE_TIMING(SP_VOL_SET_PLANE_TEMP_MEMSET);
    
    //std::cout << "SparseVolume::setPlane: nnz = " << nnz << std::endl;
    CHECK_FOR_ERROR("SparseVolume::setPlane before kernel");
    DECLARE_TIMING(SP_VOL_SET_PLANE_KERNEL);
    START_TIMING(SP_VOL_SET_PLANE_KERNEL);
    size_t block_dim = 256;
    size_t grid_dim = ceil((double)(width*height)/(double)block_dim);
    //std::cout << "block_dim = " << block_dim << ", grid_dim = " << grid_dim << std::endl;
    set_nonzeros_kernel<<<grid_dim, block_dim>>>
        (coo_idx[plane_idx], coo_value[plane_idx], plane_data_d, nnz_d, width*height);
    SAVE_TIMING(SP_VOL_SET_PLANE_KERNEL);
    CHECK_FOR_ERROR("SparseVolume::setPlane after kernel");
    
    DECLARE_TIMING(SP_VOL_SET_PLANE_TEMP_MEMCPY);
    START_TIMING(SP_VOL_SET_PLANE_TEMP_MEMCPY);
    int temp_nnz_h;
    cudaMemcpy(&temp_nnz_h, nnz_d, 1*sizeof(int), cudaMemcpyDeviceToHost);
    plane_nnz[plane_idx] = nnz;
    //std::cout << "gpu nnz = " << temp_nnz_h << std::endl;
    CHECK_FOR_ERROR("SparseVolume::setPlane after copy to temp");
    SAVE_TIMING(SP_VOL_SET_PLANE_TEMP_MEMCPY);
    
    // TODO: Get rid of this magic number, should be based on method/GPU memory
    DECLARE_TIMING(SP_VOL_SET_PLANE_FINAL_COUNT);
    START_TIMING(SP_VOL_SET_PLANE_FINAL_COUNT);
    if (this->countNonZeros() > 1024*1024*500)
    {
        std::cout << "Sparse Volume contains too much data!" << std::endl
            << "Attempting to set plane " << plane_idx << std::endl
            << "Plane now has " << this->getPlaneNnz(plane_idx) << " non-zeros" << std::endl
            << "Sparsity is " << this->getSparsity() << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    SAVE_TIMING(SP_VOL_SET_PLANE_FINAL_COUNT);
    
    //size_t num_vox = data[plane_idx].size();
    //double percent = 100*(double)num_vox / (width*height);
    //if (plane_idx == 255) printf("Plane %d has %d filled voxels (%f%%)\n", plane_idx, num_vox, percent);
    
    DECLARE_TIMING(SP_VOL_SET_PLANE_END);
    START_TIMING(SP_VOL_SET_PLANE_END);
    //free(plane_data_h);
    plane.identifier = plane_idx;
    
    CHECK_FOR_ERROR("after SparseVolume::setPlane");
    SAVE_TIMING(SP_VOL_SET_PLANE_END);
    SAVE_TIMING(SP_VOL_SET_PLANE);
    return;
}

__global__ void get_interpolated_kernel
        (float2* out_data, double ratio, float2* above, float2* below,
         size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        out_data[idx].x = below[idx].x + ratio*(above[idx].x - below[idx].x);
        out_data[idx].y = below[idx].y + ratio*(above[idx].y - below[idx].y);
    }
}

void SparseVolume::getPlaneInterpolated(CuMat* plane, size_t plane_idx)
{
    CHECK_FOR_ERROR("before SparseVolume::getPlane");
    DECLARE_TIMING(SP_VOL_GET_PLANE_INTERPOLATED);
    START_TIMING(SP_VOL_GET_PLANE_INTERPOLATED);
    
    if (plane_idx > num_planes)
    {
        std::cout << "SparseVolume::getPlane: Plane out of bounds" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    if (plane_idx % plane_subsampling == 0)
    {
        // No need to interpolate
        getPlane(plane, plane_idx);
        return;
    }
    
    // Wipe both host and device memories
    size_t size = width*height*sizeof(float2);
    plane->allocateCuData(width, height, 1, sizeof(float2));
    float2* plane_data_d = (float2*)plane->getCuData();
    cudaMemset(plane_data_d, 0, size);
    
    size_t plane_idx_below = plane_idx - (plane_idx % plane_subsampling);
    size_t plane_idx_above = plane_idx_below + plane_subsampling;
    double ratio = (double)(plane_idx - plane_idx_below) / plane_subsampling;
    
    // Shortcut if both planes are empty (rare)
    if ((plane_nnz[plane_idx_below] == 0) &&
        (plane_nnz[plane_idx_above] == 0))
    {
        plane->setCuData(plane_data_d);
        return;
    }
    
    CuMat plane_above, plane_below;
    //plane_above.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    plane_above.allocateCuData(width, height, 1, sizeof(float2));
    plane_below.allocateCuData(width, height, 1, sizeof(float2));
    
    // Get the data for the above and below planes
    if (plane_idx_above < num_planes) {
        getPlane(&plane_above, plane_idx_above);
    }
    else {
        plane_above.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    }
    if (plane_idx_below >= 0) {
        getPlane(&plane_below, plane_idx_below);
    }
    else {
        plane_below.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    }
    float2* plane_above_d = (float2*)plane_above.getCuData();
    float2* plane_below_d = (float2*)plane_below.getCuData();
    
    CHECK_FOR_ERROR("SparseVolume::getPlaneInterpolated before kernel");
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil((double)width / (double)block_dim.x),
                  ceil((double)height / (double)block_dim.y));
    get_interpolated_kernel<<<grid_dim, block_dim>>>
        (plane_data_d, ratio, plane_above_d, plane_below_d, width, height);
    cudaDeviceSynchronize();
    CHECK_FOR_ERROR("SparseVolume::getPlaneInterpolated after kernel");
    
    plane->setCuData(plane_data_d);
    
    plane_above.destroy();
    plane_below.destroy();
    
    SAVE_TIMING(SP_VOL_GET_PLANE_INTERPOLATED);
    CHECK_FOR_ERROR("after SparseVolume::getPlane");
    return;
}

void SparseVolume::setPlaneSubsampling(int new_subsampling)
{
    int old_subsampling = plane_subsampling;
    
    if ((old_subsampling != 1) && (old_subsampling != new_subsampling))
    {
        // Need to interpolate from previously subsampled planes
        
        assert(depth % new_subsampling == 0);
        
        CuMat plane;
        plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
        ignore_subsample_checks = true;
        
        for (int zid = 0; zid < params.num_planes; zid+= new_subsampling)
        {
            if (zid % old_subsampling != 0)
            {
                // Otherwise, already set and no need to interpolate
                getPlaneInterpolated(&plane, zid);
                setPlane(plane, zid);
            }
        }
        
        ignore_subsample_checks = false;
        plane.destroy();
    } // else nothing needs to be changed
    
    plane_subsampling = new_subsampling;
    return;
}

void SparseVolume::getHostData(
        size_t*** coo_idx_h,
        float2*** coo_value_h,
        size_t** plane_nnz_h,
        size_t** plane_allocated_h)
{
    CHECK_FOR_ERROR("begin SparseVolume::getHostData");
    
    // Shortcuts to avoid repetitive NULL checks
    bool use_coo_idx = coo_idx_h != NULL;
    bool use_coo_value = coo_value_h != NULL;
    bool use_plane_nnz = plane_nnz_h != NULL;
    bool use_plane_allocated = plane_allocated_h != NULL;
    
    // Allocate arrays as needed
    if (use_coo_idx)
        *coo_idx_h = (size_t**)malloc(num_planes * sizeof(size_t*));
    if (use_coo_value)
        *coo_value_h = (float2**)malloc(num_planes * sizeof(float2*));
    if (use_plane_nnz)
        *plane_nnz_h = (size_t*)malloc(num_planes * sizeof(size_t));
    if (use_plane_allocated)
        *plane_allocated_h = (size_t*)malloc(num_planes * sizeof(size_t));
    
    // Copy simple data over
    for (int z = 0; z < num_planes; ++z)
    {
        if (use_plane_nnz) (*plane_nnz_h)[z] = plane_nnz[z];
        if (use_plane_allocated) (*plane_allocated_h)[z] = plane_allocated[z];
    }
    
    // Copy index data over
    if (use_coo_idx)
    {
        for (int z = 0; z < num_planes; ++z)
        {
            (*coo_idx_h)[z] = (size_t*)malloc(plane_allocated[z]*sizeof(size_t));
            CUDA_SAFE_CALL(cudaMemcpy((*coo_idx_h)[z], coo_idx[z],
                plane_allocated[z]*sizeof(size_t), cudaMemcpyDeviceToHost));
        }
    }
    
    // Copy value data over
    if (use_coo_value)
    {
        for (int z = 0; z < num_planes; ++z)
        {
            (*coo_value_h)[z] = (float2*)malloc(plane_allocated[z]*sizeof(float2));
            CUDA_SAFE_CALL(cudaMemcpy((*coo_value_h)[z], coo_value[z],
                plane_allocated[z]*sizeof(float2), cudaMemcpyDeviceToHost));
        }
    }
    
    CHECK_FOR_ERROR("end SparseVolume::getHostData");
}

void SparseVolume::setHostData(
        size_t*** coo_idx_h,
        float2*** coo_value_h,
        size_t** plane_nnz_h,
        size_t** plane_allocated_h)
{
    CHECK_FOR_ERROR("begin SparseVolume::setHostData");
    
    // Shortcuts to avoid repetitive NULL checks
    bool use_coo_idx = coo_idx_h != NULL;
    bool use_coo_value = coo_value_h != NULL;
    bool use_plane_nnz = plane_nnz_h != NULL;
    bool use_plane_allocated = plane_allocated_h != NULL;
    
    // Copy simple data over
    for (int z = 0; z < num_planes; ++z)
    {
        if (use_plane_nnz) plane_nnz[z] = (*plane_nnz_h)[z];
        if (use_plane_allocated) plane_allocated[z] = (*plane_allocated_h)[z];
    }
    
    // Copy index data over
    if (use_coo_idx)
    {
        for (int z = 0; z < num_planes; ++z)
        {
            CUDA_SAFE_CALL(cudaMemcpy(coo_idx[z], (*coo_idx_h)[z],
                plane_allocated[z]*sizeof(size_t), cudaMemcpyHostToDevice));
        }
    }
    
    // Copy value data over
    if (use_coo_value)
    {
        for (int z = 0; z < num_planes; ++z)
        {
            CUDA_SAFE_CALL(cudaMemcpy(coo_value[z], (*coo_value_h)[z],
                plane_allocated[z]*sizeof(float2), cudaMemcpyHostToDevice));
        }
    }
    
    CHECK_FOR_ERROR("end SparseVolume::setHostData");
}

__global__ void get_value_kernel
        (float2* value, float2* values, size_t* list, size_t target, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (list[idx] == target)
        {
            value->x = values[idx].x;
            value->y = values[idx].y;
        }
    }
}

float2 SparseVolume::getValue(cv::Point3i pos)
{
    CHECK_FOR_ERROR("begin SparseVolume::getValue");
    if (!is_allocated_buffer_64_d)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_64_d, 8*sizeof(float)));
        is_allocated_buffer_64_d = true;
    }
    
    assert(pos.x >= 0 && pos.x < width);
    assert(pos.y >= 0 && pos.y < height);
    assert(pos.z >= 0 && pos.z < depth);
    
    size_t zid = pos.z;
    size_t index = pos.y * width + pos.x;
    float2* value_d = (float2*)buffer_64_d;
    cudaMemset(value_d, 0, 8*sizeof(float));
    
    int nnz = plane_nnz[zid];
    if (nnz == 0)
    {
        // Need to avoid kernel call with grid_dim=0
        float2 value_h;
        value_h.x = 0;
        value_h.y = 0;
        return value_h;
    }
    
    int block_dim = 256;
    int grid_dim = ceil((float)nnz / (float)block_dim);
    get_value_kernel<<<grid_dim, block_dim>>>
        (value_d, coo_value[zid], coo_idx[zid], index, nnz);
    
    float2 value_h;
    cudaMemcpy(&value_h, value_d, 1*sizeof(float2), cudaMemcpyDeviceToHost);
    
    CHECK_FOR_ERROR("end SparseVolume::getValue");
    return value_h;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
