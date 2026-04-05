#include "point_cloud_3d.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

PointCloud3d::PointCloud3d(int count)
{
    this->num_points = count;
    this->track_ids.resize(count);
    this->positions.resize(count);
    this->intensities.resize(count);
    for (int i = 0; i < count; ++i)
    {
        intensities[i] = 1.0;
        track_ids[i] = -1;
    }
}

PointCloud3d::PointCloud3d(cv::Mat centroids)
{
    this->num_points = centroids.rows;
    this->track_ids.resize(this->num_points);
    this->positions.resize(this->num_points);
    this->intensities.resize(this->num_points);
    
    int dims = centroids.cols;
    assert(dims >= 1 && dims <= 3);
    assert(centroids.depth() == CV_64F);
    
    for (int n = 0; n < num_points; ++n)
    {
        positions[n].x = centroids.at<double>(n,0);
        if (dims > 1) positions[n].y = centroids.at<double>(n,1);
        else positions[n].y = 0.0;
        if (dims > 2) positions[n].z = centroids.at<double>(n,2);
        else positions[n].z = 0.0;
        
        intensities[n] = 1.0;
        track_ids[n] = -1;
    }
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

__global__ void pointcloud3d_calc_residual_kernel
    (float2* out, float2* in1, float2* in2, float scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx].x = in1[idx].x - scale*in2[idx].x;
        out[idx].y = in1[idx].y - scale*in2[idx].y;
    }
}

void PointCloud3d::calcResidual(Hologram* residual, Hologram& holo)
{
    size_t width = holo.getWidth();
    size_t height = holo.getHeight();
    CuMat estimated;
    estimated.allocateCuData(width, height, 1, sizeof(float2));
    float2* estimated0_d = (float2*)estimated.getCuData();
    cudaMemset(estimated0_d, 0, width*height*sizeof(float2));
    CuMat holo_data = holo.getData();
    CuMat residual_data = residual->getData();
    if (!residual_data.isAllocated())
        residual_data.allocateCuData(width, height, 1, sizeof(float2));
    
    for (int pid = 0; pid < num_points; ++pid)
    {
        buildEstimatedHologram(&estimated, pid, holo.getParams());
    }
    CHECK_MEMORY("PointCloud3d::calcResidual after buildEstimatedHologram");
    
    // Determine relative scaling to account for error in estimated intensity
    // This is particularly important on the first iteration since the
    //   intensities are initialized to an arbitrary constant
    CuMat real_estimated = estimated.getReal();
    CuMat real_holo_data = holo_data.getReal();
    CHECK_MEMORY("PointCloud3d::calcResidual after getReal");
    cv::Mat estimated_mat = real_estimated.getMatData();
    cv::Mat holo_mat = real_holo_data.getMatData();
    CHECK_MEMORY("PointCloud3d::calcResidual after getMatData");
    double est_min, est_max, holo_min, holo_max;
    cv::minMaxLoc(estimated_mat, &est_min, &est_max);
    cv::minMaxLoc(holo_mat, &holo_min, &holo_max);
    double scale = (holo_max - holo_min) / (est_max - est_min);
    //printf("calcResidual: holo_min = %f, holo_max = %f, est_min = %f, est_max = %f, scale = %f\n",
    //    holo_max, holo_min, est_max, est_min, scale);
    CHECK_MEMORY("PointCloud3d::calcResidual after finding scaling");
    
    // Compute residual = holo - scale*estimated
    float2* estimated_d = (float2*)estimated.getCuData();
    float2* holo_d = (float2*)holo_data.getCuData();
    float2* residual_d = (float2*)residual_data.getCuData();
    dim3 block_dim(256);
    dim3 grid_dim(width*height / block_dim.x);
    pointcloud3d_calc_residual_kernel<<<grid_dim, block_dim>>>
        (residual_d, holo_d, estimated_d, scale, width*height);
    
    residual_data.setCuData(residual_d);
    residual->setData(residual_data);
    residual->setState(HOLOGRAM_STATE_ARBITRARY);
    CHECK_MEMORY("PointCloud3d::calcResidual after computing residual");
    
    // Update intensities as jump start for next time
    for (int pid = 0; pid < num_points; ++pid)
    {
        intensities[pid] *= scale;
    }
    
    //Hologram est_holo(holo.getParams());
    //holo.copyTo(&est_holo);
    //est_holo.setData(estimated);
    //est_holo.setState(holo.getState());
    //holo.show("True Hologram", false);
    //est_holo.show("Estimated", false);
    //residual->show("Residual", false);
    
    estimated.destroy();
    real_estimated.destroy();
    real_holo_data.destroy();
    estimated_mat.deallocate();
    holo_mat.deallocate();
    
    return;
}

void PointCloud3d::optimize(Hologram& residual)
{
    Parameters params = residual.getParams();
    int num_iterations = 5;
    int num_eval_steps = 51;
    int max_displacement = 2 * params.resolution;
    
    buffer1.allocateCuData(residual.getWidth(), residual.getHeight(), 1, sizeof(float2));
    buffer2.allocateCuData(residual.getWidth(), residual.getHeight(), 1, sizeof(float2));
    
    for (int pid = 0; pid < num_points; ++pid)
    {
        DECLARE_TIMING(PointCloud3d_optimize_point);
        START_TIMING(PointCloud3d_optimize_point);
        CHECK_FOR_ERROR("begin PointCloud3d::optimize pid");
        cv::Point3f pos = positions[pid];
        float intensity = intensities[pid];
        
        // Crop to smaller ROI to reduce cost, avoid having too many objects
        // in the image when computing errors
        float lambda = params.wavelength;
        float reso = params.resolution;
        int roi_width = 2 * pos.z * lambda/(reso*reso);
        if (roi_width <= 1) continue;
        cv::Point roi_center(std::round(pos.x/params.resolution), std::round(pos.y/params.resolution));
        // Condition checking used from Hologram::crop
        int start_x = roi_center.x - roi_width/2;
        int start_y = roi_center.y - roi_width/2;
        if (start_x < 0) start_x = 0;
        if (start_y < 0) start_y = 0;
        if (start_x + roi_width > residual.getWidth())
            start_x = residual.getWidth() - roi_width;
        if (start_y + roi_width > residual.getHeight())
            start_y = residual.getHeight() - roi_width;
        cv::Rect roi(start_x, start_y, roi_width, roi_width);
        Hologram small_residual = residual.crop(roi);
        
        //printf("Optimizing point %d of %d, roi width = %d\n", pid, num_points, roi_width);
        //printf("  part was at (%0.2f, %0.2f, %0.2f), %0.2f\n",
        //    pos.x, pos.y, pos.z, intensity);
        
        // Shift position for ROI
        pos.x -= start_x*params.resolution;
        pos.y -= start_y*params.resolution;
        
        // Add particle back into the residual
        // As if all particles except current one have been removed
        CHECK_FOR_ERROR("before small_residual.getData");
        CuMat small_residual_data = small_residual.getData();
        CHECK_FOR_ERROR("before buildEstimatedHologram");
        buildEstimatedHologram(&small_residual_data, pos, intensity,
            params.resolution, params.wavelength, roi_width, roi_width);
        small_residual.setData(small_residual_data);
        CHECK_FOR_ERROR("after buildEstimatedHologram");
        
        float best_fval = evaluatePosition(pos, intensity, small_residual, params);
        float best_intensity = intensity;
        cv::Point3f best_pos = pos;
        //printf("    best fval = %f\n", best_fval);
        
        for (int iteration = 0; iteration < num_iterations; ++iteration)
        {
            // Optimize Intensity
            float I_start = intensity / 2;
            float I_end = intensity * 2;
            float I_step = (I_end - I_start) / (num_eval_steps - 1);
            for (float I_test = I_start; I_test <= I_end; I_test+=I_step)
            {
                float fval = evaluatePosition(pos, I_test, small_residual, params);
                if (fval < best_fval)
                {
                    best_fval = fval;
                    best_intensity = I_test;
                    //printf("    best fval = %f\n", best_fval);
                }
            }
            intensity = best_intensity;
            
            // Optimize x
            float x_start = best_pos.x - max_displacement;
            float x_end = best_pos.x + max_displacement;
            float x_step = (x_end - x_start) / (num_eval_steps - 1);
            for (pos.x = x_start; pos.x <= x_end; pos.x+=x_step)
            {
                float fval = evaluatePosition(pos, intensity, small_residual, params);
                if (fval < best_fval)
                {
                    best_fval = fval;
                    best_pos = pos;
                    //printf("    best fval = %f\n", best_fval);
                }
            }
            pos = best_pos;
            
            // Optimize y
            float y_start = best_pos.y - max_displacement;
            float y_end = best_pos.y + max_displacement;
            float y_step = (y_end - y_start) / (num_eval_steps - 1);
            for (pos.y = y_start; pos.y <= y_end; pos.y+=y_step)
            {
                float fval = evaluatePosition(pos, intensity, small_residual, params);
                if (fval < best_fval)
                {
                    best_fval = fval;
                    best_pos = pos;
                    //printf("    best fval = %f\n", best_fval);
                }
            }
            pos = best_pos;
            
            // Optimize z
            float z_start = best_pos.z - max_displacement;
            float z_end = best_pos.z + max_displacement;
            float z_step = (z_end - z_start) / (num_eval_steps - 1);
            for (pos.z = z_start; pos.z <= z_end; pos.z+=z_step)
            {
                float fval = evaluatePosition(pos, intensity, small_residual, params);
                if (fval < best_fval)
                {
                    best_fval = fval;
                    best_pos = pos;
                    //printf("    best fval = %f\n", best_fval);
                }
            }
            pos = best_pos;
        } // for iteration
        
        pos.x += start_x*params.resolution;
        pos.y += start_y*params.resolution;
        //printf("  part now at (%0.2f, %0.2f, %0.2f), %0.2f\n",
        //    pos.x, pos.y, pos.z, intensity);
        
        positions[pid] = pos;
        intensities[pid] = intensity;
        
        small_residual.destroy();
        
        SAVE_TIMING(PointCloud3d_optimize_point);
    } // for pid
    
    buffer1.destroy();
    buffer2.destroy();
    
    return;
}

void PointCloud3d::pruneGhosts()
{
    // TODO: Find better method for intensity cutoff
    // Option is to use mean, but that only works if very few are ghosts
    float intensity_cutoff = 60;
    
    int num_removed = 0;
    for (int i = 0; i < num_points; ++i)
    {
        bool is_valid = intensities[i] > intensity_cutoff;
        if (!is_valid)
        {
            track_ids.erase(track_ids.begin()+i);
            positions.erase(positions.begin()+i);
            intensities.erase(intensities.begin()+i);
            num_points--;
            i--;
            num_removed++;
        }
    }
    
    std::cout << "PointCloud3d::pruneGhosts removed "
        << num_removed << " ghosts" << std::endl;
    return;
}

void PointCloud3d::mergeIn(PointCloud3d& new_cloud)
{
    int old_num_points = this->num_points;
    int add_num_points = new_cloud.num_points;
    
    this->num_points = old_num_points + add_num_points;
    positions.resize(num_points);
    intensities.resize(num_points);
    track_ids.resize(num_points);
    
    for (int i = 0; i < add_num_points; ++i)
    {
        positions[old_num_points+i] = new_cloud.positions[i];
        intensities[old_num_points+i] = new_cloud.intensities[i];
        track_ids[old_num_points+i] = new_cloud.track_ids[i];
    }
    
    return;
}

void PointCloud3d::round()
{
    for (int n = 0; n < num_points; ++n)
    {
        positions[n].x = std::round(positions[n].x);
        positions[n].y = std::round(positions[n].y);
        positions[n].z = std::round(positions[n].z);
    }
}

//============================= ACCESS     ===================================

cv::Point3f PointCloud3d::getPosition(int idx)
{
    assert(idx < num_points);
    return positions[idx];
}

void PointCloud3d::setPointX(int idx, float x)
{
    assert(idx < num_points);
    positions[idx].x = x;
    return;
}

void PointCloud3d::setPointY(int idx, float y)
{
    assert(idx < num_points);
    positions[idx].y = y;
    return;
}

void PointCloud3d::setPointZ(int idx, float z)
{
    assert(idx < num_points);
    positions[idx].z = z;
    return;
}

void PointCloud3d::setPosition(int idx, cv::Point3f pos)
{
    assert(idx < num_points);
    positions[idx] = pos;
    return;
}

float PointCloud3d::getIntensity(int idx)
{
    assert(idx < num_points);
    return intensities[idx];
}

int PointCloud3d::getId(int idx)
{
    assert(idx < num_points);
    return track_ids[idx];
}

void PointCloud3d::setId(int idx, int track_id)
{
    assert(idx < num_points);
    track_ids[idx] = track_id;
    return;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////

__global__ void build_estimated_holo_kernel
    (float2* estimated_d, cv::Point3f pos, float intensity,
     float reso, float lambda, int nx, int ny)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = y_idx*nx + x_idx;
    
    // Limit extent of particle signal
    float x = x_idx * reso;
    float y = y_idx * reso;
    bool x_isvalid = abs(x - pos.x) < (lambda * pos.z / reso);
    bool y_isvalid = abs(y - pos.y) < (lambda * pos.z / reso);
    
    // Computation thread selection
    if (x_isvalid && y_isvalid && (x_idx < nx) && (y_idx < ny))
    {
        // Compute the intensity of the diffraction pattern
        float k0 = 2 * M_PI / lambda;
        float A = (k0 / (2 * M_PI * pos.y)) * (k0 / (2 * M_PI * pos.y));
        float B = k0 / (M_PI * pos.z);
        float r_sqr = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y);
        float value = A + B*sin(k0*r_sqr/(2*pos.z));
        value = -value * intensity;
        
        // Window the signal using Hann window
        float L = 2 * (lambda * pos.z / (reso*reso));
        float N = L - 1;
        float pos_pix_x = pos.x / reso;
        float pos_pix_y = pos.y / reso;
        float nx = x_idx - pos_pix_x + N/2;
        float ny = y_idx - pos_pix_y + N/2;
        float win_x = ((nx >= 0) && (nx <= N))?
            0.5 * (1 - cos(2*M_PI * nx/N)) : 0.0;
        float win_y = ((ny >= 0) && (ny <= N))?
            0.5 * (1 - cos(2*M_PI * ny/N)) : 0.0;
        float win = win_x * win_y;
        
        // Use += here because we want this to build on many particles
        estimated_d[idx].x += value * win;
        estimated_d[idx].y = 0.0;
    }
}

void PointCloud3d::buildEstimatedHologram
    (CuMat* estimated, size_t idx, const Parameters& params)
{
    CHECK_FOR_ERROR("begin PointCloud3d::buildEstimatedHologram 1");
    //printf("inside PointCloud3d::buildEstimatedHologram\n");
    assert(idx < num_points);
    
    // Alias parameters for simplified kernel call
    float reso = params.resolution;
    float lambda = params.wavelength;
    //printf("before getWidth\n");
    size_t nx = estimated->getWidth();
    //printf("after getWidth\n");
    size_t ny = estimated->getHeight();
    cv::Point3f pos = positions[idx];
    float intensity = intensities[idx];
    
    //printf("before estimated->getCuData\n");
    float2* estimated_d = (float2*)estimated->getCuData();
    //printf("after estimated->getCuData\n");
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil((float)nx/(float)block_dim.x),
                  ceil((float)ny/(float)block_dim.y));
    CHECK_FOR_ERROR("before build_estimated_holo_kernel");
    build_estimated_holo_kernel<<<grid_dim, block_dim>>>
        (estimated_d, pos, intensity, reso, lambda, nx, ny);
    cudaDeviceSynchronize();
    CHECK_FOR_ERROR("after build_estimated_holo_kernel");
    
    estimated->setCuData(estimated_d);
    
    CHECK_FOR_ERROR("end PointCloud3d::buildEstimatedHologram 1");
    return;
}

void PointCloud3d::buildEstimatedHologram
    (CuMat* estimated, cv::Point3f pos, float intensity,
     float reso, float lambda, size_t nx, size_t ny)
{
    CHECK_FOR_ERROR("begin PointCloud3d::buildEstimatedHologram 2");
    
    float2* estimated_d = (float2*)estimated->getCuData();
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil((float)nx/(float)block_dim.x),
                  ceil((float)ny/(float)block_dim.y));
    build_estimated_holo_kernel<<<grid_dim, block_dim>>>
        (estimated_d, pos, intensity, reso, lambda, nx, ny);
    CHECK_FOR_ERROR("after build_estimated_holo_kernel");
    
    estimated->setCuData(estimated_d);
    
    CHECK_FOR_ERROR("end PointCloud3d::buildEstimatedHologram 2");
    return;
}

__global__ void calc_residual_holo_kernel
    (float2* residual_d, float2* holo_d, cv::Point3f pos, float intensity,
     float reso, float lambda, int nx, int ny)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = y_idx*nx + x_idx;
    
    // Limit extent of particle signal
    float x = x_idx * reso;
    float y = y_idx * reso;
    bool x_isvalid = abs(x - pos.x) < (lambda * pos.z / reso);
    bool y_isvalid = abs(y - pos.y) < (lambda * pos.z / reso);
    
    // Computation thread selection
    if (x_isvalid && y_isvalid && (x_idx < nx) && (y_idx < ny))
    {
        // Compute the intensity of the diffraction pattern
        float k0 = 2 * M_PI / lambda;
        float A = (k0 / (2 * M_PI * pos.y)) * (k0 / (2 * M_PI * pos.y));
        float B = k0 / (M_PI * pos.z);
        float r_sqr = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y);
        float value = A + B*sin(k0*r_sqr/(2*pos.z));
        value = -value * intensity;
        
        // Window the signal using Hann window
        float L = 2 * (lambda * pos.z / (reso*reso));
        float N = L - 1;
        float pos_pix_x = pos.x / reso;
        float pos_pix_y = pos.y / reso;
        float nx = x_idx - pos_pix_x + N/2;
        float ny = y_idx - pos_pix_y + N/2;
        float win_x = ((nx >= 0) && (nx <= N))?
            0.5 * (1 - cos(2*M_PI * nx/N)) : 0.0;
        float win_y = ((ny >= 0) && (ny <= N))?
            0.5 * (1 - cos(2*M_PI * ny/N)) : 0.0;
        float win = win_x * win_y;
        
        // Use += here because we want this to build on many particles
        residual_d[idx].x = holo_d[idx].x - value * win;
        residual_d[idx].y = holo_d[idx].y;
    }
}

template <unsigned int blockSize>
__global__ void my_pc3d_l2norm_kernel(float2* data_d, float* buffer, int size)
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
                      data_d[i].y*data_d[i].y;
        sdata[tid] += data_d[i+blockSize].x*data_d[i+blockSize].x + 
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

float PointCloud3d::evaluatePosition
    (cv::Point3f pos, float intensity, Hologram& holo, Parameters& params)
{
    CHECK_FOR_ERROR("begin PointCloud3d::evaluatePosition");
    
    // Alias parameters for simplified kernel call
    float reso = params.resolution;
    float lambda = params.wavelength;
    size_t nx = holo.getWidth();
    size_t ny = holo.getHeight();
    
    float2* holo_d = (float2*)holo.getData().getCuData();
    float2* residual_d = (float2*)buffer1.getCuData();
    //cudaMalloc((void**)&residual_d, nx*ny * sizeof(float2));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil((float)nx/(float)block_dim.x),
                  ceil((float)ny/(float)block_dim.y));
    calc_residual_holo_kernel<<<grid_dim, block_dim>>>
        (residual_d, holo_d, pos, intensity, reso, lambda, nx, ny);
    cudaDeviceSynchronize();
    //CHECK_FOR_ERROR("after calc_residual_holo_kernel");
    
    // Summation
    int size = nx*ny;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    //CHECK_FOR_ERROR("before my_pc3d_l2norm_kernel");
    float* buffer_d = (float*)buffer2.getCuData();
    //cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    my_pc3d_l2norm_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        (residual_d, buffer_d, size);
    cudaDeviceSynchronize();
    //CHECK_FOR_ERROR("after my_pc3d_l2norm_kernel");
    
    //printf("dimGrid.x = %d, size = %d\n", dimGrid.x, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sumsqr = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sumsqr += buffer_h[i];
    }
    
    //cudaFree(buffer_d);
    //cudaFree(residual_d);
    
    CHECK_FOR_ERROR("end PointCloud3d::evaluatePosition");
    return sumsqr;
}