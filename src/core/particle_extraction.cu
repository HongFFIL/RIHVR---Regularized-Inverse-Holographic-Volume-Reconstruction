#include "particle_extraction.h"  // class implemented

using namespace umnholo;

/************************** ParticleExtraction ******************************/

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================
ParticleExtraction::ParticleExtraction(Deconvolution& deconv) : OpticalField(deconv), norm(deconv)
{
    cmb = deconv.combinedXY();
    threshold = 0.375;
    is_dilated = false;
    return; 
}

ParticleExtraction::ParticleExtraction(Reconstruction& recon) : OpticalField(recon)
{
    if (recon.isState(OPTICALFIELD_STATE_UNALLOCATED))
    {
        threshold = -1;
        is_dilated = false;
        return;
    }
    if (!recon.isState(OPTICALFIELD_STATE_RECONSTRUCTED))
    {
        std::cout << "ParticleExtraction: constructor argument must be of state "
            << "OPTICALFIELD_STATE_RECONSTRUCTED" << std::endl;
        std::cout << "State was " << recon.getState() << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    cmb = recon.combinedXY(MAX_CMB);
    threshold = -1;
    is_dilated = false;
    return; 
}

void ParticleExtraction::destroy()
{
    this->data.destroy();
    this->undilated_data.destroy();
}

void ParticleExtraction::destroy_iterative()
{
    this->undilated_data.destroy();
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

/** 
 * Use automatic SNR enhancement and thresholding method developed by
 * Mostafa Toloui 2015
 */
void ParticleExtraction::enhance()
{
    DECLARE_TIMING(ENH_ENHANCE_FINDTHR);
    START_TIMING(ENH_ENHANCE_FINDTHR);
    float threshold2d = this->norm.findThreshold2d(cmb);
    SAVE_TIMING(ENH_ENHANCE_FINDTHR);

    DECLARE_TIMING(ENH_ENHANCE_AVE_FILTER);
    START_TIMING(ENH_ENHANCE_AVE_FILTER);
    norm.averageFilterVolume();
    //int plane_size = width*height;
    //for (int plane_idx = 0; plane_idx < params.num_planes; ++plane_idx)
    //{
    //    norm.averageFilter(plane_idx);
    //}
    SAVE_TIMING(ENH_ENHANCE_AVE_FILTER);
    DECLARE_TIMING(ENH_ENHANCE_THRESHOLD);
    START_TIMING(ENH_ENHANCE_THRESHOLD);
    norm.threshold(threshold2d);
    norm.replaceFilterBorder(256.0);
    SAVE_TIMING(ENH_ENHANCE_THRESHOLD);

    DECLARE_TIMING(ENH_ENHANCE_NORMALIZE_BLOCKS);
    START_TIMING(ENH_ENHANCE_NORMALIZE_BLOCKS);
    norm.normalizeBlocks();
    SAVE_TIMING(ENH_ENHANCE_NORMALIZE_BLOCKS);
    
    return;
}

void ParticleExtraction::enhance(float window_size)
{
    this->norm.setWindowSize(window_size);
    this->enhance();
    
    return;
}

__global__ void binarize_inv_kernel(float* normalized_d, float thr, size_t size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        normalized_d[idx] = (normalized_d[idx] > thr)? 0 : 1;
    }
    
    return;
}

__global__ void binarize_kernel(float* normalized_d, float thr, size_t size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        normalized_d[idx] = (normalized_d[idx] > thr)? 1 : 0;
    }
    
    return;
}

void ParticleExtraction::binarize(int type)
{
    if (!norm.isState(EXTRACTION_STATE_NORMALIZED))
    {
        std::cout << "ParticleExtraction::binarize: Error volume must be properly normalized first" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    this->binarize(threshold, type);
}

void ParticleExtraction::binarize(double thr, int type)
{
    assert(!this->isState(OPTICALFIELD_STATE_UNALLOCATED));
    
    CHECK_FOR_ERROR("begin ParticleExtraction::binarize");
    float* normalized_d = NULL;
    if (norm.getData().isAllocated())
        normalized_d = (float*)norm.getData().getCuData();
    else if (this->getData().isAllocated())
        normalized_d = (float*)this->getData().getCuData();
    else
    {
        std::cout << "ParticleExtraction::binarize: Missing data" << std::endl;
        throw HOLO_ERROR_MISSING_DATA;
    }
    
    size_t numel = width * height * depth;
    dim3 dimBlock(256);
    dim3 dimGrid(ceil(numel / (float)dimBlock.x));
    
    if (type == cv::THRESH_BINARY_INV)
        binarize_inv_kernel<<<dimGrid, dimBlock>>>(normalized_d, thr, numel);
    else if (type == cv::THRESH_BINARY)
        binarize_kernel<<<dimGrid, dimBlock>>>(normalized_d, thr, numel);
    else
    {
        std::cout << "ParticleExtraction::binarize: Invalid type" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    this->state = EXTRACTION_STATE_BINARY;
    CHECK_FOR_ERROR("end ParticleExtraction::binarize");
    return;
}

CuMat ParticleExtraction::combinedXY()
{
    assert(!this->isState(OPTICALFIELD_STATE_UNALLOCATED));
    
    cv::Mat result = cv::Mat::ones(height, width, CV_32F);
    result *= 255;

    CuMat single_plane;
    size_t plane_size = width * height;
    single_plane.setCuData(data.getCuData(), data.getRows(), data.getCols(), data.getDataType());
    
    if (this->state == EXTRACTION_STATE_NORMALIZED)
    {
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            single_plane = this->getPlane(zidx);
            cv::min(result, single_plane.getMatData(), result);
        }
    }
    else if (this->state == EXTRACTION_STATE_BINARY)
    {
        result = cv::Mat::zeros(height, width, CV_32F);
        for (int zidx = 0; zidx < params.num_planes; ++zidx)
        {
            single_plane = this->getPlane(zidx);
            cv::max(result, single_plane.getMatData(), result);
        }
    }
    else
    {
        std::cout << "ParticleExtraction::combinedXY Error: ";
        std::cout << "state must be either EXTRACTION_STATE_NORMALIZED or EXTRACTION_STATE_BINARY" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    CuMat cumat_result;
    cumat_result.setMatData(result);

    return cumat_result;
}

__global__ void dilation_3x3x3_kernel(float* data_d, size_t width, size_t height, size_t depth, float* out_d)
{
    // IMPORTANT: Blocks must overlap, do not use standard gridDim calculation
    // Example:
    //     dim3 blockDim(32, 32, 1);
    //     dim3 gridDim(ceil(width / 30.0), ceil(height / 30.0), depth);
    //     size_t shared_size = blockDim.x * blockDim.y * sizeof(bool);
    // Use register tiling in z. Load one plane into shared memory. 
    // Calculate the sum of the 9 elements of interest in that plane. 
    // Use this sum for 3 consecutive z steps before discarding it.

    // Indices into the global memory
    int gx = blockIdx.x*(blockDim.x-2) + threadIdx.x-1;
    int gy = blockIdx.y*(blockDim.y-2) + threadIdx.y-1;
    int gz = blockIdx.z - 1;

    int gidx = gz*width*height + gy*width + gx;
    
    // Indices into shared memory
    // Shared memory will include halo data because of 3x reuse
    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int sidx = sy * blockDim.x + sx;
    
    extern __shared__ bool sh_plane[];
    
    // 1 pixel border overlaps. Identify threads that are not in that region
    bool compute_thread = ((sx > 0) && (sy > 0) && 
                           (sx < blockDim.x - 1) && (sy < blockDim.y - 1) && 
                           (gx < width) && (gy < height));
    
    int cidx = sidx;
    int write_idx = gidx;
    int sy_stride = blockDim.x;
    int gz_stride = width*height;
    
    // For register tiling
    float top = 0;
    float middle = 0;
    float bottom = 0;
    
    // Calculate the sum of the voxels in plane z - 1
    sh_plane[sidx] = ((gx > 0) && (gy > 0) && (gz > 0) && (gx < width) && (gy < height))? 
        data_d[gidx]==1.0 : false;
    __syncthreads();
    if (compute_thread)
    {
        top = 
                sh_plane[cidx - sy_stride - 1] + 
                sh_plane[cidx - sy_stride - 0] + 
                sh_plane[cidx - sy_stride + 1] + 
                sh_plane[cidx -         0 - 1] + 
                sh_plane[cidx -         0 - 0] + 
                sh_plane[cidx -         0 + 1] + 
                sh_plane[cidx + sy_stride - 1] + 
                sh_plane[cidx + sy_stride - 0] + 
                sh_plane[cidx + sy_stride + 1];
    }
    
    // Calculate the sum of the voxels in plane z
    gidx += gz_stride;
    gz++;
    write_idx += gz_stride;
    sh_plane[sidx] = ((gx > 0) && (gy > 0) && (gx < width) && (gy < height))? 
        data_d[gidx]==1.0 : false;
    __syncthreads();
    if (compute_thread)
    {
        middle = 
                sh_plane[cidx - sy_stride - 1] + 
                sh_plane[cidx - sy_stride - 0] + 
                sh_plane[cidx - sy_stride + 1] + 
                sh_plane[cidx -         0 - 1] + 
                sh_plane[cidx -         0 - 0] + 
                sh_plane[cidx -         0 + 1] + 
                sh_plane[cidx + sy_stride - 1] + 
                sh_plane[cidx + sy_stride - 0] + 
                sh_plane[cidx + sy_stride + 1];
    }
    
    // Calculate the sum of the voxels in plane z + 1
    gidx += gz_stride;
    gz++;
    sh_plane[sidx] = ((gx > 0) && (gy > 0) && (gx < width) && (gy < height) && (gz < depth))? 
        data_d[gidx]==1.0 : false;
    __syncthreads();
    if (compute_thread)
    {
        bottom = 
                sh_plane[cidx - sy_stride - 1] + 
                sh_plane[cidx - sy_stride - 0] + 
                sh_plane[cidx - sy_stride + 1] + 
                sh_plane[cidx -         0 - 1] + 
                sh_plane[cidx -         0 - 0] + 
                sh_plane[cidx -         0 + 1] + 
                sh_plane[cidx + sy_stride - 1] + 
                sh_plane[cidx + sy_stride - 0] + 
                sh_plane[cidx + sy_stride + 1];
        
        out_d[write_idx] = (top + middle + bottom > 0)? 1.0 : 0.0;
    }
}

__global__ void dilation_se_kernel
        (float* data_d, const uint8_t* struct_elem, size_t se_size, 
         size_t width, size_t height, size_t depth, float* out_d)
{
    // Indices into the global memory
    int gx = blockIdx.x*(blockDim.x-2) + threadIdx.x-1;
    int gy = blockIdx.y*(blockDim.y-2) + threadIdx.y-1;
    int gz = blockIdx.z;
    int write_idx = gz*width*height + gy*width + gx;
    //write_idx = gz*width*height + gx*width + gy;
    
    // Indices into shared memory
    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int sidx = threadIdx.y * blockDim.x + threadIdx.x;
    int share_width = blockDim.x;
    
    // 1 pixel border overlaps. Identify threads that are not in border
    bool compute_thread = ((sx > 0) && (sy > 0) && 
                           (sx < blockDim.x - 1) && (sy < blockDim.y - 1) && 
                           (gx < width) && (gy < height));
    
    extern __shared__ uint8_t sh_plane3[];
    
    // Matlab sets center = floor((size + 1)/2)
    // but, Matlab uses convolution notation where matrices flip
    int center = (se_size-1) - (floor((se_size + 1)/2.0) - 1); // Matches with MATLAB
    
    // Each thread calculates result for one voxel
    // Must check each neighbor voxel against the structuring element (SE)
    
    // Allow better readibility when checking bounds
    bool in_global;
    bool in_shared;
    
    float vox_count = 0;
    
    // Outer loop over planes for better memory accesses
    for (int sezid = 0; sezid < se_size; ++sezid)
    {
        int zdiff = sezid - center; // relative position of plane to SE
        
        // Load plane into shared memory
        in_global = (gz+zdiff >= 0) && (gx >= 0) && (gy >= 0) && 
                    (gx < width) && (gy < height) && (gz+zdiff < depth);
        int gidx = (gz+zdiff)*width*height + gy*width + gx;
        sh_plane3[sidx] = (in_global)? data_d[gidx] : 0;
        __syncthreads();
        
        if (compute_thread)
        {
            for (int sexid = 0; sexid < se_size; ++sexid)
            {
                int xdiff = sexid - center;
                for (int seyid = 0; seyid < se_size; ++seyid)
                {
                    int ydiff = seyid - center;
                    in_global = (gx+xdiff >= 0) && (gy+ydiff >= 0) && (gz+zdiff >= 0) && 
                                (gx+xdiff < width) && (gy+ydiff < height) && (gz+zdiff < depth);
                    in_shared = (sx+xdiff >= 0) && (sy+ydiff >= 0) && 
                                (sx+xdiff < share_width) && (sy+ydiff < share_width);
                    int gidx2 = (gz+zdiff)*width*height + (gy+ydiff)*width + (gx+xdiff);
                    int sidx2 = (sy+ydiff)*share_width + (sx+xdiff);
                    int se_idx = sezid*se_size*se_size + seyid*se_size + sexid;
                    uint8_t se_val = struct_elem[se_idx];
                    vox_count += (in_shared)? sh_plane3[sidx2]*se_val : 
                                 (in_global)? data_d[gidx2]*se_val : 0;
                    //vox_count += (in_shared)? sh_plane3[sidx2]*se_val : 
                    //             ((in_global)? 0 : 0);
                }
            }
        }
        __syncthreads();
    }
    
    if (compute_thread) out_d[write_idx] = (vox_count > 0)? 1.0 : 0.0;
}

void ParticleExtraction::dilate(int size)
{
    CHECK_FOR_ERROR("begin ParticleExtraction::dilate");
    if (this->state != EXTRACTION_STATE_BINARY)
    {
        std::cout << "ParticleExtraction::dilate Error: ";
        std::cout << "State must be EXTRACTION_STATE_BINARY" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    float* data_d = (float*)this->data.getCuData();
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil(width / 30.0), ceil(height / 30.0), depth);
    size_t shared_size = blockDim.x * blockDim.y * sizeof(bool);
    
    undilated_data.allocateCuData(width, height, depth, sizeof(float));
    float* undilated_d = (float*)undilated_data.getCuData();
    cudaMemcpy(undilated_d, data_d, width*height*depth*sizeof(float), cudaMemcpyDeviceToDevice);
    
    /*float* undilated_h = (float*)malloc(width*height*depth*sizeof(float));
    cudaMemcpy(undilated_h, undilated_d, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);
    int voxel_count = 0;
    for (int i =0; i < width*height*depth; ++i)
    {
        if (undilated_h[i] != 0)
        {
            voxel_count++;
            if (voxel_count < 100) printf("undilated_h[%d] = %f\n", i, undilated_h[i]);
        }
    }
    printf("Counted %d voxels in undilated_h\n", voxel_count);*/
    
    CuMat struct_element;
    struct_element.allocateCuData(size, size, size, sizeof(uint8_t));
    uint8_t* struct_elem_d = (uint8_t*)struct_element.getCuData();
    uint8_t* struct_elem_h = (uint8_t*)malloc(size*size*size*sizeof(uint8_t));
    for (int i = 0; i < size*size*size; ++i) struct_elem_h[i] = 1;
    cudaMemcpy(struct_elem_d, struct_elem_h, size*size*size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    dilation_se_kernel<<<gridDim, blockDim, shared_size>>>
        (undilated_d, struct_elem_d, size, width, height, depth, data_d);
    
    is_dilated = true;
    CHECK_FOR_ERROR("ParticleExtraction::dilate");
    return;
}

__global__ void maskReplace_kernel(uint32_t* inout, float* mask, uint32_t set_val, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        inout[idx] = (mask[idx] != 0)? inout[idx] : set_val;
    }
}

void ParticleExtraction::undoDilate(uint32_t zero_val)
{
    if (!is_dilated) return; // Nothing to undo
    
    float* undilated_d = (float*)undilated_data.getCuData();
    uint32_t* modified_d = (uint32_t*)data.getCuData();
    
    int numel = width*height*depth;
    dim3 blockDim(256, 1, 1);
    dim3 gridDim(ceil(numel / 256.0), 1, 1);
    maskReplace_kernel<<<gridDim, blockDim>>>(modified_d, undilated_d, zero_val, numel);
    
    is_dilated = false;
    
    CHECK_FOR_ERROR("ParticleExtraction::undoDilate");
    return;
}

ObjectCloud ParticleExtraction::extractObjects()
{
    CHECK_FOR_ERROR("begin ParticleExtraction::extractObjects");
    if (this->state != EXTRACTION_STATE_BINARY)
    {
        std::cout << "ParticleExtraction::extractObjects Error: ";
        std::cout << "Volume must be binarized prior to extracting objects" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    this->labelConnectedComponents();
    
    //printf("ParticleExtraction::extractObjects before ObjectCloud constructor\n");
    ObjectCloud objects1;
    int num_objects = objects1.countConnectedObjects(data);
    objects1.allocateObjects(num_objects);
    
    // Need to undo dilation after counting particles because particles
    // are counted assuming they have value = index
    uint32_t zero_val = -1;
    this->undoDilate(zero_val);
    
    objects1.extractObjects(data);
    
    //ObjectCloud cloud2;
    //printf("Using countConnectedObjects rather than constructor: %d\n", cloud2.countConnectedObjects(data));
    //ObjectCloud objects(data);
    
    this->state = OPTICALFIELD_STATE_GARBAGE;
    return objects1;
}

PointCloud3d ParticleExtraction::extractCentroids(Reconstruction& recon, Hologram& holo)
{
    CHECK_MEMORY("begin ParticleExtraction::extractCentroids");
    Parameters params = recon.getParams();
    
    Hologram temp_holo;
    holo.copyTo(&temp_holo); // to avoid any possible changes
    
    /*{cv::namedWindow("Holo", CV_WINDOW_AUTOSIZE);
    cv::Mat plane = holo.getData().getReal().getMatData();
    double plane_min, plane_max;
    cv::minMaxIdx(plane, &plane_min, &plane_max);
    cv::Mat plane2 = plane - plane_min;
    plane2 = plane2 / (plane_max - plane_min);
    cv::imshow("Holo", plane2);
    while (true)
    {
        char key = (char)cv::waitKey(1);
        if (key == 27) // 'esc' key was pressed
        {
            cv::destroyAllWindows();
            break;
        }
    }}*/
    
    // Compute and threshold max intensity projection
    CHECK_MEMORY("before ParticleExtraction::projectMaxIntenisty");
    cmb = recon.projectMaxIntensity(temp_holo);
    CHECK_MEMORY("after ParticleExtraction::projectMaxIntenisty");
    
    /*{cv::namedWindow("CMB", CV_WINDOW_AUTOSIZE);
    cv::Mat plane = cmb.getMatData();
    double plane_min, plane_max;
    cv::minMaxIdx(plane, &plane_min, &plane_max);
    printf("cmb min = %f, max = %f\n", plane_min, plane_max);
    cv::Mat plane2 = plane - plane_min;
    plane2 = plane2 / (plane_max - plane_min);
    cv::imshow("CMB", plane2);
    while (true)
    {
        char key = (char)cv::waitKey(1);
        if (key == 27) // 'esc' key was pressed
        {
            cv::destroyAllWindows();
            break;
        }
    }}*/
    
    double theshold = 0.0;
    threshold = this->computeThreshold(THRESHOLD_OTSU_CMB);
    CHECK_MEMORY("after ParticleExtraction::computeThreshold");
    //std::cout << "threshold = " << threshold << std::endl;
    //printf("threshold = %f\n", threshold);
    cmb.threshold(threshold, cv::THRESH_BINARY, 255);
    cmb.convertTo(CV_8U);
    
    /*{
        cv::namedWindow("Thresh CMB", CV_WINDOW_AUTOSIZE);
        cv::Mat plane = cmb.getMatData();
        double plane_min, plane_max;
        cv::minMaxIdx(plane, &plane_min, &plane_max);
        printf("thresholded cmb min = %f, max = %f\n", plane_min, plane_max);
        cv::Mat plane2 = plane;// - plane_min;
        //plane2 = plane2 / (plane_max - plane_min);
        cv::imshow("Thresh CMB", plane2);
        cv::waitKey();
        cv::destroyAllWindows();
    }//*/
    
    // Extract 2D centroids
    CHECK_MEMORY("begin ParticleExtraction::extracting 2d centroids");
    cv::Mat stats, centroids;
    cv::Mat dst = cmb.getMatData();
    cv::Mat src;
    dst.copyTo(src);
    //printf("depth src = %d, dst = %d\n", src.depth(), dst.depth());
    //printf("CV_8U = %d, CV_8S = %d, CV_32S = %d, CV_32F = %d\n", CV_8U, CV_8S, CV_32S, CV_32F);
    cv::connectedComponentsWithStats(src, dst, stats, centroids, 8, CV_16U);
    
    int num_particles = centroids.rows - 1; // Subtract 1 because background is included
    printf("extracted %d centroids\n", num_particles);
    
    if (num_particles == 0)
    {
        PointCloud3d empty_points(0);
        return empty_points;
    }
    
    centroids *= params.resolution; // Scale to length units
    
    PointCloud3d points(centroids.rowRange(1,num_particles+1));
    //points.round(); // So that we can use as index
    cv::Mat intensity_profiles(num_particles, params.num_planes, CV_32F);
    
    // Reconstruct planes and create intensity profiles
    CHECK_MEMORY("begin ParticleExtraction::reconstructing intensity profiles");
    temp_holo.fft(); // Must be done before reconstructIntensityTo
    CuMat recon_plane;
    cv::Mat recon_plane_mat;
    for (int zidx = 0; zidx < params.num_planes; ++zidx)
    {
        float z = params.plane_list[zidx];
        recon.reconstructIntensityTo(&recon_plane, temp_holo, z);
        recon_plane_mat = recon_plane.getMatData();
        
        // Fill intensity profile with value at each particle location
        for (int pid = 0; pid < num_particles; ++pid)
        {
            int c = std::round(points.getPosition(pid).x / params.resolution);
            int r = std::round(points.getPosition(pid).y / params.resolution);
            float val = recon_plane_mat.at<float>(r,c);
            intensity_profiles.at<float>(pid, zidx) = val;
        }
    }
    
    // Find peak of each particle's intensity profile
    CHECK_MEMORY("begin ParticleExtraction::intensity profile peak finding");
    for (int pid = 0; pid < num_particles; ++pid)
    {
        int zidx[2];
        double minVal = 0.0;
        cv::minMaxIdx(intensity_profiles.row(pid), &minVal, NULL, NULL, zidx);
        float z = params.plane_list[zidx[1]];
        points.setPointZ(pid, z);
        
        if (zidx == 0)
        {
            std::cout << "Warning, particle " << pid << " has focus plane at 0" << std::endl;
            std::cout << "Intensity profile: " << std::endl;
            std::cout << intensity_profiles.row(pid) << std::endl;
            std::cout << "minVal = " << minVal << std::endl;
            
            throw HOLO_ERROR_CRITICAL_ASSUMPTION;
        }
    }
    
    temp_holo.destroy();
    recon_plane.destroy();
    
    CHECK_MEMORY("end ParticleExtraction::extractCentroids");
    return points;
}

__global__ void cclInitialize_kernel(uint32_t* labels, size_t numel)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numel)
    {
        float val = ((float*)labels)[idx];
        labels[idx] = (val == 1)? idx : (uint32_t)-1;
    }
}

__global__ void cclScanConnected_kernel(uint32_t* labels, int width, int height, int depth, bool* updated)
{
    // IMPORTANT: Blocks must overlap, do not use standard gridDim calculation
    // Use register tiling in z. Load one plane into shared memory. 
    // Calculate the sum of the 9 elements of interest in that plane. 
    // Use this sum for 3 consecutive z steps before discarding it.

    // Indices into the global memory
    int gx = blockIdx.x*(blockDim.x-2) + threadIdx.x-1;
    int gy = blockIdx.y*(blockDim.y-2) + threadIdx.y-1;
    int gz = blockIdx.z - 1;

    int gidx = gz*width*height + gy*width + gx;
    
    // Indices into shared memory
    // Shared memory will include halo data because of 3x reuse
    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int sidx = sy * blockDim.x + sx;
    
    extern __shared__ uint32_t sh_plane2[];
    
    // 1 pixel border overlaps. Identify threads that are not in that region
    bool compute_thread = ((sx > 0) && (sy > 0) && 
                           (sx < blockDim.x - 1) && (sy < blockDim.y - 1) && 
                           (gx < width) && (gy < height));
    
    int cidx = sidx;
    int write_idx = gidx;
    int sy_stride = blockDim.x;
    int gz_stride = width*height;
    
    // For register tiling
    uint32_t top = (uint32_t)-1;
    uint32_t middle = (uint32_t)-1;
    uint32_t bottom = (uint32_t)-1;
    
    // Calculate the sum of the voxels in plane z - 1
    sh_plane2[sidx] = ((gx >= 0) && (gy >= 0) && (gz >= 0) && (gx < width) && (gy < height) && (gz < depth))? 
        labels[gidx] : (uint32_t)-1;
    __syncthreads();
    if (compute_thread)
    {
        // Load the surrounding elements from shared memory
        uint32_t l_lt = sh_plane2[cidx - sy_stride - 1];
        uint32_t l_mt = sh_plane2[cidx - sy_stride - 0];
        uint32_t l_rt = sh_plane2[cidx - sy_stride + 1];
        uint32_t l_lm = sh_plane2[cidx -         0 - 1];
        uint32_t l_mm = sh_plane2[cidx -         0 - 0];
        uint32_t l_rm = sh_plane2[cidx -         0 + 1];
        uint32_t l_lb = sh_plane2[cidx + sy_stride - 1];
        uint32_t l_mb = sh_plane2[cidx + sy_stride - 0];
        uint32_t l_rb = sh_plane2[cidx + sy_stride + 1];
        
        // Select the min of all the loaded elements
        bottom = (uint32_t)-1;
        bottom = min(l_lt, bottom);
        bottom = min(l_mt, bottom);
        bottom = min(l_rt, bottom);
        bottom = min(l_lm, bottom);
        bottom = min(l_mm, bottom);
        bottom = min(l_rm, bottom);
        bottom = min(l_lb, bottom);
        bottom = min(l_mb, bottom);
        bottom = min(l_rb, bottom);
        top = bottom;
    }
    
    // Calculate the min of the voxels in plane z
    gidx += gz_stride;
    gz++;
    write_idx += gz_stride;
    sh_plane2[sidx] = ((gx >= 0) && (gy >= 0) && (gz >= 0) && (gx < width) && (gy < height) && (gz < depth))? 
        labels[gidx] : (uint32_t)-1;
    __syncthreads();
    if (compute_thread)
    {
        // Load the surrounding elements from shared memory
        uint32_t l_lt = sh_plane2[cidx - sy_stride - 1];
        uint32_t l_mt = sh_plane2[cidx - sy_stride - 0];
        uint32_t l_rt = sh_plane2[cidx - sy_stride + 1];
        uint32_t l_lm = sh_plane2[cidx -         0 - 1];
        uint32_t l_mm = sh_plane2[cidx -         0 - 0];
        uint32_t l_rm = sh_plane2[cidx -         0 + 1];
        uint32_t l_lb = sh_plane2[cidx + sy_stride - 1];
        uint32_t l_mb = sh_plane2[cidx + sy_stride - 0];
        uint32_t l_rb = sh_plane2[cidx + sy_stride + 1];
        
        // Select the min of all the loaded elements
        bottom = (uint32_t)-1;
        bottom = min(l_lt, bottom);
        bottom = min(l_mt, bottom);
        bottom = min(l_rt, bottom);
        bottom = min(l_lm, bottom);
        bottom = min(l_mm, bottom);
        bottom = min(l_rm, bottom);
        bottom = min(l_lb, bottom);
        bottom = min(l_mb, bottom);
        bottom = min(l_rb, bottom);
        middle = bottom;
    }
    
    // Calculate the min of the voxels in plane z + 1
    gidx += gz_stride;
    gz++;
    sh_plane2[sidx] = ((gx >= 0) && (gy >= 0) && (gz >= 0) && (gx < width) && (gy < height) && (gz < depth))? 
        labels[gidx] : (uint32_t)-1;
    __syncthreads();
    if (compute_thread)
    {
        // Load the surrounding elements from shared memory
        uint32_t l_lt = sh_plane2[cidx - sy_stride - 1];
        uint32_t l_mt = sh_plane2[cidx - sy_stride - 0];
        uint32_t l_rt = sh_plane2[cidx - sy_stride + 1];
        uint32_t l_lm = sh_plane2[cidx -         0 - 1];
        uint32_t l_mm = sh_plane2[cidx -         0 - 0];
        uint32_t l_rm = sh_plane2[cidx -         0 + 1];
        uint32_t l_lb = sh_plane2[cidx + sy_stride - 1];
        uint32_t l_mb = sh_plane2[cidx + sy_stride - 0];
        uint32_t l_rb = sh_plane2[cidx + sy_stride + 1];
        
        // Select the min of all the loaded elements
        bottom = (uint32_t)-1;
        bottom = min(l_lt, bottom);
        bottom = min(l_mt, bottom);
        bottom = min(l_rt, bottom);
        bottom = min(l_lm, bottom);
        bottom = min(l_mm, bottom);
        bottom = min(l_rm, bottom);
        bottom = min(l_lb, bottom);
        bottom = min(l_mb, bottom);
        bottom = min(l_rb, bottom);
        
        uint32_t minl = top;
        minl = min(middle, minl);
        minl = min(bottom, minl);
        
        // Update only if voxel is an object and perviously had a higher label
        if (labels[write_idx] < (uint32_t)-1)
        {
            if (minl < labels[write_idx])
            {
                labels[write_idx] = minl;
                updated[0] = true;
            }
        }
    }
}

__global__ void cclFollowPath_kernel(uint32_t* labels, int size)
{
    // Each voxel contains the index of another voxel in the connected
    // component. Because of thread concurrency in the scanning step, these
    // indices can be followed in a chain to the lowest index in the component
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        // this_label points to another voxel in the component
        // May also point to itself if there is no earlier voxel
        uint32_t this_label = labels[idx];
        if (this_label < (uint32_t)-1)
        {
            int next_label = labels[this_label];
            while (next_label < this_label)
            {
                this_label = labels[next_label];
                next_label = labels[this_label];
            }
            labels[idx] = this_label;
        }
    }
}

void ParticleExtraction::labelConnectedComponents()
{
    CHECK_FOR_ERROR("begin ParticleExtraction::labelConnectedComponents");
    if (this->state != EXTRACTION_STATE_BINARY)
    {
        std::cout << "ParticleExtraction::labelConnectedComponents Error: ";
        std::cout << "State must be EXTRACTION_STATE_BINARY" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    size_t numel = width * height * depth;
    if (numel > std::numeric_limits<uint32_t>::max())
    {
        std::cout << "ParticleExtraction::labelConnectedComponents Error: ";
        std::cout << "Volume size exceeds limits" << std::endl;
        throw HOLO_ERROR_INVALID_DATA;
    }
    if (sizeof(float) != sizeof(uint32_t))
    {
        std::cout << "ParticleExtraction::labelConnectedComponents Error: ";
        std::cout << "CRITICAL: Assumes that float is 32 bit" << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    // Cast from float to uint32 for labeling convenience
    uint32_t* labels = (uint32_t*)data.getCuData();
    
    // Initial labeling. Object voxels get labeled with index. Others are
    // maximum uint32 (cast of -1)
    cclInitialize_kernel<<<ceil(numel / 512.0), 512>>>(labels, numel);
    cudaDeviceSynchronize();
    
    bool* updated = new bool;
    updated[0] = true;
    bool* updated_d = NULL;
    cudaMalloc((void**)&updated_d, sizeof(bool));
    cudaMemcpy(updated_d, updated, sizeof(bool), cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil(width / 30.0), ceil(height / 30.0), depth);
    size_t shared_mem = blockDim.x * blockDim.y * sizeof(uint32_t);
    
    while (updated[0])
    {
        updated[0] = false;
        cudaMemcpy(updated_d, updated, sizeof(bool), cudaMemcpyHostToDevice);
        cclScanConnected_kernel<<<gridDim, blockDim, shared_mem>>>(labels, width, height, depth, updated_d);
        cudaDeviceSynchronize();
        
        cclFollowPath_kernel<<<ceil(numel / 512.0), 512>>>(labels, numel);
        cudaDeviceSynchronize();
        cudaMemcpy(updated, updated_d, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(updated_d);
    this->state = EXTRACTION_STATE_CC_LABELED;
    CHECK_FOR_ERROR("ParticleExtraction::labelConnectedComponents");
}

double ParticleExtraction::computeThreshold(ThresholdMethod method)
{
    cv::Mat mat_cmb;
    cmb.getMatData().copyTo(mat_cmb);
    
    if (mat_cmb.channels() > 1)
    {
        std::cout << "ParticleExtraction::computeThreshold: Error: "
            << "cmb should only have 1 channel" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    double thr = 0;
    
    switch (method)
    {
    case THRESHOLD_OTSU_CMB:
    {
        // OpenCV Otsu threshold is only available for 8-bit images
        double minimum, maximum;
        cv::minMaxLoc(mat_cmb, &minimum, &maximum);
        mat_cmb = (mat_cmb - minimum) / (maximum - minimum);
        mat_cmb.convertTo(mat_cmb, CV_8U, 255);
    
        //cv::namedWindow("CMB", CV_WINDOW_AUTOSIZE);
        //cv::imshow("CMB", mat_cmb);
        //cv::waitKey(0);
        //cv::destroyAllWindows();
        
        //printf("cmb min = %f, max = %f\n", minimum, maximum);
        
        thr = cv::threshold(mat_cmb, mat_cmb, 0, 1, cv::THRESH_OTSU+cv::THRESH_BINARY);
        //printf("threshold for 8u = %f\n", thr);
        thr = (thr/255.0) * (maximum - minimum) + minimum;
        ///printf("scaled threshold = %f\n", thr);
        break;
    }
    case THRESHOLD_OTSU_VOLUME:
    {
        assert(!this->isState(OPTICALFIELD_STATE_UNALLOCATED));
        
        // TODO: Break histogram calculation into separate method
        // Volume is too large to use OpenCV, use NPPI instead
        
        NppiSize roi = {width*height, depth};
        int num_bins = 256;
        int num_levels = num_bins+1;
        int line_step = width*height*sizeof(float);
        float* data_d = (float*)this->data.getCuData();
        
        // First, find the required buffer size
        //int buffer_size = 0;
        //int alt_buffer_size = 0;
		size_t buffer_size = 0;
        size_t alt_buffer_size = 0;
        
        NPP_SAFE_CALL(nppiHistogramRangeGetBufferSize_32f_C1R(roi, num_levels, &buffer_size));
        NPP_SAFE_CALL(nppiMinMaxGetBufferHostSize_32f_C1R(roi, &alt_buffer_size));
        
        buffer_size = std::max(buffer_size, alt_buffer_size);
        
        // Allocate buffer
        Npp8u* buffer_d;
        CUDA_SAFE_CALL(cudaMalloc((void**)&buffer_d, buffer_size));
        
        // Find the min and max of the data
        Npp32f* data_min_d;
        Npp32f* data_max_d;
        CUDA_SAFE_CALL(cudaMalloc((void**)&data_min_d, 1*sizeof(Npp32f)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&data_max_d, 1*sizeof(Npp32f)));
        NPP_SAFE_CALL(nppiMinMax_32f_C1R(data_d, line_step, roi, data_min_d, data_max_d, buffer_d));
        Npp32f data_min, data_max;
        CUDA_SAFE_CALL(cudaMemcpy(&data_min, data_min_d, 1*sizeof(Npp32f), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&data_max, data_max_d, 1*sizeof(Npp32f), cudaMemcpyDeviceToHost));
        
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
        CUDA_SAFE_CALL(cudaMalloc((void**)&hist_d, num_bins*sizeof(Npp32s)));
        cudaMemset(hist_d, 0, num_bins*sizeof(Npp32s));
        NPP_SAFE_CALL(nppiHistogramRange_32f_C1R(data_d, line_step, roi, hist_d, bins_d, num_levels, buffer_d));
        Npp32s* hist_h = (Npp32s*)malloc(num_bins*sizeof(Npp32s));
        CUDA_SAFE_CALL(cudaMemcpy(hist_h, hist_d, num_bins*sizeof(Npp32s), cudaMemcpyDeviceToHost));
        
        // Save histogram to file
        char hist_filename[FILENAME_MAX];
        sprintf(hist_filename, "%s/histogram_volume.txt", params.output_path);
        FILE* fid = NULL;
        fid = fopen(hist_filename, "w");
        if (fid == NULL)
        {
            std::cout << "Unable to open file: " << hist_filename << std::endl;
            throw HOLO_ERROR_INVALID_FILE;
        }
        fprintf(fid, "begin, end, count\n");
        for (int i = 0; i < num_bins; ++i)
        {
            fprintf(fid, "%f, %f, %d\n", bins_h[i], bins_h[i+1], hist_h[i]);
        }
        fclose(fid);
        
        // Compute Otsu threshold from the histogram
        // This section is developed from the wikipedia page for Otsu's method
        double total = width*height*depth;
        double sum_b = 0;
        double w_b = 0;
        double maximum = 0;
        double sum1 = 0;
        int level = 0;
        for (int i = 0; i < num_bins; ++i)
            sum1 += i * hist_h[i];
        
        for (int i = 0; i < num_bins; ++i)
        {
            w_b = w_b + hist_h[i];
            double w_f = total - w_b;
            if ((w_b == 0) || (w_f == 0)) continue;
            sum_b = sum_b + i*hist_h[i];
            double m_f = (sum1 - sum_b) / w_f;
            double between = w_b*w_f*((sum_b/w_b) - m_f)*((sum_b/w_b) - m_f);
            if (between > maximum)
            {
                level = i;
                maximum = between;
            }
        }
        
        thr = bins_h[level+1];
        
        cudaFree(buffer_d);
        cudaFree(data_min_d);
        cudaFree(data_max_d);
        cudaFree(bins_d);
        cudaFree(hist_d);
        free(bins_h);
        free(hist_h);
        
        CHECK_FOR_ERROR("ParticleExtraction::computeThreshold THRESHOLD_OTSU_VOLUME");
        break;
    }
    default:
    {
        std::cout << "ParticleExtraction::computeThreshold: Error: "
            << "Unknown method: " << method << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    //printf("end ParticleExtraction::computeThreshold: thr = %f\n", thr);
    CHECK_FOR_ERROR("end ParticleExtraction::computeThreshold");
    return thr;
}

//============================= ACCESS     ===================================
//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////


/************************** TolouiSnrNormalize ******************************/
/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================
TolouiSnrNormalize::TolouiSnrNormalize()
{
    window_size = 6;
    filter_tile_size = 8;
    block_size = 32;
    threshold_tile_size = 8;
    return;
}

TolouiSnrNormalize::TolouiSnrNormalize(Deconvolution& deconv) : OpticalField(deconv)
{
    if (!deconv.isState(OPTICALFIELD_STATE_DECONVOLVED_REAL)) throw HOLO_ERROR_INVALID_STATE;
    
    window_size = 6;
    filter_tile_size = 8;
    block_size = 32;
    threshold_tile_size = 8;
    return;
}
//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

float TolouiSnrNormalize::findThreshold2d(CuMat cmb)
{
    int block_size = window_size * params.particle_diameter / params.resolution;
    int num_blocks_x = (cmb.getWidth() / block_size)*2 - 1;
    int num_blocks_y = (cmb.getHeight() / block_size)*2 - 1;
    int num_blocks = num_blocks_x * num_blocks_y;
    int block_step = block_size / 2;

    std::vector<double> minima(num_blocks);
    std::vector<double> stdevs(num_blocks);
    cv::Scalar mean_temp;
    cv::Scalar stdev_temp;
    
    // Apply scale to convert from population stdev (OpenCV) to sample
    // standard deviation (matlab default)
    double N = block_size * block_size;
    double stdev_scale = sqrt(N) / sqrt(N-1);

    cv::Mat roi(block_size, block_size, CV_32F, 0.0);

    int block_idx = 0;
    int start_x = 0;
    int start_y = 0;
    cv::Mat temp_min(num_blocks_x, num_blocks_y, CV_32F);
    cv::Mat temp_std(num_blocks_x, num_blocks_y, CV_32F);
    for (int idx_x = 0; idx_x < num_blocks_x; ++idx_x)
    {
        start_y = 0;
        for (int idx_y = 0; idx_y < num_blocks_y; ++idx_y)
        {
            roi = cmb.getMatData()(cv::Rect(start_x, start_y, block_size, block_size));
            roi.convertTo(roi, CV_32F);
            cv::minMaxIdx(roi, &(minima[block_idx]), NULL);

            cv::meanStdDev(roi, mean_temp, stdev_temp);
            stdevs[block_idx] = stdev_temp.val[0] * stdev_scale;

            start_y += block_step;
            block_idx++;
        }
        start_x += block_step;
    }

    double mean_min = std::accumulate(minima.begin(), minima.end(), 0.0) / minima.size();
    double mean_stdev = std::accumulate(stdevs.begin(), stdevs.end(), 0.0) / stdevs.size();
    double sq_sum = std::inner_product(stdevs.begin(), stdevs.end(), stdevs.begin(), 0.0);
    double stdev_stdev = std::sqrt((sq_sum / (stdevs.size() - 1)) - mean_stdev * mean_stdev);
    
    CHECK_FOR_ERROR("findThreshold2d");
    return mean_min - mean_stdev - stdev_stdev;
}

__global__ void averageFilter_kernel(float* input_d, int width, int height, float* filtered_d) {

    extern __shared__ float tile[]; // Size should be (blockDim.x)*(blockDim.y)*sizeof(float)

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = y*width + x;

    int tile_w = blockDim.x;
    int tile_h = blockDim.y;
    int shared_idx = threadIdx.y*tile_w + threadIdx.x;

    float base = 1.0/((float)(2*2));
    float edge = 0.0;

    tile[shared_idx] = input_d[idx];
    __syncthreads();

    // This is the actual filter
    // The conditionals are somewhat ugly but are necessary to reduce divergence
    // All they do is check the boundary conditions, first for shared memory and
    // then for the entire image.
    float sum = 0;

    // center
    sum += tile[shared_idx +      0 + 0];

    // sides
    sum += (tx <   tile_w-1) ? tile[shared_idx +      0 + 1] :
        (x ==  width-1) ? edge : input_d[idx +     0 + 1];
    sum += (ty <   tile_h-1) ? tile[shared_idx + tile_w + 0] :
        (y == height-1) ? edge : input_d[idx + width + 0];

    // corners
    sum += ((ty < tile_h-1) && (tx < tile_w-1)) ? tile[shared_idx + tile_w + 1] :
        (y == height-1) ? edge :
        (x == width-1) ? edge : input_d[idx + width + 1];

    filtered_d[idx] = sum*base;
}

void TolouiSnrNormalize::averageFilter(int plane_idx)
{
    CHECK_FOR_ERROR("before averageFilter");
    dim3 gridDim(width/filter_tile_size, height/filter_tile_size, 1);
    dim3 blockDim(filter_tile_size, filter_tile_size, 1);

    int plane_size = width * height;

    float* in_plane_d = NULL;
    float* out_plane_d = (float*)data.getCuData() + plane_size * plane_idx;
    CUDA_SAFE_CALL( cudaMalloc((void**)&in_plane_d, plane_size*sizeof(float)) );
    CHECK_FOR_ERROR("after malloc");
    cudaMemcpy(in_plane_d, out_plane_d, plane_size*sizeof(float), cudaMemcpyDeviceToDevice);
    CHECK_FOR_ERROR("after memcpy");

    //float* temp_h = (float*)malloc(width*height*sizeof(float));
    //cudaMemcpy(temp_h, out_plane_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    averageFilter_kernel<<<gridDim, blockDim, (blockDim.x)*(blockDim.y)*sizeof(float)>>>
        (in_plane_d, width, height, out_plane_d);
    //cudaMemcpy(temp_h, out_plane_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_plane_d);

    CHECK_FOR_ERROR("average_filter");
    return;
}

void TolouiSnrNormalize::averageFilterVolume()
{
    CHECK_FOR_ERROR("before averageFilterVolume");
    dim3 gridDim(width/filter_tile_size, height/filter_tile_size, 1);
    dim3 blockDim(filter_tile_size, filter_tile_size, 1);

    int plane_size = width * height;

    float* in_plane_d = NULL;
    CUDA_SAFE_CALL( cudaMalloc((void**)&in_plane_d, plane_size*sizeof(float)) );
    
    for (int plane_idx = 0; plane_idx < this->params.num_planes; ++plane_idx)
    {
        float* out_plane_d = (float*)data.getCuData() + plane_size * plane_idx;
        cudaMemcpy(in_plane_d, out_plane_d, plane_size*sizeof(float), cudaMemcpyDeviceToDevice);

        averageFilter_kernel<<<gridDim, blockDim, (blockDim.x)*(blockDim.y)*sizeof(float)>>>
            (in_plane_d, width, height, out_plane_d);
    }

    cudaFree(in_plane_d);

    CHECK_FOR_ERROR("averageFilterVolume");
    return;
}

__global__ void replaceFilterBorder_kernel(float* data, float value, size_t size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int width = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;
    int idx = z*width*height + y*width + x;
    
    if (idx < size)
    {
        data[idx] = (x == width-1)? value : (y == height-1)? value : data[idx];
    }
}

void TolouiSnrNormalize::replaceFilterBorder(float value)
{
    CHECK_FOR_ERROR("before TolouiSnrNormalize::replaceFilterBorder");
    
    dim3 gridDim(width/filter_tile_size, height/filter_tile_size, depth);
    dim3 blockDim(filter_tile_size, filter_tile_size, 1);
    replaceFilterBorder_kernel<<<gridDim, blockDim>>>
        ((float*)data.getCuData(), value, width*height*depth);
    
    CHECK_FOR_ERROR("TolouiSnrNormalize::replaceFilterBorder");
    return;
}

__global__ void my_threshold_kernel(float* data_d, size_t size, float thr_val, float top_val)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        data_d[idx] = (data_d[idx] < thr_val) ? data_d[idx] : top_val;
    }
    return;
}
__global__ void my_threshold_kernel(float* data_d, int width, int height, float threshold1, float topval)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = y*width + x;

    if (idx < width*height)
    {
    //    data_d[idx] = (data_d[idx] > threshold1) ? topval : data_d[idx];
        //intensity_map_d[idx] = (intensity_map_d[idx] > threshold1) ? 255 : 0;
    }
}

void TolouiSnrNormalize::threshold(float value)
{
    CHECK_FOR_ERROR("before TolouiSnrNormalize::threshold");

    int numel = width * height * depth;
    my_threshold_kernel<<<ceil(numel / 1024.0), 1024>>>
        ((float*)data.getCuData(), numel, value, 256.0);

    CHECK_FOR_ERROR("after TolouiSnrNormalize::threshold");
    return;
}

__global__ void min_max_block_kernel(float* data_d, int width, int height, int depth, int block_size, float* minima, float* maxima)
{
    // Each thread handles its own block
    // Note: This has awful memory coalescing behavior
    int bx = blockIdx.x*blockDim.x + threadIdx.x;
    int by = blockIdx.y*blockDim.y + threadIdx.y;
    int bz = blockIdx.z*blockDim.z + threadIdx.z;

    // Blocks have 50% overlap
    int startX = bx*block_size/2;
    int startY = by*block_size/2;
    int startZ = bz*block_size/2;

    int num_tiles_x = 2*width/block_size - 1;
    int num_tiles_y = 2*height/block_size - 1;
    int num_tiles_z = 2*depth/block_size - 1;

    int bidx = bz*num_tiles_x*num_tiles_y + by*num_tiles_x + bx;

    if ((bx < num_tiles_x) && (by < num_tiles_y) && (bz < num_tiles_z))
    {

        float current_min = FLT_MAX;
        float current_max = -FLT_MAX;

        int start_idx = startZ*width*height + startY*width + startX;
        int idx = start_idx;

        for (int z = 0; z < block_size; ++z)
        {
            for (int y = 0; y < block_size; ++y)
            {
                for (int x = 0; x < block_size; ++x)
                {
                    idx = start_idx + z*width*height + y*width + x;
                    current_min = (data_d[idx] < current_min) ? data_d[idx] : current_min;
                    current_max = (data_d[idx] > current_max) ? data_d[idx] : current_max;
                }
            }
        }

        minima[bidx] = current_min;
        maxima[bidx] = current_max;

    }
}

void TolouiSnrNormalize::minMaxBlocks(float* minima_d, float* maxima_d, int &num_blocks)
{
    int num_blocks_x = 2*width/block_size - 1;
    int num_blocks_y = 2*height/block_size - 1;
    int num_blocks_z = 2*depth/block_size - 1;
    num_blocks = num_blocks_x*num_blocks_y*num_blocks_z;
    
    // Allocate only if not already allocated
    if ((minima_d == NULL) || (maxima_d == NULL))
    {
        std::cout << "TolouiSnrNormalize::minMaxBlocks error: Pointer must be allocated" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    float* min_h = new float;
    float* max_h = new float;
    
    CuMat block;
    for (int bz = 0; bz < num_blocks_z; ++bz)
    {
        for (int by = 0; by < num_blocks_y; ++by)
        {
            for (int bx = 0; bx < num_blocks_x; ++bx)
            {
                int start_x = bx * block_size/2;
                int start_y = by * block_size/2;
                int start_z = bz * block_size/2;
                block = this->getRoi(start_x, start_y, start_z, block_size, block_size, block_size);
                int bidx = bz*num_blocks_x*num_blocks_y + by*num_blocks_x + bx;
                block.minMax<float>(minima_d + bidx, maxima_d + bidx);
            }
        }
    }
    
    return;
}

__global__ void deconv_normalize_block_kernel(float* data_d, int width, int height, int depth, int block_size, float* minima, float* maxima)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int idx = z*width*height + y*width + x;
    if (idx < width*height*depth)
    {

        //int bsize = blockDim.x;
        int bsize = block_size;
        int halfbs = bsize / 2;

        // Determine which block the point belongs to
        // Select the block with the highest bidx for this to belong to
        int grid_width = width/(bsize/2) - 1;
        int grid_height = height/(bsize/2) - 1;
        int grid_depth = depth/(bsize/2) - 1;
        int bx = min(x / halfbs, grid_width-1);
        int by = min(y / halfbs, grid_height-1);
        int bz = min(z / halfbs, grid_depth-1);
        int bidx = bz*grid_width*grid_height + by*grid_width + bx;

        // Calculate the normalized value using the min and max of block
        float minimum = minima[bidx];
        float maxmin = maxima[bidx] - minimum;
        data_d[idx] = (maxmin > 0) ? (data_d[idx] - minimum)/maxmin : 1;
    }
}
void TolouiSnrNormalize::normalizeBlocks()
{
    int num_blocks_x = 2*width/block_size - 1;
    int num_blocks_y = 2*height/block_size - 1;
    int num_blocks_z = 2*depth/block_size - 1;
    int num_blocks = num_blocks_x*num_blocks_y*num_blocks_z;

    float* minima;
    float* maxima;
    float* minima_h = (float*)malloc(num_blocks*sizeof(float));
    float* maxima_h = (float*)malloc(num_blocks*sizeof(float));
    CUDA_SAFE_CALL(cudaMalloc((void**)&minima, num_blocks*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&maxima, num_blocks*sizeof(float)));
    cudaMemset(minima, 0.0, num_blocks*sizeof(float));
    cudaMemset(maxima, 0.0, num_blocks*sizeof(float));

    float* data_d = (float*)data.getCuData();

    dim3 dimBlock2(8, 8, 8);
    dim3 dimGrid2(ceil(num_blocks_x/8.0), ceil(num_blocks_y/8.0), ceil(num_blocks_z/8.0));
    min_max_block_kernel<<<dimGrid2, dimBlock2>>>(data_d, width, height, depth, block_size, minima, maxima);

    dim3 dimBlock(block_size, block_size, 1); // Thread coarsening accounts for remaining 32 in z
    dim3 dimGrid(ceil(width/block_size), ceil(height/block_size), ceil(depth/1.0));
    deconv_normalize_block_kernel<<<dimGrid, dimBlock>>>(data_d, width, height, depth, 32, minima, maxima);

    CHECK_FOR_ERROR("TolouiSnrNormalize::normalizeBlocks");
    this->state = EXTRACTION_STATE_NORMALIZED;
    return;
}
//============================= ACCESS     ===================================
//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////


/***************************** Extraction2D *********************************/

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

Extraction2D::Extraction2D()
{
    filter_size = cv::Size(10, 10);
    filter_center = cv::Point(4, 4);
    thresh_value = 100;
    opening_size = 10;
    //opening_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
    //    cv::Size(2*opening_size + 1, 2*opening_size + 1), 
    //    cv::Point(opening_size, opening_size));
    //std::cout << "Extraction2D::Extraction2D opening_element: " << std::endl;
    //std::cout << opening_element << std::endl;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

// Apply average filter, threshold, then extract blobs
PointCloud Extraction2D::extractCentroids(HoloSequence holos)
{
    DECLARE_TIMING(READ_IMAGE);
    DECLARE_TIMING(AVERAGE_FILTER);
    DECLARE_TIMING(FIND_THRESHOLD);
    DECLARE_TIMING(APPLY_THRESHOLD);
    DECLARE_TIMING(OPENING);
    DECLARE_TIMING(CCL);

    cv::Mat src, dst; // Source and destination for all operations
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat stats, centroids;

    Hologram holo;
    PointCloud result;
    CuMat data;

    //cv::Mat opening_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
    //    cv::Size(2*opening_size + 1, 2*opening_size + 1), 
    //    cv::Point(opening_size, opening_size));
    char opening_element_filename[FILENAME_MAX];
    sprintf(opening_element_filename, "%s/config/opening_element.tif", holos.getParams().exe_path);
    cv::Mat opening_element = cv::imread(opening_element_filename, cv::IMREAD_UNCHANGED);
    if (opening_element.empty())
    {
        std::cout << "Extraction2D::extractCentroids: Unable to open opening_element.tif" << std::endl;
        std::cout << "  full search path: <" << opening_element_filename << ">" << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }
    opening_element /= 255;
    CuMat open_data;
    open_data.setMatData(opening_element);
    CuMat buffer;
    buffer.setMatData(cv::Mat(holos.getHeight(), holos.getWidth(), CV_32F));

    //std::cout << "thresh_value = " << this->thresh_value << std::endl;
    //std::cout << "opening_element: " << std::endl << opening_element << std::endl;
    //std::cout << "type = " << opening_element.type() << std::endl;

    holos.reset();
    int num_processed = 0;
    while (holos.loadNextChunk(LOAD_UNCHANGED))
    {
        for (int i = 0; i < holos.getNumHolograms(); ++i)
        {
            START_TIMING(READ_IMAGE);
            holo = holos.getHologram(i);
            data = holo.getData().getReal();
            src = data.getMatData();
            dst = src.clone();
            STOP_TIMING(READ_IMAGE);
            SAVE_TIMING(READ_IMAGE);

            //char enhanced_filename[FILENAME_MAX];
            //cv::Mat temp = dst.clone();
            //temp.convertTo(temp, CV_8U);
            //sprintf(enhanced_filename, "%s/gpu_enh_%03d.tif", holos.getParams().output_path, num_processed + i);
            //cv::imwrite(enhanced_filename, temp);
            //printf("wrote to <%s>\n", enhanced_filename);

            // Average filter
            START_TIMING(AVERAGE_FILTER);
            blur(src, dst, filter_size, filter_center);
            //dst.copyTo(src);
            STOP_TIMING(AVERAGE_FILTER);
            SAVE_TIMING(AVERAGE_FILTER);

            // Threshold
            data.setMatData(dst);
            START_TIMING(FIND_THRESHOLD);
            dst.convertTo(dst, CV_8U);
            thresh_value = calcThreshold(&dst);
            STOP_TIMING(FIND_THRESHOLD);
            SAVE_TIMING(FIND_THRESHOLD);
            //std::cout << ";Threshold;" << thresh_value << std::endl;
            START_TIMING(APPLY_THRESHOLD);
            dst.convertTo(dst, CV_32F);
            //threshold(src, dst, thresh_value, 255, cv::THRESH_BINARY_INV);
            data.getCuData();
            data.threshold(thresh_value, cv::THRESH_BINARY_INV, 255);
            dst = data.getMatData();
            //dst.copyTo(src);
            STOP_TIMING(APPLY_THRESHOLD);
            SAVE_TIMING(APPLY_THRESHOLD);

            // Morphological opening
            START_TIMING(OPENING);
            data.morphOpening(open_data, buffer);
            dst = data.getMatData();
            dst.convertTo(dst, CV_8U);
            dst.copyTo(src);
            /*
            morphologyEx(src, dst, cv::MORPH_OPEN, opening_element);
            */
            dst.copyTo(src);
            STOP_TIMING(OPENING);
            SAVE_TIMING(OPENING);

            //char binarized_filename[FILENAME_MAX];
            //sprintf(binarized_filename, "%s/gpu_bw_%03d.tif", holos.getParams().output_path, num_processed + i);
            //cv::imwrite(binarized_filename, dst);

            // Extract blob properties
            //dst.convertTo(dst, CV_8U);
            //dst.copyTo(src);
            START_TIMING(CCL);
            cv::connectedComponentsWithStats(src, dst, stats, centroids, 8, CV_16U);
            
            // The first connected component is the background
            int num_blobs = centroids.rows-1;
            printf("  extractCentroids: holo %03d found %03d objects\n", num_processed + i, num_blobs);
            for (int n = 0; n < num_blobs; ++n)
            {
                Particle part;
                part.size = stats.at<int>(n+1, cv::CC_STAT_AREA);
                part.x = centroids.at<double>(n+1, 0);
                part.y = centroids.at<double>(n+1, 1);
                part.time = num_processed + i;
                result.setParticle(num_processed+i, n, part);
            }
            STOP_TIMING(CCL);
            SAVE_TIMING(CCL);
        }
        num_processed += holos.getNumHolograms();
    }
    
    printf("Extraction2D::extractCentroids processed %d holograms\n", num_processed);
    

    return result;
}

//============================= ACCESS     ===================================
//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

double Extraction2D::calcThreshold(cv::Mat* image)
{
    if (image->type() != CV_8U)
    {
        std::cout << "Extraction2d::calcThreshold: Error: Invalid input type" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }

    // Set parameters for histogram calculation
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;

    cv::calcHist(image, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, uniform, accumulate);

    // Find peak of histogram (intensity value)
    double max_val = 0;
    cv::Point max_idx = cv::Point(0, 0);
    cv::minMaxLoc(hist, NULL, &max_val, NULL, &max_idx);

    double peak = max_idx.y;
    return (peak - 0) / 2;
}

/////////////////////////////// PRIVATE    ///////////////////////////////////
