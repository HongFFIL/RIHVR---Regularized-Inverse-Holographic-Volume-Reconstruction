#include "sparse_segmentation.h"
#include "opencv_float2_traits.h"

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

SparseSegmentation::SparseSegmentation(SparseVolume input_data)
{
    params = input_data.getParams();
    
    width = input_data.getWidth();
    height = input_data.getHeight();
    num_planes = params.num_planes;
    
    // Make an internal copy of the input data
    complex_data.initialize(input_data);
    binary_data.initialize(input_data);
    working_data.initialize(input_data);
    complex_data = input_data;
    binary_data = input_data;
    working_data = input_data;
    
    complex_plane.allocateCuData(width, height, 1, sizeof(float2));
    binary_plane.allocateCuData(width, height, 1, sizeof(float2));
    
    is_binarized = false;
    is_labeled = false;
    
    return;
}

//============================= OPERATORS ====================================

//============================= OPERATIONS ===================================

__global__ void binarize_kernel(float2* binary, float2* input, double thr, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float intensity = input[idx].x*input[idx].x + input[idx].y*input[idx].y;
        // intensity = sqrt(intensity);
        //binary[idx].x = (intensity > thr)? sqrt(intensity) : 0.0;
        binary[idx].x = (intensity > thr)? 1.0 : 0.0;
        binary[idx].y = 0.0;
    }
}

__global__ void mask_kernel(float2* output, float2* input, double thr, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float intensity = input[idx].x*input[idx].x + input[idx].y*input[idx].y;
        output[idx].x = (intensity > thr)? input[idx].x : 0.0;
        output[idx].y = (intensity > thr)? input[idx].y : 0.0;
    }
}

void SparseSegmentation::binarize()
{
    CHECK_FOR_ERROR("begin SparseSegmentation::binarize");
    printf("binarizing volume...\n");
    // Compute the threshold
    // Assuming that input min intensity is in normalized pixel values (0-255)
    double max_intensity = complex_data.calcMaxIntensity();
    // double max_intensity = std::sqrt(complex_data.calcMaxIntensity());
    double thr = params.segment_min_intensity*max_intensity/256.0;
    printf("max intensity = %f, thr = %f\n", max_intensity, thr);
    
    size_t block_dim = 256;
    size_t grid_dim = ceil((double)(width*height)/(double)block_dim);
    
    // Binarize the volume
    for (size_t zid = 0; zid < num_planes; zid++)
    {
        //printf("binarizing plane %d of %d\n", zid, num_planes);
        complex_data.getPlane(&complex_plane, zid);
        binary_data.getPlane(&binary_plane, zid);
        float2* complex_d = (float2*)complex_plane.getCuData();
        float2* binary_d = (float2*)binary_plane.getCuData();
        
        binarize_kernel<<<grid_dim, block_dim>>>
            (binary_d, complex_d, thr, width*height);
        mask_kernel<<<grid_dim, block_dim>>>
            (complex_d, complex_d, thr, width*height);
        
        binary_plane.setCuData(binary_d);
        complex_plane.setCuData(complex_d);
        binary_data.setPlane(binary_plane, zid);
        complex_data.setPlane(complex_plane, zid);
    }
    
    this->is_binarized = true;
    
    char filename[FILENAME_MAX];
    
    /*
    printf("saving intermediate results...\n");
    sprintf(filename, "complex_voxels_%04d",
        params.start_image);
    complex_data.saveData(filename);
    sprintf(filename, "binarized_voxels_%04d",
        params.start_image);
    binary_data.saveData(filename);
    */
    CHECK_FOR_ERROR("end SparseSegmentation::binarize");
}

void SparseSegmentation::binarizeWorking()
{
    CHECK_FOR_ERROR("begin SparseSegmentation::binarizeWorking");
    printf("binarizing volume...\n");
    double thr = 0.0;
    
    size_t block_dim = 256;
    size_t grid_dim = ceil((double)(width*height)/(double)block_dim);
    
    // Binarize the volume
    for (size_t zid = 0; zid < num_planes; zid++)
    {
        //printf("binarizing plane %d of %d\n", zid, num_planes);
        working_data.getPlane(&complex_plane, zid);
        binary_data.getPlane(&binary_plane, zid);
        float2* complex_d = (float2*)complex_plane.getCuData();
        float2* binary_d = (float2*)binary_plane.getCuData();
        
        binarize_kernel<<<grid_dim, block_dim>>>
            (binary_d, complex_d, thr, width*height);
        
        binary_plane.setCuData(binary_d);
        binary_data.setPlane(binary_plane, zid);
        working_data.unGetPlane(&complex_plane, zid);
        binary_data.unGetPlane(&binary_plane, zid);
    }
    
    this->is_binarized = true;
    
    CHECK_FOR_ERROR("end SparseSegmentation::binarizeWorking");
}

void SparseSegmentation::close()
{
    assert(is_binarized);
    
    /*
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    complex_data.getHostData(&coo_idx, NULL, &plane_nnz, NULL);
    
    for (size_t zid = 0; zid < 20; ++zid)
    {
        printf("  host plane_nnz[%d] = %d\n", zid, plane_nnz[zid]);
    }
    
    printf("Plane 2:\n");
    for (int idx = 0; idx < plane_nnz[2]; ++idx)
    {
        printf("    index %d = %d\n", idx, coo_idx[2][idx]);
    }
    
    binary_data.getPlane(&binary_plane, 0);
    cv::Mat plane_mat = binary_plane.getMatData();
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            float2 value = plane_mat.at<float2>(i,j);
            if (value.x != 0.0 || value.y != 0.0)
            {
                printf("found non-zero at (%d,%d) = %f,%f\n", 
                    i,j, value.x, value.y);
                
                plane_mat.at<float2>(i,j).x += 2.0;
            }
        }
    }
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            float2 value = plane_mat.at<float2>(i,j);
            if (value.x != 0.0 || value.y != 0.0)
            {
                printf("      non-zero at (%d,%d) = %f,%f\n", 
                    i,j, value.x, value.y);
                
                value.x = 2;
                plane_mat.at<float2>(i,j) = value;
            }
        }
    }
    
    //this->dilate();
    //this->erode();
    */
    
    /*
    printf("saving intermediate results...\n");
    char filename[FILENAME_MAX];
    sprintf(filename, "pre_closed_%04d", params.start_image);
    binary_data.saveProjections(filename, MAX_CMB, true);
    */
    
    printf("dilate...\n");
    simpleMorph(cv::MORPH_DILATE);
    printf("erode...\n");
    simpleMorph(cv::MORPH_ERODE);
    
    /*
    printf("saving intermediate results...\n");
    sprintf(filename, "closed_voxels_%04d",
        params.start_image);
    binary_data.saveData(filename);
    sprintf(filename, "post_closed_%04d", params.start_image);
    binary_data.saveProjections(filename, MAX_CMB, true);
    */
}

void SparseSegmentation::simpleMorph(cv::MorphTypes morph_type)
{
    if (params.segment_close_size <= 0) return;
    
    switch (morph_type)
    {
        case cv::MORPH_DILATE:
        case cv::MORPH_ERODE:
            break;
        default:
        {
            std::cout << "SparseSegmentation::simpleMorph: Unrecognized MorphType\n";
            throw HOLO_ERROR_INVALID_ARGUMENT;
        }
    }
    
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    binary_data.getHostData(&coo_idx, NULL, &plane_nnz, NULL);
    
    int cs = params.segment_close_size;
    
    // Count number of voxels in the structuring element
    int num_strel_values = 0;
    for (int sz = -cs; sz <= cs; ++sz) {
        for (int sx = -cs; sx <= cs; ++sx) {
            for (int sy = -cs; sy <= cs; ++sy) {
                // Check that point is in spherical structuring element
                int dr2 = sx*sx + sy*sy + sz*sz;
                if (dr2 <= cs*cs)
                {
                    num_strel_values++;
                }
            }
        }
    }
    
    cv::Mat plane_mat;
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        // Get the plane we will be working on
        plane_mat = cv::Mat::zeros(height, width, CV_32FC2);
        
        for (int sz = -cs; sz <= cs; ++sz)
        {
            int plane_z = (int)zid + sz;
            if (plane_z < 0 || plane_z >= num_planes) continue;
            
            // Every point on this plane will affect the target plane
            for (int idx = 0; idx < plane_nnz[plane_z]; ++idx)
            {
                size_t index = coo_idx[plane_z][idx];
                size_t xid = index % width;
                size_t yid = index / width;
                
                // Determine target points affected by this point
                for (int sx = -cs; sx <= cs; ++sx)
                {
                    for (int sy = -cs; sy <= cs; ++sy)
                    {
                        // Check that point is in spherical structuring element
                        int dr2 = sx*sx + sy*sy + sz*sz;
                        int target_x = (int)xid + sx;
                        int target_y = (int)yid + sy;
                        bool valid_x = (target_x >= 0) && (target_x < width);
                        bool valid_y = (target_y >= 0) && (target_y < height);
                        if (dr2 <= cs*cs && valid_x && valid_y)
                        {
                            plane_mat.at<float2>(target_y, target_x).x += 1;
                        }
                    }
                }
            }
        }
        
        // Push the changes to the volume
        binary_plane.setMatData(plane_mat);
        if (morph_type == cv::MORPH_ERODE)
        {
            // Avoid equality test by decreasing threshold slightly
            double thr = num_strel_values - 0.1;
            binary_plane.threshold(thr, cv::THRESH_TOZERO, 0.0);
        }
        binary_data.setPlane(binary_plane, zid);
    }
    
    // Free all the host data
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        free(coo_idx[zid]);
    }
    free(plane_nnz);
}

void my_sort(size_t* array, size_t numel)
{
    // Use bubble sort for now because it's easy to write
    // If time is an issue, consider using heapsort instead
    // see https://brilliant.org/wiki/sorting-algorithms/
    if (numel < 2) return; // Is already sorted
    for (size_t n = numel; n > 0; --n)
    {
        for (size_t i = 0; i < n-1; ++i)
        {
            //printf("      i = %d\n", i);
            if (array[i] > array[i+1])
            {
                size_t temp = array[i+1];
                array[i+1] = array[i];
                array[i] = temp;
            }
        }
    }
    
    // Test the sorting
    for (size_t i = 0; i < numel-1; ++i)
    {
        assert(array[i] < array[i+1]);
    }
    
    return;
}

void my_sort(size_t* array, size_t numel, float2* follower)
{
    // Use bubble sort for now because it's easy to write
    // If time is an issue, consider using heapsort instead
    // see https://brilliant.org/wiki/sorting-algorithms/
    if (numel < 2) return; // Is already sorted
    for (size_t n = numel; n > 0; --n)
    {
        for (size_t i = 0; i < n-1; ++i)
        {
            //printf("      i = %d\n", i);
            if (array[i] > array[i+1])
            {
                size_t temp = array[i+1];
                array[i+1] = array[i];
                array[i] = temp;
                
                float2 temp2 = follower[i+1];
                follower[i+1] = follower[i];
                follower[i] = temp2;
            }
        }
    }
    
    // // Test the sorting
    // for (size_t i = 0; i < numel-1; ++i)
    // {
    //     assert(array[i] < array[i+1]);
    // }
    
    return;
}

int my_contains(size_t* array, size_t target, size_t numel)
{
    // Assume that array has been sorted using my_sort
    
    // Handle trivial cases first
    if (numel == 0) return -1;
    if (numel == 1)
    {
        if (array[0] == target) return 0;
        else return -1;
    }
    
    // Use a binary search algorithm
    size_t left_idx = 0;
    size_t right_idx = numel-1;
    while (right_idx - left_idx > 1)
    {
        size_t test_idx = (left_idx + right_idx) / 2;
        if (array[test_idx] >= target)
            right_idx = test_idx;
        else
            left_idx = test_idx;
    }
    size_t left_val = array[left_idx];
    size_t right_val = array[right_idx];
    if (left_val == target)
        return left_idx;
    if (right_val == target)
        return right_idx;
    return -1;//*/
}

void SparseSegmentation::labelConnectedComponents()
{
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    float2** coo_value;     // array of planes, values of non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    binary_data.getHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    
    // Sort the index values to speed up later searches
    // value data will be overwritten so no need to include in sort
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        my_sort(coo_idx[zid], plane_nnz[zid]);
    }
    
    // Initialize by setting value equal to index
    // Labels will be size_t, cast to float2 for sake of storage
    assert(sizeof(float2) == sizeof(size_t));
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t global_idx = coo_idx[zid][idx] + zid*width*height;
            coo_value[zid][idx] = *reinterpret_cast<float2*>(&global_idx);
        }
    }
    
    printf("count total_nnz\n");
    size_t total_nnz = 0;
    for (size_t zid = 0; zid < num_planes; ++zid) total_nnz += plane_nnz[zid];
    
    // Allocate list of all indices to neighbors
    size_t*** neighbors = (size_t***)malloc(num_planes*sizeof(size_t**));
    for (int zid = 0; zid < num_planes; ++zid)
    {
        neighbors[zid] = (size_t**)malloc(plane_nnz[zid]*sizeof(size_t*));
        for (int idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            neighbors[zid][idx] = (size_t*)malloc(27*sizeof(size_t));
        }
    }
    
    // Make list of all the offset values
    int offsets[9]; // 9 = 3*3 block of surrounding voxels on plane
    int oid = 0;
    for (int xs = -1; xs <= 1; ++xs)
    {
        for (int ys = -1; ys <= 1; ++ys)
        {
            offsets[oid] = ys*width + xs;
            oid++;
        }
    }
   
   for (size_t zid = 0; zid < num_planes; ++zid)
   {
       for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
       {
            size_t this_idx = coo_idx[zid][idx];
            for (int oid = 0; oid < 9; ++oid)
            {
                for (int zs = -1; zs <= 1; ++zs)
                {
                    size_t target_idx = this_idx + offsets[oid];
                    int dst = my_contains(
                        coo_idx[zid+zs], target_idx, plane_nnz[zid+zs]);
                    size_t neighbor_idx = oid + (zs+1)*9;
                    neighbors[zid][idx][neighbor_idx] = dst;
                }
            }
       }
   }
    
    bool is_changed = true;
    int while_loop_count = 0;
    while (is_changed)
    {
        int num_changed = 0;
        while_loop_count++;
        is_changed = false;
        for (size_t zid = 0; zid < num_planes; ++zid)
        {
            for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
            {
                size_t this_idx = coo_idx[zid][idx];
                size_t this_val = *reinterpret_cast<size_t*>(&(coo_value[zid][idx]));
                for (size_t nidx = 0; nidx < 27; ++nidx)
                {
                    int dst = neighbors[zid][idx][nidx];
                    int zs = (nidx / 9) - 1;
                    if (dst >= 0)
                    {
                        size_t target_val = 
                            *reinterpret_cast<size_t*>(&(coo_value[zid+zs][dst]));
                        if (target_val < this_val)
                        {
                            this_val = target_val;
                            is_changed = true;
                            num_changed++;
                        }
                    }
                }
                coo_value[zid][idx] = *reinterpret_cast<float2*>(&this_val);
            }
        }
        //printf("CCL loop number %d: num changed = %d\n",
        //    while_loop_count, num_changed);
    }
    
    binary_data.setHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    
    // Free all the host data
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        free(coo_idx[zid]);
        free(coo_value[zid]);
    }
    free(plane_nnz);
    
    free(neighbors);
    
    is_labeled = true;
}

void SparseSegmentation::hExtendedMaxima(double h)
{
    CHECK_FOR_ERROR("begin SparseSegmentation::hExtendedMaxima");
    printf("begin SparseSegmentation::hExtendedMaxima\n");
    double max_intensity = complex_data.calcMaxIntensity();
    printf("max intensity = %f, h = %f\n", max_intensity, h);
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    float2** coo_value;     // array of planes, values of non-zero elements
    float2** original_coo_value;     // array of planes, values of non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    printf("getHostData\n");
    complex_data.getHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    printf("get original data\n");
    complex_data.getHostData(NULL, &original_coo_value, NULL, NULL);
    //working_data.saveProjections("hmaxima_initial", MAX_CMB, true);
    
    // Sort the index values to speed up later searches
    // This time we do need ot include the value in the sort
    printf("sort\n");
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        my_sort(coo_idx[zid], plane_nnz[zid], coo_value[zid]);
    }
    
    // Convert to intenisty for segmentation
    printf("convert to intensity\n");
    h = sqrt(h);
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            float2 val = coo_value[zid][idx];
            assert(val.y == 0.0);
            coo_value[zid][idx].x = sqrt(val.x*val.x + val.y*val.y) - h;
            coo_value[zid][idx].y = 0.0;
            original_coo_value[zid][idx].x = sqrt(val.x*val.x + val.y*val.y);
            original_coo_value[zid][idx].y = 0.0;
        }
    }
    
    // Allocate list of all indices to neighbors
    printf("allocate neighbors\n");
    size_t*** neighbors = (size_t***)malloc(num_planes*sizeof(size_t**));
    for (int zid = 0; zid < num_planes; ++zid)
    {
        neighbors[zid] = (size_t**)malloc(plane_nnz[zid]*sizeof(size_t*));
        for (int idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            neighbors[zid][idx] = (size_t*)malloc(27*sizeof(size_t));
        }
    }
    
    // Make list of all the offset values
    int offsets[9]; // 9 = 3*3 block of surrounding voxels on plane
    int oid = 0;
    for (int xs = -1; xs <= 1; ++xs)
    {
        for (int ys = -1; ys <= 1; ++ys)
        {
            offsets[oid] = ys*width + xs;
            oid++;
        }
    }
    
    printf("set neighbors\n");
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
                size_t this_idx = coo_idx[zid][idx];
                for (int oid = 0; oid < 9; ++oid)
                {
                    for (int zs = -1; zs <= 1; ++zs)
                    {
                        if ((zid+zs < 0) || (zid+zs >= num_planes))
                        {
                            size_t neighbor_idx = oid + (zs+1)*9;
                            neighbors[zid][idx][neighbor_idx] = -1;
                            continue;
                        }
                        
                        size_t target_idx = this_idx + offsets[oid];
                        int dst = my_contains(
                            coo_idx[zid+zs], target_idx, plane_nnz[zid+zs]);
                        size_t neighbor_idx = oid + (zs+1)*9;
                        neighbors[zid][idx][neighbor_idx] = dst;
                    }
                }
        }
    }
    
    // h-maxima transform is morphological image reconstruction with a marker
    // I-h and mask I
    printf("transform\n");
    bool is_changed = true;
    int while_loop_count = 0;
    while (is_changed)
    {
        printf("while loop count: %d\n", while_loop_count++);
        int num_changed = 0;
        int rare_cases = 0;
        is_changed = false;
        for (size_t zid = 0; zid < num_planes; ++zid)
        {
            for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
            {
                float this_val = coo_value[zid][idx].x;
                float this_original_val = original_coo_value[zid][idx].x;
                for (size_t nidx = 0; nidx < 27; ++nidx)
                {
                    int dst = neighbors[zid][idx][nidx];
                    int zs = (nidx / 9) - 1;
                    if (dst >= 0)
                    {
                        float target_val = coo_value[zid+zs][dst].x;
                        if (target_val > this_val)
                        {
                            if (target_val < this_original_val)
                            {
                                this_val = target_val;
                                is_changed = true;
                                num_changed++;
                            }
                            else if (this_val < this_original_val)
                            {
                                this_val = this_original_val;
                                is_changed = true;
                                num_changed++;
                                rare_cases++;
                            }
                        }
                    }
                }
                if (this_val < 0) this_val = 0.0;
                coo_value[zid][idx].x = this_val;
            }
        }
        printf("  num_changed = %06d, rare_cases = %06d\n", num_changed, rare_cases);
    }
    
    /*
    printf("threshold\n");
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        // printf("zid = %d: plane_nnz[zid] = %d\n", zid, plane_nnz[zid]);
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            float this_val = coo_value[zid][idx].x;
            // float this_original_val = original_coo_value[zid][idx].x;
            // printf("zid=%04d idx=%06d: this_val = %f\n", zid, idx, this_val);
            
            if (this_val < 0.0)
            {
                coo_value[zid][idx].x = 0;
            }
        }
    }
    */
   /*
    printf("threshold\n");
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        // printf("zid = %d: plane_nnz[zid] = %d\n", zid, plane_nnz[zid]);
        for (size_t idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            float this_val = coo_value[zid][idx].x;
            float this_original_val = original_coo_value[zid][idx].x;
            // printf("zid=%04d idx=%06d: this_val = %f\n", zid, idx, this_val);
            
            if (this_val > 0.0)
            {
                // printf("had a value > 0.0\n");
                this_val = this_original_val - this_val;
                this_val = (this_val < 0)? 0.0 : this_val;
                coo_value[zid][idx].x = this_val;
            }
        }
    }
    */
    
    working_data.setHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    working_data.saveProjections("hmaxima", MAX_CMB, true);
    
    // Free all the host data
    for (size_t zid = 0; zid < num_planes; ++zid)
    {
        free(coo_idx[zid]);
        free(coo_value[zid]);
    }
    free(plane_nnz);
    // free(neighbors);
    
    printf("end SparseSegmentation::hExtendedMaxima\n");
    CHECK_FOR_ERROR("end SparseSegmentation::hExtendedMaxima");
    return;
}

ObjectCloud SparseSegmentation::extractObjects()
{
    assert(is_labeled);
    
    // ObjectCloud constructor does all the work
    ObjectCloud particles(binary_data);
    particles.includeIntensity(complex_data);
    return particles;
}

//============================= ACCESS     ===================================

//============================= INQUIRY    ===================================

/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////

