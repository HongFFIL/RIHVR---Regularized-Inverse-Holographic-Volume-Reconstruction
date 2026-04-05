#include "object_cloud.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

ObjectCloud::ObjectCloud()
{
    num_objects = 0;
    size_object_ids = 0;
    allocated_objects = 0;
    width = 0;
    height = 0;
}

ObjectCloud::ObjectCloud(size_t count)
{
    num_objects = 0;
    objects = new Blob3d[count];
    allocated_objects = count;
    size_object_ids = 0;
    width = 0;
    height = 0;
}

ObjectCloud::ObjectCloud(CuMat ccl_data)
{
    num_objects = -1;
    num_objects = countConnectedObjects(ccl_data);
    size_object_ids = 0;
    
    objects = new Blob3d[num_objects];
    allocated_objects = num_objects;
    for (int n = 0; n < num_objects; ++n)
        objects[n].setSourceSize(ccl_data.getWidth(), ccl_data.getHeight());
    
    if (num_objects > 0) extractObjects(ccl_data);
    
    return;
}

ObjectCloud::ObjectCloud(SparseVolume ccl_data)
{
    num_objects = -1;
    num_objects = countConnectedObjects(ccl_data);
    size_object_ids = 0;
    
    objects = new Blob3d[num_objects];
    allocated_objects = num_objects;
    for (int n = 0; n < num_objects; ++n)
        objects[n].setSourceSize(ccl_data.getWidth(), ccl_data.getHeight());
    
    if (num_objects > 0) extractObjects(ccl_data);
    
    return;
}

void ObjectCloud::destroy()
{
    if (size_object_ids > 0) free(object_ids);
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

__global__ void countLabels_kernel(uint32_t* labels, uint32_t* count, uint32_t* object_ids, size_t max_objects, size_t size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        // Unique objects have label equal to their index
        if (labels[idx] == idx)
        {
            uint32_t object_id = atomicAdd(count, 1);
            if (object_id < max_objects)
            {
                object_ids[object_id] = idx;
            }
        }
    }
}

size_t ObjectCloud::countConnectedObjects(CuMat ccl_data)
{
    if (ccl_data.getElemSize() != sizeof(uint32_t))
    {
        std::cout << "ObjectCloud::ObjectCloud Error: ";
        std::cout << "Input data must have elements of size 32 bits" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    size_t width = ccl_data.getWidth();
    size_t height = ccl_data.getHeight();
    size_t depth = ccl_data.getDepth();
    size_t numel = width * height * depth;
    
    // Count the number of objects in the field
    uint32_t* cc_count_d = NULL;
    cudaMalloc((void**)&cc_count_d, sizeof(uint32_t));
    cudaMemset(cc_count_d, 0, sizeof(uint32_t));
    uint32_t* ccl_data_d = (uint32_t*)ccl_data.getCuData();
    
    uint32_t* object_ids_d = NULL;
    cudaMalloc((void**)&object_ids_d, max_num_objects * sizeof(uint32_t));
    
    countLabels_kernel<<<ceil(numel / 512.0), 512>>>(ccl_data_d, cc_count_d, object_ids_d, max_num_objects, numel);
    
    uint32_t* cc_count_h = new uint32_t;
    cudaMemcpy(cc_count_h, cc_count_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    num_objects = *cc_count_h;
    
    object_ids = (uint32_t*)malloc(max_num_objects * sizeof(uint32_t));
    size_object_ids = max_num_objects;
    cudaMemcpy(object_ids, object_ids_d, max_num_objects * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    if (num_objects > max_num_objects)
    {
        std::cout << "ObjectCloud::ObjectCloud: Found too many objects" << std::endl;
    }
    
    cudaFree(cc_count_d);
    cudaFree(object_ids_d);
    return num_objects;
}

size_t ObjectCloud::countConnectedObjects(SparseVolume ccl_data)
{
    size_t depth = ccl_data.getDepth();
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    float2** coo_value;     // array of planes, values of non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    ccl_data.getHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    
    num_objects = 0;
    object_ids = (uint32_t*)malloc(max_num_objects * sizeof(uint32_t));
    for (int i = 0; i < max_num_objects; ++i) object_ids[i] = 0;
    size_object_ids = max_num_objects;
    for (int zid = 0; zid < depth; ++zid)
    {
        for (int idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t this_val = *reinterpret_cast<size_t*>(&(coo_value[zid][idx]));
            uint32_t obj_id = this_val;
            uint32_t* id = std::find(object_ids, object_ids + num_objects, obj_id);
            bool is_in = id != object_ids+num_objects;
            if (!is_in)
            {
                if (num_objects > max_num_objects)
                {
                    std::cout << "ObjectCloud::ObjectCloud: Found too many objects\n";
                    break;
                }
                object_ids[num_objects] = this_val;
                num_objects++;
            }
        }
    }
    printf("Number of particles: %d\n", num_objects);
    
    // Free all the host data
    for (size_t zid = 0; zid < depth; ++zid)
    {
        free(coo_idx[zid]);
        free(coo_value[zid]);
    }
    free(plane_nnz);
    
    return num_objects;
}

void ObjectCloud::extractObjects(CuMat ccl_data)
{
    CHECK_FOR_ERROR("before ObjectCloud::exractObjects");
    if (num_objects == -1)
    {
        std::cout << "ObjectCloud::extractObjects Error: ";
        std::cout << "Must count number of objects first" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    size_t width = ccl_data.getWidth();
    size_t height = ccl_data.getHeight();
    size_t depth = ccl_data.getDepth();
    size_t numel = width * height * depth;
    
    for (int n = 0; n < allocated_objects; ++n)
        objects[n].setSourceSize(ccl_data.getWidth(), ccl_data.getHeight());
    
    size_t plane_size = width * height;
    uint32_t* plane_h = (uint32_t*)malloc(plane_size * sizeof(uint32_t));
    
    for (int zid = 0; zid < ccl_data.getDepth(); ++zid)
    {
        // Copy plane to host for processing
        uint32_t* plane_d = (uint32_t*)ccl_data.getCuData() + zid * plane_size;
        cudaMemcpy(plane_h, plane_d, plane_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        size_t start_idx = zid * plane_size;
        for (size_t pidx = 0; pidx < plane_size; ++pidx)
        {
            // If voxel is part of an object, add it to that Blob3d
            uint32_t obj_id = plane_h[pidx];
            if (obj_id < (uint32_t)-1)
            {
                uint32_t* id = std::find(object_ids, object_ids + num_objects, obj_id);
                size_t blob_idx = std::distance(object_ids, id);
                
                objects[blob_idx].addVoxel(start_idx + pidx);
            }
        }
    }
    
    free(plane_h);
    CHECK_FOR_ERROR("ObjectCloud::exractObjects");
    return;
}

void ObjectCloud::extractObjects(SparseVolume ccl_data)
{
    CHECK_FOR_ERROR("before ObjectCloud::extractObjects");
    if (num_objects == -1)
    {
        std::cout << "ObjectCloud::extractObjects Error: ";
        std::cout << "Must count number of objects first" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    size_t width = ccl_data.getWidth();
    size_t height = ccl_data.getHeight();
    size_t depth = ccl_data.getDepth();
    
    for (int n = 0; n < allocated_objects; ++n)
        objects[n].setSourceSize(width, height);
    
    size_t plane_size = width * height;
    
    // Sparse data format
    size_t** coo_idx;       // array of planes, indices to non-zero elements
    float2** coo_value;     // array of planes, values of non-zero elements
    size_t* plane_nnz;      // Number of non-zero elements per plane
    ccl_data.getHostData(&coo_idx, &coo_value, &plane_nnz, NULL);
    
    for (int zid = 0; zid < depth; ++zid)
    {
        size_t start_idx = zid * plane_size;
        for (int idx = 0; idx < plane_nnz[zid]; ++idx)
        {
            size_t this_val = *reinterpret_cast<size_t*>(&(coo_value[zid][idx]));
            uint32_t obj_id = this_val;
            uint32_t* id = std::find(object_ids, object_ids + num_objects, obj_id);
            size_t blob_idx = std::distance(object_ids, id);
            
            objects[blob_idx].addVoxel(start_idx + coo_idx[zid][idx]);
        }
    }
    
    for (size_t zid = 0; zid < depth; ++zid)
    {
        free(coo_idx[zid]);
        free(coo_value[zid]);
    }
    free(plane_nnz);
    
    CHECK_FOR_ERROR("ObjectCloud::exractObjects");
    return;
}

void ObjectCloud::includeIntensity(SparseVolume value_data)
{
    // Make sure objects have been extracted
    for (int i = 0; i < num_objects; ++i)
    {
        assert(objects[i].getNumVoxels() > 0);
    }
    
    for (int obj_id = 0; obj_id < num_objects; ++obj_id)
    {
        float intensity_sum = 0;
        for (int n = 0; n < objects[obj_id].getNumVoxels(); ++n)
        {
            cv::Point3i vox = objects[obj_id].getVoxel(n);
            float2 value = value_data.getValue(vox);
            float intensity = value.x*value.x + value.y*value.y;
            objects[obj_id].setValue(n, intensity);
            intensity_sum += intensity;
        }
    }
}

void ObjectCloud::allocateObjects(size_t count)
{
    objects = new Blob3d[count];
    allocated_objects = count;
    
    return;
}

void ObjectCloud::blobDepthFilter(int min_length)
{
    for (int n = 0; n < num_objects; ++n)
    {
        Blob3d obj = objects[n];
        if (obj.getLength() < min_length)
        {
            objects[n] = objects[num_objects-1];
            n--;
            num_objects--;
        }
    }
}

void ObjectCloud::blobSizeFilter(int min_size)
{
    int num_removed = 0;
    for (int n = 0; n < num_objects; ++n)
    {
        Blob3d obj = objects[n];
        if (obj.getNumVoxels() < min_size)
        {
            objects[n] = objects[num_objects-1];
            n--;
            num_objects--;
            num_removed++;
        }
    }
    printf("blobSizeFilter removed %d objects\n", num_removed);
}

void ObjectCloud::add(ObjectCloud cloud2)
{
    for (int n = 0; n < cloud2.getNumObjects(); ++n)
    {
        // Handle necessary expansion of allocated space
        if (num_objects+1 > allocated_objects)
        {
            // reallocate with twice as many objects
            size_t new_allocated_objects = allocated_objects;
            if (allocated_objects == 0)
                new_allocated_objects = 100;
            else
                new_allocated_objects = 2 * allocated_objects;
            
            Blob3d* temp_objects = objects;
            objects = new Blob3d[new_allocated_objects];
            for (int i = 0; i < allocated_objects; ++i)
            {
                objects[i] = temp_objects[i];
            }
            
            if (allocated_objects > 0) delete[] temp_objects;
            
            allocated_objects = new_allocated_objects;
        }
        
        objects[num_objects] = cloud2.objects[n];
        objects[num_objects].rebase(cv::Rect(0, 0, width, height));
        num_objects++;
    }
    
    return;
}

void ObjectCloud::prune(float dist)
{
    bool all_merged = false;
    
    if (dist <= 0)
    {
        // Compare each object to each other to see if they intersect
        for (int n1 = 0; n1 < num_objects; ++n1)
        {
            for (int n2 = n1+1; n2 < num_objects; ++n2)
            {
                if (objects[n1].intersects(objects[n2]))
                {
                    // If they intersect, merge them together and remove the 2nd
                    objects[n1].mergeIn(objects[n2]);
                    objects[n2] = objects[num_objects];
                    num_objects--;
                }
            }
        }
    }
    else
    {
        // Compare each object to each other to see if they intersect
        for (int n1 = 0; n1 < num_objects; ++n1)
        {
            for (int n2 = n1+1; n2 < num_objects; ++n2)
            {
                if (objects[n1].overlaps(objects[n2], dist))
                {
                    // If they intersect, merge them together and remove the 2nd
                    objects[n1].mergeIn(objects[n2]);
                    objects[n1].setSourceSize(this->width, this->height);
                    objects[n2] = objects[num_objects-1];
                    objects[n2].setSourceSize(this->width, this->height);
                    num_objects--;
                }
            }
        }
    }
    
    return;
}

void ObjectCloud::writeCentroids(char* filename, bool use_weighted_centroid, bool verbose)
{
    FILE* fid = fopen(filename, "w");
    if (fid == NULL)
    {
        std::cout << "ObjectCloud::writeCentroids: Error: Unable to open file: " << filename << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }
    
    // Write a header
    
    if (!verbose)
    {
        fprintf(fid, "x (pixels), y (pixels), z (planes), # of voxels\n");
    }
    else
    {
        fprintf(fid, "x (pixels), y (pixels), z (planes)");
        fprintf(fid, ", weighted x, weighted y, weighted z");
        fprintf(fid, ", # of voxels");
        fprintf(fid, ", max value");
        fprintf(fid, ", mean value");
        fprintf(fid, ", xmin, xmax, ymin, ymax, zmin, zmax");
        fprintf(fid, "\n");
    }
    
    for (int n = 0; n < num_objects; ++n)
    {
        cv::Point3f cent;
        if (use_weighted_centroid)
        {
            cent = objects[n].getWeightedCentroid();
        }
        else
            cent = objects[n].getCentroid();
        
        size_t num_voxels = objects[n].getNumVoxels();
        if (!verbose)
        {
            fprintf(fid, "%f, %f, %f, %d\n", cent.x, cent.y, cent.z, num_voxels);
        }
        else
        {
            cent = objects[n].getCentroid();
            cv::Point3f wcent = objects[n].getWeightedCentroid();
            fprintf(fid, "%f, %f, %f", cent.x, cent.y, cent.z);
            fprintf(fid, ", %f, %f, %f", wcent.x, wcent.y, wcent.z);
            fprintf(fid, ", %d", num_voxels);
            fprintf(fid, ", %e", objects[n].getMaxValue());
            fprintf(fid, ", %e", objects[n].getMeanValue());
            
            BoundingBox bounds = objects[n].getBounds();
            fprintf(fid, ", %d, %d, %d, %d, %d, %d",
                bounds.xmin, bounds.xmax,
                bounds.ymin, bounds.ymax,
                bounds.zmin, bounds.zmax);
            fprintf(fid, "\n");
        }
    }
    fclose(fid);
    
    return;
}

void ObjectCloud::offset(cv::Rect roi)
{
    cv::Point3f shift;
    shift.x = roi.x;
    shift.y = roi.y;
    shift.z = 0;
    
    for (int n = 0; n < num_objects; ++n)
    {
        objects[n].setSourceSize(roi.width, roi.height);
        objects[n].setGlobalOffset(shift);
    }
}

//============================= ACCESS     ===================================

Blob3d ObjectCloud::getObject(int id)
{
    if (id >= num_objects)
    {
        Blob3d obj;
        return obj;
    }
    
    return objects[id];
}

void ObjectCloud::addObject(Blob3d obj)
{
    if (allocated_objects > num_objects)
    {
        objects[num_objects] = obj;
        num_objects++;
    }
    else
    {
        std::cout << "Unable to add add object, not enough space" << std::endl;
        std::cout << "Already stored " << num_objects << " of allocated " << allocated_objects << std::endl;
        throw HOLO_ERROR_OUT_OF_MEMORY;
    }
    
    return;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
