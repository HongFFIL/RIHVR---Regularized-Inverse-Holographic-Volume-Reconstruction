#ifndef SPARSE_VOLUME_H
#define SPARSE_VOLUME_H

// SYSTEM INCLUDES
#include <vector>
#include <math.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"

// LOCAL INCLUDES
#include "umnholo.h"
#include "holo_ui.h"
#include "hologram.h"
#include "cumat.h"

namespace umnholo {
    
    // Forward declaration to remove circular refs
    class Hologram;
    
    struct CV_EXPORTS Voxel
    {
        double x;
        double y;
        float2 value;
    };

    /** @brief Sparse storage structure for reconstructed volume
     */
    class CV_EXPORTS SparseVolume
    {
    public:
    // LIFECYCLE
    
    /**
     * @brief Trivial constructor initializing to zero
     */
    SparseVolume();
    
    /**
     * @brief Initializes size parameters from hologram
     * @param holo Sample hologram to be processed
     * @param subsampling Plane index increment for subsampling
     */
    void initialize(Hologram holo, int subsampling = 1);
    
    /**
     * @brief Initializes size parameters from another volume
     *      Does not copy any data
     * @param volume Example volume of same size as this is to be initialized
     */
    void initialize(SparseVolume volume);
    
    /**
     * @brief Returns to state after initialize, all zeros
     */
    void erase();
    
    void destroy();
    
    // OPERATORS
    
    SparseVolume& operator=(const SparseVolume& other);
    
    // OPERATIONS
    
    void calcL1NormComplex(double* norm);
    
    void calcTVNorm(double* norm);
    
    void calcTVNorm2d(double* norm);
    
    size_t countNonZeros();
    
    /**
     * @brief Returns the maximum complex intensity of the data
     */
    double calcMaxIntensity();
    
    void savePlaneNonZeros(char* prefix);
    
    void saveData(char* prefix);
    
    /**
     * @brief Load the SparseVolume data previously saved with saveData
     *      Requires initialize() to have been called first
     */
    void loadData(char* filename);
        
    /**
     * @brief Writes XY, YZ, and XZ max projections to tiff files
     * @param prefix Prefix to be appended to filename. Format will
     *        be [output_path]/[prefix]_[xy|xz|yz].tif
     * @param method Indicates method for taking projection (max or min)
     *        This is included only to match the OpticalField method.
     */
    void saveProjections(char* prefix, CmbMethod method = MAX_CMB, bool prefix_as_suffix = false);
    
    /**
     * @brief Calculates and returns XY, XZ, and/or XZ max magnitude projections
     * @param xycmb Returned XY projection. Allocated or resized if necessary
     *        Output ignored if NULL
     * @param xzcmb Returned XZ projection. Allocated or resized if necessary
     *        Output ignored if NULL
     * @param yzcmb Returned YZ projection. Allocated or resized if necessary
     *        Output ignored if NULL
     * @param method Indicates method for taking projection (max or min)
     *        This is included only to match the OpticalField method.
     */
    void calcProjections(
        CuMat* xycmb,
        CuMat* xzcmb = NULL,
        CuMat* yzcmb = NULL,
        CmbMethod method = MAX_CMB);
    
    // ACCESS
    
    size_t getNumPlanes() { return num_planes; }
    
    double getSparsity();
    
    size_t getPlaneNnz(size_t plane_idx);
    
    /*
     * @brief Returns a single 2D plane of the volume
     * @param plane Returned plane data, will overwrite any old data
     *        Size is equal to the size of the hologram
     * @param plane_idx Identifier for which plane to get. Less than getNumPlanes.
     */
    void getPlane(CuMat* plane, size_t plane_idx);
    
    /*
     * @brief Option for erasing plane data from sparse CuMat
     *      Must be paired with getPlane on the same plane_idx, no checking is
     *      used to confirm the result
     *      Intended as alternative to cudaMemSet to 0 to save time
     * @param plane Data previously used with getPlane
     * @param plane_idx See getPlane
     * @returns plane Data is all zero
     */
    void unGetPlane(CuMat* plane, size_t plane_idx);
    
    void setPlane(CuMat plane, size_t plane_idx);
    
    /*
     * @brief Returns a single 2D plane of the volume, interpolating if necessary
     * @param plane Returned plane data, will overwrite any old data
     *        Size is equal to the size of the hologram
     *        Linear interpolation will be used if plane subsampling is used
     * @param plane_idx Identifier for which plane to get. Less than getNumPlanes.
     */
    void getPlaneInterpolated(CuMat* plane, size_t plane_idx);
    
    /*
     * @brief Indicates only planes 0:subsampling:depth are modified
     *      If accessing a different plane, the data will be interpolated
     */
    void setPlaneSubsampling(int subsampling);
    
    int getPlaneSubsampling() { return plane_subsampling; }
    
    Parameters getParams() { return params; }
    
    size_t getWidth() { return width; }
    size_t getHeight() { return height; }
    size_t getDepth() { return depth; }
    
    /**
     * @brief Copies sparse representation to host memory
     *      Assumes that all arrays are unallocated
     *      Pass NULL pointer to indicate arguments to ignore
     */
    void getHostData(size_t*** coo_idx_h,
                     float2*** coo_value_h,
                     size_t** plane_nnz_h,
                     size_t** plane_allocated_h);
    
    /**
     * @brief Copies sparse representation from host memory
     *      Assumes arrays haven't changed size since getHostData
     *      Pass NULL pointer to indicate arguments to ignore
     */
    void setHostData(size_t*** coo_idx_h,
                     float2*** coo_value_h,
                     size_t** plane_nnz_h,
                     size_t** plane_allocated_h);
    
    /**
     * @brief Gets the value of a prescribed voxel in the volume
     * @param pos Location of the target voxel
     * @returns Complex value
     */
    float2 getValue(cv::Point3i pos);
    
    // INQUIRY

    protected:
    private:
        
        Parameters params;
        
        // Data is organized as vector of planes
        // Units are voxels
        //std::vector<std::vector<Voxel> > data;
        //std::vector<size_t> plane_list;
        
        size_t** coo_idx;       // array of planes, indices to non-zero elements
        float2** coo_value;     // array of planes, values of non-zero elements
        size_t* plane_nnz;      // Number of non-zero elements per plane
        size_t* plane_allocated;// Allocated device data per plane
        
        // Global dimensions equal to user input number of planes and size of image
        size_t width;
        size_t height;
        size_t depth;
        //size_t num_voxels;
        
        // Used for scaling volume to require fewer voxels
        // TODO: This functionality has not yet been implemented
        size_t num_planes;
        cv::Point3f voxel_size;
        int plane_subsampling;
        bool ignore_subsample_checks;
        
        // buffer for getPlane and setPlane
        //float2* plane_data_h;
        
        // Small buffer (~64 bytes) for sums and other tasks
        void* buffer_64_d;
        bool is_allocated_buffer_64_d;
    };
} // namespace umnholo

#endif // SPARSE_VOLUME_H