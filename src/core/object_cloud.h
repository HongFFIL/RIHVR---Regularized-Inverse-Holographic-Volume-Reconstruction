#ifndef OBJECT_CLOUD_H
#define OBJECT_CLOUD_H

// SYSTEM INCLUDES
#include <iostream>
#include <algorithm>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"

// LOCAL INCLUDES
#include "cumat.h"
#include "blob3d.h"
#include "sparse_volume.h"

namespace umnholo {
    
    // Forward declaration to remove circular refs
    class SparseVolume;
    
    const size_t max_num_objects = 20*1024;
    
    /** 
     * @brief Arbitrary number of objects extracted from holograms
     */
    class CV_EXPORTS ObjectCloud
    {
    public:
        // LIFECYCLE
        
        /**
         * @brief Default constructor should only be needed for debugging
         */
         ObjectCloud();
        
        /**
         * @brief Initialize with enough space for count objects
         */
         ObjectCloud(size_t count);
        
        /**
         * @brief Extract cloud from connected component labeled volume
         * @param ccl_data Must be labeled data (untested assumption). 
         *                 Data must be convertable to uint32_t
         * @returns Cloud of objects corresponding to each unique label
         */
        ObjectCloud(CuMat ccl_data);
        
        /**
         * @brief Extract cloud from connected component labeled volume
         * @param ccl_data Must be labeled data (untested assumption).
         * @returns Cloud of objects corresponding to each unique label
         */
        ObjectCloud(SparseVolume ccl_data);
        
        /**
         * @brief Frees all allocated memory (host and device)
         */
        void destroy();
        
        // OPERATORS
        // OPERATIONS
        
        /**
         * @brief Counts the number of unique connected components in labeled
         *        data. Also stores the label for each object internally 
         *        for future extraction
         * @param ccl_data Must be labeled data (untested assumption). 
         *                 Data must be convertable to uint32_t
         * @returns Number of objects, modifies internal data
         */
        size_t countConnectedObjects(CuMat ccl_data);
        
        /**
         * @brief Counts the number of unique connected components in labeled
         *        data. Also stores the label for each object internally 
         *        for future extraction
         * @param ccl_data Must be labeled data (untested assumption).
         * @returns Number of objects, modifies internal data
         */
        size_t countConnectedObjects(SparseVolume ccl_data);
        
        /**
         * @brief Create sparse representation of objects in field
         * @param ccl_data Must be labeled data (untested assumption). 
         *                 Data must be convertable to uint32_t
         * @param Objects must have been counted using countConnectedObjects
         * @return Stores results internally
         */
        void extractObjects(CuMat ccl_data);
        
        /**
         * @brief Create sparse representation of objects in field
         * @param ccl_data Must be labeled data (untested assumption).
         * @param Objects must have been counted using countConnectedObjects
         * @return Stores results internally
         */
        void extractObjects(SparseVolume ccl_data);
        
        /**
         * @brief Include voxel intensities with the objects
         * @param value_data Complex-valued data (from reconstruction)
         * @param Objects must have been extracted using extractObjects
         * @return Stores results internally
         */
        void includeIntensity(SparseVolume value_data);
        
        /**
         * @brief Allocates space for as many objects as necessary
         * @param count Number of objects to allocate space for
         */
        void allocateObjects(size_t count);
        
        /**
         * @brief Removes all objects that are shorter than a given z
         * @param min_length Minimum length of objects to be left in the 
         *                   cloud. Test is inclusive.
         */
        void blobDepthFilter(int min_length);
        
        /**
         * @brief Removes all objects with small size (number of voxels)
         * @param min_size Minimum number of voxels of object to be left in
         *      the cloud. Test is inclusive (vox < min is removed).
         */
        void blobSizeFilter(int min_size);
        
        /**
         * @brief Adds all objects from cloud2 to this
         *        Does not perform any checks for overlapping objects
         * @param cloud2 ObjectCloud containing arbitrary number of objects
         */
        void add(ObjectCloud cloud2);
        
        /**
         * @brief Tests for overlapping objects. Any that intersect will
         *        be merged together.
         * @param dist Objects within dist of each other will be merged
         *             Dist must be >0 or will be treated as 0
         */
        void prune(float dist = 0.0);
        
        /**
         * @brief Write all centroids to file in a csv format
         *        Rows are objects and columns are x, y, z
         * @param filename Absolute or relative path to file to write to
         * @param use_weighted_centroid Indicates whether to use intensity-
         *      weighted centroid or standard (equal weights). Must have
         *      included intensity data explicitely with includeIntensity()
         * @param verbose Indicates whether to include a large number of
         *      additional output columns. This is for exporatory purposes and
         *      extreme caution should be used if attempting to read this file
         *      in another program.
         */
        void writeCentroids(
                char* filename,
                bool use_weighted_centroid = false,
                bool verbose = false);
        
        /**
         * @brief Shift reported x and y values to indicate that the objects
         *        are all located in the given region of interest.
         *        Used when processing in sections and adding to a whole
         * @param roi Region of interest in which objects are located
         */
        void offset(cv::Rect roi);
        
        // ACCESS
        
        size_t getNumObjects() { return num_objects; }
        
        /**
         * @brief Accessor for objects in the cloud
         * @param id Index value into object vector [0, num_objects)
         * @returns Blob3d object data
         *          If id is greater than num_objects, return value will
         *          be result of default constructor
         */
        Blob3d getObject(int id);
        
        /**
         * @brief Add an object to the cloud. 
         * @param obj Blob3d object to be added to the cloud. obj will be 
         *            stored in index num_objects
         * @returns Increases num_objects by one
         */
        void addObject(Blob3d obj);
        
        /**
         * @brief Set the boundaries of objects in the field
         * @param w Width of field (voxels)
         * @param h Height of field (voxels)
         */
        void setSize(size_t w, size_t h) { width = w; height = h; }
        
        // INQUIRY

    protected:
    private:
        size_t num_objects;
        size_t allocated_objects;
        uint32_t* object_ids;
        size_t size_object_ids;
        Blob3d* objects;
        //std::vector<uint32_t> object_ids;
        //std::vector<Blob3d> objects;
        size_t width;
        size_t height;
    };
    
} // namespace umnholo

#endif // OBJECT_CLOUD_H
