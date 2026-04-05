#ifndef BLOB3D_H
#define BLOB3D_H

// SYSTEM INCLUDES
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"

// LOCAL INCLUDES
#include "cumat.h"
#include "umnholo.h"

namespace umnholo {
    
    class BoundingBox
    {
    public:
        int xmin;
        int xmax;
        int ymin;
        int ymax;
        int zmin;
        int zmax;
        bool is_set;
        
        /**
         * @brief Returns true if the boxes touch or overlap
         */
        bool intersect(BoundingBox bb2);
        
        /**
         * @brief Expands the bounding box by dist in all directions
         */
        void expand(int dist);
    };
    
    /** 
     * @brief 3D object defined by voxels, arbitrary shape
     */
    class CV_EXPORTS Blob3d
    {
    public:
        // LIFECYCLE
        
        Blob3d();
        
        // OPERATORS
        // OPERATIONS
        
        /**
         * @brief Include a voxel in the blob
         * @param global_idx Volume index to the voxel to be added
         */
        void addVoxel(size_t global_idx);
        
        /**
         * @brief Include a voxel in the blob
         * @param point Structure with x, y, z fields, in global coords
         */
        void addVoxel(cv::Point3i point);
        
        /**
         * @brief Determines bounding box of object
         *        Does nothing if blob is empty
         */
        void findBounds();
        
        /**
         * @brief Merge the 2nd blob into this.
         *        The result is the union of the blobs
         */
        void mergeIn(Blob3d obj2);
        
        /**
         * @brief Adjusts source data to account for shift to new source
         * @param new_source x and y used to set global_offset, width and
         *                   height used as source width/height respectively
         */
        void rebase(cv::Rect new_source);
        
        // ACCESS
        
        /**
         * @brief Initialize blob to accept data from a source volume
         * @param src_width Width of source 3d volume
         * @param src_height Height of source 3d volume
         */
        void setSourceSize(size_t src_width, size_t src_height);
        
        /**
         * @brief Allows for use of a global coordinate system offset
         *        from the default (i.e. region near global origin may
         *        be free of objects)
         * @param offset Shift applied to reported locations to convert 
         *               to global coordinate system
         */
        void setGlobalOffset(cv::Point3f offset) { global_offset = offset; }
        
        size_t getNumVoxels() { return num_voxels; }
        
        /**
         * @brief Get the coordinates or a particular voxel in the blob
         *        Coordinates are within the specified source size
         * @param id Voxel index range 0 to num_voxels-1
         *           If id is outside allowable range, returns all 0
         * @return Structure with x, y, z fields
         */
        cv::Point3i getVoxelLocal(size_t id);
        
        /**
         * @brief Alias for getVoxelLocal
         */
        cv::Point3i getVoxel(size_t id) { return getVoxelLocal(id); }
        
        /**
         * @brief Get the coordinates or a particular voxel in the blob
         *        Coordinates are relative to a a global origin
         * @param id Voxel index range 0 to num_voxels-1
         *           If id is outside allowable range, returns all 0
         * @return Structure with x, y, z fields
         */
        cv::Point3i getVoxelGlobal(size_t id);
        
        /**
         * @brief Sets the value value of a blob voxel
         * @param id Voxel index in range 0 to num_voxels-1
         *        assertion fails if index out of range
         * @return Modifies internal data
         */
        void setValue(size_t id, float value);
        
        /**
         * @brief Returns centroid of object
         *        Centroid calculated as mean of x, y, and z of voxels
         */
        cv::Point3f getCentroid();
        
        /**
         * @brief Returns intensity-weighted centroid of object
         *        Values must have been included with setValue(). Otherwise,
         *        will use non-weighted centroid.
         * @returns Centroid calculated as weighted mean of x, y, and z of voxels
         */
        cv::Point3f getWeightedCentroid();
        
        /**
         * @brief Returns the plane where the object is in focus
         *        Determined as the midpoint of the z extrema
         */
        float getFocusZ();
        
        /**
         * @brief Binary mask of object focus plane
         * @returns 2d binary uint8 array of same size as source 3d volume
         *          Pixels occluded by the object are 1, others are 0
         */
        CuMat getFocusMask();
        
        /**
         * @brief Returns length of the blob in voxels
         *        Computed as zmax - zmin using bounding box
         */
        int getLength();
        
        /**
         * @brief Returns the box bounding the blob on each side
         */
        BoundingBox getBounds();
        
        /**
         * @brief Returns the maximum value of the voxels in the blob
         */
        float getMaxValue();
        
        /**
         * @brief Returns the arithmetic mean value of the voxels in the blob
         */
        float getMeanValue();
        
        // INQUIRY
        
        /**
         * @brief Returns true if the two objects share voxels
         */
        bool intersects(Blob3d obj2);
        
        /**
         * @brief Returns true if the two objects are within a given distance
         * @param dist Buffer distance applied to object before testing overlap
         *             If dist=0, equivalent to intersects.
         *             If dist < 0, treats as dist = 0
         */
        bool overlaps(Blob3d obj2, float dist);
        
    protected:
    private:
        // Counter of stored voxels for array access
        size_t num_voxels;
        
        // Array of voxels contained in the blob
        // Voxels are stored as the index into the source 3d volume
        std::vector<size_t>  voxels;
        
        // Array of values for the voxels contained in the blob
        // Usage of this is optional, only allocated if setValue() is called
        std::vector<float>  values;
        
        // Dimensions of source 3d volume for converting from index to xyz
        size_t source_width;
        size_t source_height;
        
        cv::Point3f centroid;
        bool centroid_updated;
        cv::Point3f weighted_centroid;
        bool weighted_centroid_updated;
        
        cv::Point3f global_offset;
        
        BoundingBox bounds;
    };
    
} // namespace umnholo

#endif // BLOB3D_H
