#ifndef SPARSE_SEGMENTATION_H
#define SPARSE_SEGMENTATION_H

// SYSTEM INCLUDES

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"

// LOCAL INCLUDES
#include "sparse_volume.h"
#include "object_cloud.h"


namespace umnholo {
    
    /**
     * @brief Segmentation and centroid extraction for sparse volumes
     */
    class CV_EXPORTS SparseSegmentation
    {
    public:
        // LIFECYCLE
        
        /**
         * @brief Initialize using the volume to be segmented
         */
        SparseSegmentation(SparseVolume input_data);
        
        // OPERATORS
        // OPERATIONS
        
        /**
         * @brief Creates the binarized version of the data
         * @returns Internal data modified
         */
        void binarize();
        
        /**
         * @brief Creates the binarized version of the working data
         * @returns Internal data modified
         */
        void binarizeWorking();
        
        /**
         * @brief Morphological closing on the binarized data
         * @returns Modifies internal binary data
         */
        void close();
        
        /**
         * @brief Simple morphological operations (dilation and erosion)
         * @param morph_type Type of morphological operation to perform
         *      only MORPH_ERODE and MORPH_DILATE are supported
         * @returns Modifies internal binary data
         */
        void simpleMorph(cv::MorphTypes morph_type);
        
        /**
         * @brief Label using 26 neighbor CCL. Label is lowest index
         * @param Volume must be binarized
         * @returns Modifies internal binary data
         */
        void labelConnectedComponents();
        
        /**
         * @brief Performs peak detection using the h-maxima transform
         * @param h Minimum peak height
         * @returns Modifies internal working data to remove small peaks
         */
        void hExtendedMaxima(double h);
        
        /**
         * @brief Identifies the individual objects and returns them
         * @param Volume must have been connected component labeled
         * @returns ObjectCloud containing all segmented objects
         */
        ObjectCloud extractObjects();
        
        // ACCESS
        // INQUIRY

    protected:
    private:
        
        Parameters params;
        
        size_t width;
        size_t height;
        size_t num_planes;
        
        SparseVolume complex_data;
        SparseVolume binary_data;
        SparseVolume working_data;
        
        CuMat complex_plane;
        CuMat binary_plane;
        
        bool is_binarized;
        bool is_labeled;
        
    };
} // namespace umnholo

#endif // SPARSE_SEGMENTATION_H
