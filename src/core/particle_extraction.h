#ifndef PARTICLE_EXTRACTION_H
#define PARTICLE_EXTRACTION_H

// SYSTEM INCLUDES
#include <numeric>
#include <limits>
#include <math.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "nppi.h"

// LOCAL INCLUDES
#include "umnholo.h"
#include "optical_field.h"
#include "deconvolution.h"
#include "object_cloud.h"
#include "point_cloud.h"
#include "holo_sequence.h"
#include "point_cloud_3d.h"
#include "holo_ui.h"

namespace umnholo {

    enum ExtractionState
    {
        EXTRACTION_STATE_NORMALIZED = 1,
        EXTRACTION_STATE_BINARY = 2,
        EXTRACTION_STATE_CC_LABELED = 3
    };
    
    enum ThresholdMethod
    {
        THRESHOLD_OTSU_CMB = 0,
        THRESHOLD_OTSU_VOLUME = 1
    };

    /** @brief Specific enhancement algorithm for use in ParticleExtraction **/
    class CV_EXPORTS TolouiSnrNormalize : public OpticalField
    {
    public:
        // LIFECYCLE

        /**
         * @brief Default constructor is intended for testing purposes only
         */
        TolouiSnrNormalize();

        /**
         * @brief See contructor for ParticleExtraction
         */
        TolouiSnrNormalize(Deconvolution& deconv);

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Find first threshold value using combined image
         * @param cmb Minimum intensity image for the volume
         */
        float findThreshold2d(CuMat cmb);

        /**
         * @brief Performs average filter on single plane
         * @param plane_idx Index of plane to filter
         */
        void averageFilter(int plane_idx);
        
        /**
         * @brief Performs average filter on entire volume
         *        Equivalent to running averageFilter for each plane
         */
        void averageFilterVolume();

        /**
         * @brief Threshold pixels setting all above the threshold value to white
         * @param value Threshold value for comparing pixels to
         */
        void threshold(float value);
        
        /**
         * @brief Replace the border where x or y are maximum with value
         *        This exists to match processing with matlab. The
         *        the related matlab step should be changed
         */
        void replaceFilterBorder(float value);

        /**
         * @brief Renormalize one region of the volume
         * @returns Modifies internal data
         */
        void normalizeBlocks();
        
        /**
         * @brief Determine the max and min of each block
         * @param minima_h Unallocated array where the min of each block
         *               will be stored. Storage pattern matches C
         *               convenction [x1y1z1,x2y1z1,y2x1z1,y2x2z1,x1y1z1,...]
         * @param maxima_h Same as minima_h but stores max of each block
         * @param num_blocks Output number of blocks used
         */
        void minMaxBlocks(float* minima_h, float* maxima_h, int &num_blocks);

        // ACCESS
        
        void setWindowSize(float win) { window_size = win; }
        
        // INQUIRY

    protected:
    private:
        float window_size;
        int filter_tile_size;
        int block_size;
        int threshold_tile_size;

    };
    
    /** @brief Can perform all steps to segment volume into particles **/
    class CV_EXPORTS ParticleExtraction : public OpticalField
    {
    public:
        // LIFECYCLE
        
        /**
         * @brief Default constructor for debugging. Should not be used.
         */
         ParticleExtraction() { state = -100; is_dilated = false; }

        /**
         * @brief Create object for extracting particles from a deconvolved 
         *        volume
         * @param deconv Pointer to deconvolved volume. State must be
         *               DECONVOLVED. Otherwise, throws exception. Changes
         *               state of deconv if modified.
         */
        ParticleExtraction(Deconvolution& deconv);

        /**
         * @brief Create object for extracting particles from a reconstructed
         *        volume
         * @param recon Pointer to reconstructed volume. State must be
         *               RECONSTRUCTED. Otherwise, throws exception.
         * @returns Garbage in recon. State is OPTICALFIELD_STATE_GARBAGE
         */
        ParticleExtraction(Reconstruction& recon);
        
        void destroy();
        
        /**
         * @brief Incomplete destruction for reusing data from input deconv
         */
        void destroy_iterative();

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Enhance SNR of volume data
         * @returns Modifies internal data
         */
        void enhance();
        
        /**
         * @brief Same as enance() with user specified window size
         *        window size changes the selected 2d threshold
         * @param window_size Multiplies the particle size to get true size
         */
        void enhance(float window_size);
        
        /**
         * @brief Binarize data setting all voxels above threshold to 1, 
         *        all below to 0
         * @param type Thresholding type, see OpenCV threshold documentation
         *        So far, options are THRESH_BINARY_INV, and THRESH_BINARY
         * @returns State set to EXTRACTION_STATE_BINARY
         */
         void binarize(int type = cv::THRESH_BINARY_INV);
        
        /**
         * @brief Binarize data setting all voxels above threshold to 1, 
         *        all below to 0
         * @param thr Theshold to use for binarization. Ignores stored value.
         * @param type Thresholding type, see OpenCV threshold documentation
         *        So far, options are THRESH_BINARY_INV, and THRESH_BINARY
         * @returns State set to EXTRACTION_STATE_BINARY
         */
         void binarize(double thr, int type = cv::THRESH_BINARY_INV);
         
         /**
          * @brief Morphological dilation of volume
          * @param size Defines structuring element for dilation as cube with
          *             radius size. Side length is 2*size + 1
          * @param State must be EXTRACTION_STATE_BINARY
          * @returns Changes internal data
          */
         void dilate(int size);
         
         /**
          * @brief Reverses effects of dilate such that non-zero/zero state
          *        is maintained for each voxel. Does not change values
          * @param zero_val Value to set zero voxels to
          * @returns Changes internal data. Does not change value of
          *          non-zero voxels that were non-zero before dilate.
          */
         void undoDilate(uint32_t zero_val);
         
         /**
         * @brief Returns the XY minimum intensity image of the 3D volume
         * @param Data must be of either NORMALIZED or BINARY state
         * @returns Result of min (normalized data) or max (binary) 
         *          in z direction for each XY pixel.
         */
        CuMat combinedXY();
        
        /**
         * @brief Identify and extract information of each object in field
         * @param State must be EXTRACTION_STATE_BINARY
         * @returns ObjectCloud class containing all pertinent information
         */
        ObjectCloud extractObjects();
        
        /**
         * @brief Simplistic centroid extraction method using reconstruction
         * @param recon Same object used for initialization, should be of state
         *      OPTICALFIELD_STAT_UNALLOCATED
         * @returns 3D centroids of objects in the reconstructed volume
         */
        PointCloud3d extractCentroids(Reconstruction& recon, Hologram& holo);
        
        /**
         * @brief Label using 26 neighbor CCL. Label is lowest index
         * @param State must be EXTRACTION_STATE_BINARY
         *        Volume size is limited to UINT32_MAX
         * @returns State is EXTRACTION_STATE_CCL_LABELED
         */
        void labelConnectedComponents();
        
        /**
         * @brief Compute threshold to use for binarization
         * @param method Technique to use when computing threshold
         *        Supported methods are: THRESHOLD_OTSU_CMB
         *                               THRESHOLD_OTSU_VOLUME
         * @return The value of the threshold
         */
        double computeThreshold(ThresholdMethod method);

        // ACCESS
        // INQUIRY

    protected:
    private:
        CuMat cmb;

        TolouiSnrNormalize norm;
        
        float threshold;
        
        bool is_dilated;
        CuMat undilated_data;
    };
    
    /** @brief 2D particle extraction using blob detection **/
    class CV_EXPORTS Extraction2D
    {
    public:
        // LIFECYCLE

        Extraction2D();

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Extract information from 2D objects in holograms
         * @param holos Sequence of enhanced holograms
         * @returns Locations of each object in hologram in 2D
         */
        PointCloud extractCentroids(HoloSequence holo);

        // ACCESS
        // INQUIRY

    protected:

        /**
         * @brief Calculate threshold as half peak of histogram
         *        Equivalent MATLAB expression: Thr = (mode(image(:)) - 1)/ 2
         * @param image Pointer to CV_8U image
         * @returns Value of the selected threshold
         */
        double calcThreshold(cv::Mat* image);

    private:
        cv::Size filter_size;
        cv::Point filter_center;
        double thresh_value;
        int opening_size;
        //cv::Mat opening_element;
    };

} // namespace umnholo

#endif // PARTICLE_EXTRACTION_H
