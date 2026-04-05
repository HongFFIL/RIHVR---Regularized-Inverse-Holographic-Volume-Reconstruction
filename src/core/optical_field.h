#ifndef OPTICAL_FIELD_H
#define OPTICAL_FIELD_H

// SYSTEM INCLUDES
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda_runtime_api.h"

// LOCAL INCLUDES
#include "umnholo.h"
#include "cumat.h"
#include "hologram.h"
#include "object_cloud.h"

namespace umnholo {

    enum OpticalFieldState
    {
        OPTICALFIELD_STATE_UNALLOCATED = -2,
        OPTICALFIELD_STATE_GARBAGE = -1,
        OPTICALFIELD_STATE_EMPTY = 0,
        OPTICALFIELD_STATE_RECONSTRUCTED = 1,
        OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX = 2,
        OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX_FOURIER = 3,
        OPTICALFIELD_STATE_PSF = 4,
        OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX_REAL = 5,
        OPTICALFIELD_STATE_DECONVOLVED = 6,
        OPTICALFIELD_STATE_DECONVOLVED_REAL = 7,
        OPTICALFIELD_STATE_FULL_COMPLEX = 8
    };

    enum OpticalFieldScale
    {
        SCALE_UNKNOWN = -1,
        SCALE_0_255 = 0,
        SCALE_0_1 = 1,
        SCALE_DECONV_NORM = 2
    };
    
    enum ComplexDataFormat
    {
        REAL_IMAGINARY = 0,
        AMPLITUDE_PHASE = 1
    };
    
    enum OpticalFieldAllocation
    {
        OPTICALFIELD_ALLOCATE = 0,
        OPTICALFIELD_DO_NOT_ALLOCATE = 1
    };

    /**
     * @brief Reconstructed 3D volume from a hologram
     */
    class CV_EXPORTS OpticalField : public cv::Algorithm
    {
    public:
        // LIFECYCLE

        /**
         * @brief Default constructor initializes parameters to 0 and nothing more
         */
        OpticalField();

        /** 
         * @brief Use Hologram to initialize, does not reconstruct yet
         */
        OpticalField(Hologram holo, OpticalFieldAllocation allocation_method = OPTICALFIELD_ALLOCATE);
        
        /**
         * @brief Initialize parameters and volume size
         */
        OpticalField(Parameters params, size_t width, size_t height, size_t depth);

        /**
         * @brief Calls destroy for internal CuMat data
         */
        void destroy();

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Returns the XY min or max intensity image of the 3D volume
         * @param method Identifies whether max (MAX_CMB) or min (MIN_CMB)
         *        intensity projection should be used
         * @returns Result of min in z direction for each XY pixel. Returned as
         *          a CuMat containing a CV_8U Mat that can be saved with imwrite
         */
        CuMat combinedXY(CmbMethod method = MIN_CMB);
        
        /**
         * @brief Calculate the mean and standard deviation of the volume
         * @param Volume must be of type CV_32F
         * @returns mean Mean of all elements in volume
         * @returns std Standard deviation of all elements in volume
         */
        void calcMeanStd(float &mean, float &std);
        
        /**
         * @brief Calculate the sum of the volume elements
         * @param Volume must be of type CV_32FC2
         * @returns sum Sum of all elements in volume
         */
        void calcSum(float2 &sum);
        
        /**
         * @brief Calculate the sum of the complex magnitudes squared
         *        This is the same as the L2-norm squared
         * @param Volume must be of type CV_32FC2
         *        Non-complex data can be treated as complex through a 
         *        type reinterpretation and will give the desired result
         * @returns sumsqr Result is sum of complex magnitude squared
         */
        void calcSumMagnitude(float &sumsqr);
        
        /**
         * @brief Calculate the L1-norm (sum absolute value)
         * @param Volume must be of type CV_32FC2
         * @returns sumsqr Result is sum of complex magnitude
         */
        void calcL1Norm(float &sumsqr);
        
        /**
         * @brief Calculate the L1-norm (sum absolute value)
         * @param Volume must be of type CV_32FC2
         * @returns sumsqr Result is sum of abs of complex components
         */
        void calcL1NormComplex(float &sumsqr);
        
        /**
         * @brief Count the number of non-zero elements
         * @param eps Epsilon value used to determine equality to zero
         * @returns Integer number of elements which are not identically zero
         */
        size_t countNonZeros() { return this->data.countNonZeros(); } 
        
        /**
         * @brief Compute the Shannon entropy of the full volume
         *        Entropy computed using 256 bins from min to max
         * @param Data must be real valued, suggest calling 
         *        OpticalField::convertComplexRepresentationTo(AMPLITUDE_PHASE)
         *        and OpticalField::makeReal() before this function
         * @returns Entropy value
         */
        double calcEntropy();
        
        /**
         * @brief Computes the histogram of the full volume
         *        Uses 256 bins from min to max
         * @param Data must be real valued, suggest calling 
         *        OpticalField::convertComplexRepresentationTo(AMPLITUDE_PHASE)
         *        and OpticalField::makeReal() before this function
         * @param bins_h and hist_h are unallocated host pointers
         * @return bins_h Length 257 array of bin edges
         * @return hist_h Length 256 arrary of bin counts
         */
        void calcHistogram(Npp32f* bins_h, Npp32s* hist_h);

        /**
         * @brief Opens a window and allows the user to step throught the 
         *        reconstructed volume by pressing a key to increment z
         */
        void view();
        
        /**
         * @brief Writes each plane in the volume as a tiff file
         *        Filename is formated [output_path]/plane_%04d.tif
         */
        void save();
        
        /**
         * @brief Writes each plane in the volume as a tiff file
         * @param prefix Prefix to be appended to filename. Format will
         *               be [output_path]/[prefix]plane_%04d.tif
         * @param State must be OPTICALFIELD_STATE_FULL_COMPLEX,  
         *        DECONVOLVED, RECONSTRUCTED, or DECONVOLVED_REAL
         */
        void save(char* prefix);
        
        /**
         * @brief Writes position and intensity of each voxel
         *        Output file is csv with columns x, y, z, value
         * @param prefix Prefix to be appended to filename. Fromat will 
         *        be [output_path]/[prefix].csv
         * @param Data must be real
         */
        void saveSparse(char* prefix);
        
        /**
         * @brief Writes XY, YZ, and XZ max projections to tiff files
         * @param prefix Prefix to be appended to filename. Format will
         *               be [output_path]/[prefix]_[xy|xz|yz].tif
         * @param method Indicates method for taking projection (max or min)
         * @param State must be OPTICALFIELD_STATE_FULL_COMPLEX,  
         *        DECONVOLVED, RECONSTRUCTED, or DECONVOLVED_REAL
         */
        void saveProjections(char* prefix, CmbMethod method = MAX_CMB);

        /**
         * @brief Normalization of volume
         *        Currently only implemented conversion from SCALE_DECONV_NORM
         *        to SCALE_0_255 which undoes the hologram mean0 normalization
         * @param new_scale Scale to convert to. Only acceptable option is 
         *                  SCALE_0_255
         * @returns Modifies internal data and changes scale
         */
        void renormalize(OpticalFieldScale new_scale);
        
        /**
         * @brief Scales image to the requested range
         * @param new_scale Scale to convert to. Options are SCALE_0_1
         *                  and SCALE_0_255. Current state must also be
         *                  one of these optiosn
         * @returns Changes internal data so that the values are within
         *          the requested range. Changes state.
         */
        void scaleImage(OpticalFieldScale new_scale);
        
        /**
         * @brief Rounds data to nearest integer
         *        Original purpose is for replicating result of saving
         *        and reading image (loss of precision)
         * @param Requires scale to be SCALE_0_255
         * @return Changes data. No change to designated state
         */
        void round();
        
        void trashState() { state = OPTICALFIELD_STATE_GARBAGE; }
        
        /**
         * @brief Convert between the two representations of complex data
         * @param rep Complex representation to be used. May be one of
         *        AMPLITUDE_PHASE or REAL_IMAGINARY
         * @return Changes internal complex_representation 
         */
        void convertComplexRepresentationTo(ComplexDataFormat rep);
        
        void makeReal() { data.makeReal(); }
        
        void fillField(ObjectCloud cloud);

        // ACCESS

        /**
         * @brief Access a single plane from the volume
         * @param plane_idx Index value for the plane. Allowable range is 0 to 
         *                  num_planes. Other values will throw error
         * @returns CuMat object containing the plane data
         */
        CuMat getPlane(size_t plane_idx);
        
        /**
         * @brief Pull 3D region of interest out of larger volume
         * @param x X coordinate of top left front corner of roi
         * @param y Y coordinate of top left front corner of roi
         * @param z Z coordinate of top left front corner of roi
         * @param width Width (x dimension) of roi in pixels
         * @param heigh Height (y dimension) of roi in pixels
         * @param depth Depth (z dimension) of roi in pixels
         * @return New CuMat with unlinked data
         */
        CuMat getRoi(int x, int y, int z, int width, int height, int depth);

        Parameters getParams() { return this->params; }

        void setParams(Parameters p) { params = p; }
        
        int getWidth() { return this->width; }
        
        int getHeight() { return this->height; }
        
        int getDepth() { return this->depth; }
        
        void getScaleMinMax(double* min_val, double* max_val)
            { *min_val=scale_min; *max_val=scale_max; }

        /**
         * @brief Debugging utility. Manually set data
         * @param data Filled data matrix
         */
        void setData(CuMat data);
        
        /**
         * @brief Debugging utility. Only methods should access data
         */
        CuMat getData() { return data; }
        
        /**
         * @brief Debugging utility. Only methods should modify state
         */
         void setState(int s) { state = s; }
         
         int getState() { return state; }

        // INQUIRY

        /**
        * @brief Compare the state of the data to the user input
        */
        bool isState(int s) const { return state == s; }

    protected:
        Parameters params;
        CuMat data;

        int state;
        int width;
        int height;
        int depth;

        float hologram_original_mean;
        
        double scale_min;
        double scale_max;

        OpticalFieldScale scale;
        ComplexDataFormat complex_representation;
    private:
    };

} // namespace umnholo

#endif // OPTICAL_FIELD_H
