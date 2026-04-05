#ifndef HOLOGRAM_H
#define HOLOGRAM_H

// SYSTEM INCLUDES
#define _USE_MATH_DEFINES
#include <math.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include "cufft.h"
#include "npp.h"

// LOCAL INCLUDES
#include "umnholo.h"
#include "holo_ui.h"
#include "cumat.h"
#include "object_cloud.h"
#include "blob3d.h"
//#include "reconstruction.h"

namespace umnholo {
    
    // Forward declaration to remove circular refs
    class ObjectCloud;
    
    enum HologramState
    {
        HOLOGRAM_STATE_EMPTY = 0,
        HOLOGRAM_STATE_LOADED = 1,
        HOLOGRAM_STATE_NORM_MEAND = 2,
        HOLOGRAM_STATE_FOURIER = 3,
        HOLOGRAM_STATE_FOURIER_SHIFTED = 4,
        HOLOGRAM_STATE_NORM_MEAND_ZERO = 5,
        HOLOGRAM_STATE_NORM_INVERSE = 6,
        HOLOGRAM_STATE_RECONSTRUCTED = 7,
        HOLOGRAM_STATE_RESIDUAL = 8,
        HOLOGRAM_STATE_ARBITRARY = 9
    };

    enum HologramScale
    {
        HOLOGRAM_SCALE_0_1 = 0,
        HOLOGRAM_SCALE_0_255 = 1,
        HOLOGRAM_SCALE_OTHER = 3
    };

    enum WindowMethod
    {
        WINDOW_TUKEY = 1,
        WINDOW_GAUSSSIAN = 2
    };
    
    const int removal_dilation = 2;
    const int num_removal_iterations = 4;

    /**
     * @brief Base class for holographic images
     */
    class CV_EXPORTS Hologram : public cv::Algorithm
    {
    public:
        // LIFECYCLE

        /**
         * @brief Default constructor must be combined with init or default
         *        reconstruction parameters will be used without warning.
         */
        Hologram();

        /** 
         * @brief Use user parameters to initialize
         */
        Hologram(const Parameters in);

        /**
         * @brief Replicates behavior of non-default constructor
         */
        void init(const Parameters in);

        /**
         * @brief Deallocate all data
         */
        void destroy();

        // OPERATORS
        
        /**
         * @brief Copy all data, no linking
         */
        void copyTo(Hologram* output);
        
        // OPERATIONS

        /**
         * @brief Reads hologram from file
         * @param filename Valid filename of image. May be any format that can
         *                 be read by OpenCV imread function. Must be cstring.
         * @returns Stores image internally. Changes state to indicate image 
         *          has been loaded. Throws exception if unable to read file
         */
        void read(char* filename);
        

        /**
         * @brief Reads hologram from file
         * @param filename Valid filename of image. May be any format that can
         *                 be read by OpenCV imread function. Must be cstring.
         * @param output_size Image is scaled to match the desired output size.
         *        The result is mean-padded if the aspect ratio changes
         * @returns Stores image internally. Changes state to indicate image 
         *          has been loaded. Throws exception if unable to read file
         *          Returns the scale ratio output/input
         */
        double read(char* filename, cv::Size output_size);

        /**
         * @brief Loads hologram data from non-image file
         * @param filename File to read data from. Format must match that used
         *                 by CuMat load method.
         * @param format Format of file loaded. See CuMat::load for details
         * @param state Force state of resultant hologram. No checking is done
         *              to confirm that state is accurate
         */
        void load(char* filename, CuMatFileMode format, HologramState state);

        /**
         * @brief Normalize the hologram
         * @returns Modifies data by dividing by the mean. Changes state
         */
        void normalize();

        /**
         * @brief Normalize the hologram and invert it
         * @returns Modifies data by dividing by the mean and subtracting from
         *          one. Mathematically: result = 1 - data/mean(data)
         *          Changes state.
         */
        void normalize_inverse();

        /**
         * @brief Normalize the hologram by dividing by mean and subtracting 1
         * @returns Modifies data, result has mean of 0
         */
        void normalize_mean0();
        
        /**
         * @brief Reverses the effects of nomalization functions
         * @param State must be one of the following:
         *          HOLOGRAM_STATE_NORM_MEAND
         *          HOLOGRAM_STATE_NORM_MEAND_ZERO
         *          HOLOGRAM_STATE_NORM_INVERSE
         * @returns state is HOLOGRAM_STATE_LOADED
         */
        void reverse_normalize();

        /**
         * @brief Scales data to change range of possible values
         * @param new_scale Must be either HOLOGRAM_SCALE_0_255 or 
         *                  HOLOGRAM_SCALE_0_1 for scaling corresponding to 
         *                  uint8 and double images respectively. Scale of
         *                  calling object must be opposite or thows HoloError
         * @returns Modifies data, changes scale to new_scale
         */
        void rescale(HologramScale new_scale);
        
        /**
         * @brief Converts complex valued data into real via magnitude
         *        Real component is complex magnitude, imaginary is all zero
         * @returns Modifies internal data
         */
        void mag();
        
        /**
         * @brief Rescales image to match given mean and standard deviation
         *        Duplicates behavior of Mostafa's Contrast_Recovery
         *        matlab function
         * @param mean Target mean as returned by cv::meanStdDev
         * @param std Target standard deviation as returned by cv::meanStdDev
         * @returns Modifies internal data
         */
         void matchMeanStd(cv::Scalar mean, cv::Scalar std);
         
        /**
         * @brief Rounds data to nearest integer
         *        Original purpose is for replicating result of saving
         *        and reading image (loss of precision)
         * @param Requires scale to be HOLOGRAM_SCALE_0_255
         * @return Changes data. No change to designated state
         */
         void round();

        /**
         * @brief Performs Fourier transform on hologram data
         * @returns Internal data changed to frequency domain. Zero frequency
         *          is in the corners (unshifted). State changed
         */
        void fft();

        /**
         * @brief Performs Fourier transform of arbitrary size
         * @param Nx Size of FFT to perform in X
         * @param Ny Size of FFT in Y
         *           Efficiency is optimized if Nx and Ny are equal and powers
         *           of 2. Hologram will be resized to new size and zero padded
         *           if necessary.
         */
        void fft(int Nx, int Ny);
        
        /**
         * @brief Performs inverse Fourier transform on hologram data
         * @returns Internal data changed to spatial domain.
         *          State changed to HOLOGRAM_STATE_LOADED
         */
        void ifft();
        
        /**
         * @brief Performs inverse Fourier transform of arbitrary size
         * @param Nx Size of FFT to perform in X
         * @param Ny Size of FFT in Y
         *           Efficiency is optimized if Nx and Ny are equal and powers
         *           of 2. Hologram will be resized to new size and zero padded
         *           if necessary.
         */
        void ifft(int Nx, int Ny);

        /**
         * @brief Shift result of fft so that low frequency is in the center
         * @param State must be HOLOGRAM_STATE_FOURIER
         * @returns Modifies internal data and changes state to
         *          HOLOGRAM_STATE_FOURIER_SHIFTED
         */
        void fftshift();

        /**
         * @brief Reverses effect of fftshift
         * @param State must be HOLOGRAM_STATE_FOURIER_SHIFTED
         * @returns Modifies internal data and changes state to
         *          HOLOGRAM_STATE_FOURIER
         */
        void ifftshift();
        
        /**
         * @brief Crop the image using the center and size from internal params
         */
        Hologram crop();

        /**
         * @brief Crop the image to a particular size around a point
         * @param center Center of the ROI to be cropped (relative to original)
         * @param size Size of one side of the ROI (ROI is square)
         * @returns New Hologram with the specified size and data
         */
        Hologram crop(cv::Point center, int size);
        
        /**
         * @brief Crop the image to a region of interest
         * @param roi Region of interest to be used for cropping
         * @returns New Hologram with specified data. Returned hologram
         *          data is not linked to original and is safe to modify
         */
        Hologram crop(cv::Rect roi);
        
        /**
         * @brief Pad the image with zeros
         * @param padding Integer number of pixels of padding on each side
         * @return New hologram of the revised size
         */
        Hologram zeroPad(size_t padding);
        
        /**
         * @brief Set the entire image equal to its mean
         */
        void setToMean();

        /**
         * @brief Applies windowing function to hologram
         * @param method Type of windowing function to use. Currently, only 
         *               Tukey window is implemented
         * @param window_param Parameter necessary for window function. See 
         *                     Matlab documentation for details
         */
        void window(WindowMethod method, float window_param);

        /**
         * @brief Applies low pass filter in Fourier space
         * @param method Type of windowing function to use. Currently, only
         *               Gaussian window is implemented
         * @param window_param Parameter necessary for window function. See
         *                     Matlab documentation for details
         */
        void lowpass(WindowMethod method, float window_param);

        /**
        * @brief Mimics behavior of Matlab tukeywin function. See Matlab
        *        documentation for parameter details
        */
        CuMat createTukeyWindow(int n, double r);

        /**
        * @brief Mimics behavior of Matlab gausswin function. See Matlab
        *        documentation for parameter details
        */
        CuMat createGaussianWindow(int L, float a);
        
        /**
         * @brief Remove 3d objects from the hologram
         *        Reconstruct to in-focus plane and fill object with background
         *        then reconstruct back to the camera plane
         * @param parts Objects to remove
         * @param Initial state must be HOLOGRAM_STATE_NORM_MEAND_ZERO
         */
        void removeObjects(ObjectCloud parts);
        
        /**
         * @brief Determine number and location of tiles with overlap
         *        Guarantees that the full hologram will be covered although
         *        degree of tile overlap may be higher along edges
         * @param size Edge length of square tile to use. Suggested: 256
         * @return rois Vector containing the roi of each tile
         * @return count Number of tiles necessary
         */
        void findTiles(int size, std::vector<cv::Rect> &rois, int &count);

        /**
         * @brief Remove background from images by subtraction
         *        Mimics behavior of MT_ImgEnh_Fn.m
         *        See HongFlowFieldImagingLab/ImageEnhancement/MeanSubtraction
         * @param bg Mean image to be subtracted
         * @param alpha Scaling factor. See Matlab code for details
         * @param beta Scaling factor. See Matlab code for details
         * @param LL_GS Low level greyscale, result shifted by this value
         * @param bgmean Mean of data in bg (i.e. mean(bg))
         * @returns Modifies internal data
         */
        void subtractBackground(Hologram bg, double alpha, double beta, double LL_GS, double bgmean = -1000.0);

        /**
         * @brief Remove background from images by division
         * @param bg Mean image to be removed
         * @returns Modifies internal data
         */
        void divideBackground(Hologram bg);
        
        /**
         * @brief Removes effects of background from image
         *        result is (this - bg) / sqrt(bg)
         * @param bg Background image to be removed. States of bg and this
         *        must match
         * @returns Modifies internal data
         */
        void removeBackground(Hologram bg);
        
        /**
         * @brief Clamp values to saturation limits to simulate recording
         *        Applies the same limits to both complex components
         * @param reference Hologram to match saturation values to. For
         *        example, if reference was of type CV_8U, the saturation
         *        values would be 0 and 255.
         * @returns Modifies internal data
         */
        void applySaturation(Hologram& reference);
        
        /**
         * @brief Determine min and max values of the real part of the hologram
         *      These are the values which would be used for normalization when
         *      writing the hologram to an image file
         * @returns min_val Minimum of real component of hologram
         * @returns max_val Maximum of real component of hologram
         */
        void getMinMax(double* min_val, double* max_val);

        // ACCESS

        /**
         * @brief Opens a window and displays the stored image. Waits for user
         *        before closing the window. Window name is "Hologram"
         */
        void show();

        /**
         * @brief Opens a window and displays the stored image. Waits for user
         *        before closing the window.
         */
        void show(char* name, bool delete_after = true);
        
        /**
         * @brief Writes image to file
         * @param filename Complete destination file
         * @param Calling object must be of state HOLOGRAM_STATE_LOADED
         */
        void write(char* filename);

        /**
         * @brief Force hologram scale when writing
         * @param filename Complete destination file
         * @param new_scale Scaling to assume when writing data, actual scale
         *        will not be changed
         * @param Calling object may be of any state
         */
        void write(char* filename, HologramScale new_scale);
        
        /**
         * @brief Force hologram scale when writing
         * @param filename Complete destination file
         * @param min Min used for normalization
         * @param max Max used for normalization
         */
        void write(char* filename, double min_val, double max_val);

        Parameters getParams() { return params; }
        
        HologramScale getScale() { return scale; }
        
        /**
         * @brief Sets interpreted scale only, no checks performed
         */
        void setScale(HologramScale s) { this->scale = s; }

        /**
         * @brief Returns image data as modifiable CuMat
         */
        umnholo::CuMat getData() { return data; }

        /**
         * @brief Returns background data as modifiable CuMat
         */
        umnholo::CuMat getBgData() { return bg_data; }

        float getOriginalMean() { return original_mean; }
        
        float getSqrtBackgroundMean() { return sqrt_background_mean; }

        void setData(CuMat input) { this->data = input; }
        void setState(int new_state) { this->state = new_state; }

        void setDevicePtr(void* ptr_d, size_t size) { this->data.setCuData(ptr_d, size); }
        
        float getReconstructedPlane() { return reconstructed_plane; }
        void setReconstructedPlane(float z);
        
        void getSaturationLimits(float* lower, float* upper);
        
        size_t getWidth() { return this->data.getWidth(); }
        
        size_t getHeight() { return this->data.getHeight(); }
        
        void setSize(size_t width, size_t height);

        // INQUIRY

        /**
         * @brief Compare the state of the hologram to the user input
         */
        bool isState(int s) const { return state == s; }

        /**
         * @brief Compare the scale of the hologram to the user input
         */
        bool isState(HologramScale s) const { return scale == s; }

        int getState() { return state; }

    protected:
    private:
        Parameters params;
        int state; // Status or state of the hologram using HologramState
        HologramScale scale;

        umnholo::CuMat data;
        umnholo::CuMat bg_data;

        float original_mean;
        HologramScale original_scale;
        float sqrt_background_mean;
        
        float lower_saturation_limit;
        float upper_saturation_limit;
        
        // Status conditions that allow hologram to be reconstructed in 
        // place and returned back to the initial state
        float reconstructed_plane;
        int previous_state;
    };

} // namespace umnholo

#endif // HOLOGRAM_H
