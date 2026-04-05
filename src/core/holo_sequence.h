#ifndef HOLO_SEQUENCE_H
#define HOLO_SEQUENCE_H

// SYSTEM INCLUDES
#include <stdint.h>
#include <stdlib.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"

// LOCAL INCLUDES
#include "hologram.h"
#include "holo_ui.h"
#include "umnholo.h"
#include "point_cloud.h"

namespace umnholo {

    const int MULT_FOCUS_METRIC = 2;

    enum HoloSequenceState
    {
        UNKNOWN = -1,
        UNINITIALIZED = 0,
        INITIALIZED = 1
    };
    
    enum HoloSequenceProcess
    {
        LOAD_UNCHANGED = 0,
        LOAD_NORM_INVERSE = 1,
        LOAD_BG_DIVISION = 2
    };

    /**
     * @brief For efficient processing of large number of images concurrently
     */
    class CV_EXPORTS HoloSequence
    {
    public:
        // LIFECYCLE

        /**
         * @brief Use parameters to determine number and size of images
         */
        HoloSequence(const Parameters params);

        /**
         * @brief Deallocate all stored data
         *        Do not call delete on individual holograms or will break
         */
        void destroy();

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Set size to maximum size that can be stored in gpu memory
         * @param multiplier Indicates that additional space will be needed
         *                   for further processing. Some constants are defined
         *                   to aid in this selection.
         *                   Data will be sized such that the size of N images
         *                   will be less than multiplier*N*image_size
         */
        int findLargestCount(int multiplier);

        /**
         * @brief Indicate that image should use a cropped region of interest
         *        of the image for processing rather than the whole thing
         * @param size Size of ROI. ROI will be square.
         * @param centers Indicates center of ROI's for each image
         */
        void useRoi(int size, PointCloud centers);

        /**
         * @brief Allocates space on the gpu for as many images as possible. 
         * @param findLargestCount must have been previously called. Otherwise
         *        throws HOLO_ERROR_INVALID_STATE
         */
        void init();

        /**
         * @brief Reads and loads as many images into memory as possible
         * @param Space must have been allocated using init
         * @returns false if there is no more data to be loaded
         */
        bool loadNextChunk(HoloSequenceProcess norm = LOAD_NORM_INVERSE);
        
        /**
         * @brief Result is same as previous call to loadNextChunk 
         *        regardless of changes to data
         */
        bool reloadChunk(HoloSequenceProcess norm = LOAD_NORM_INVERSE);

        /**
         * @brief Resets chunk to start of sequence to read everything again
         *        In the case that loadNextChunk has not been called, does
         *        nothing.
         */
        void reset();

        /**
         * @brief Shows all the images on the screen in quick succession
         */
        void view();

        /**
        * @brief Normalize the hologram and invert it
        * @returns Modifies data by dividing by the mean and subtracting from
        *          one. Mathematically: result = 1 - data/mean(data)
        *          Changes state.
        */
        //void normalize_inverse();

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
        * @brief Performs Fourier transform of arbitrary size
        * @param Nx Size of FFT to perform in X
        * @param Ny Size of FFT in Y
        *           Efficiency is optimized if Nx and Ny are equal and powers
        *           of 2. Hologram will be resized to new size and zero padded
        *           if necessary.
        */
        void fft(int Nx, int Ny);

        /**
        * @brief Shift result of fft so that low frequency is in the center
        * @param State must be HOLOGRAM_STATE_FOURIER
        * @returns Modifies internal data and changes state to
        *          HOLOGRAM_STATE_FOURIER_SHIFTED
        */
        void fftshift();

        /**
         * @brief Remove background from images by subtraction
         *        Mimics behavior of MT_ImgEnh_Fn.m
         *        See HongFlowFieldImagingLab/ImageEnhancement/MeanSubtraction
         * @param bg Mean of all images in the sequence
         * @param alpha Scaling factor calculated as 
         *        mean(diff_mean) - A*mean(diff_std)
         *        where diff_mean and diff_std are the mean and standard
         *        deviation of the images after subtracting bg
         * @param beta Scaling factor calculated as
         *        255 / (2 * A*mean(diff_std))
         * @param LL_GS Low level greyscale, result shifted by this vlaue
         * @param bgmean Mean of data in bg (i.e. mean(bg))
         * @returns Modifies internal data
         */
        void subtractBackground(Hologram bg, double alpha, double beta, double LL_GS, double bgmean); 

        /**
         * @brief Causes images to be enhanced with background subtraction
         *        Enhancement is performed after loading so calls to_array
         *        getHologram will return the enhanced image.
         *        Background and statistics for the entire image sequence
         *        are calculated in this call, necessitating 2 reads of the 
         *        full image sequence.
         * @returns Modifies internal data
         */
        void useBackgroundSubtraction();

        // ACCESS

        /**
         * @brief Returns pointer to device array containing all hologram data
         */
        void* getCuData();

        size_t getWidth() { return image_width; }
        size_t getHeight() { return image_height; }
        int getNumHolograms() { return num_loaded; }
        
        Hologram getHologram(int idx) { return holos[idx]; }

        Hologram getBackground();

        void getBackgroundStats(double* alpha, double* beta, double* minimum, double* average);

        int getImageStepsize() { return image_stepsize; }

        Parameters getParams() { return params; }

        /**
         * @brief Force the sequence to account for images of the given size
         */
        void setSize(size_t width, size_t height);

        /**
         * @brief Images will be loaded in increments of the given step
         *        Default: 1
         */
        void setImageStepsize(int stepsize) { image_stepsize = stepsize; }

        /**
         * @brief When loading images, the time from the point cloud will be
         *        used as the expected file identifier.
         *        Only applicable if useRoi has been called.
         */
        void useTimeAsImageId() { use_time_as_image_id = true; }

        // INQUIRY

    protected:
    private:
        HoloSequenceState state;
        Parameters params;

        Hologram* holos;
        void* holo_data_d;
        int store_count;
        int num_loaded;

        bool use_time_as_image_id;

        size_t image_width;
        size_t image_height;
        size_t image_bytes;

        bool use_roi;
        int roi_size;
        PointCloud points;

        bool use_bg_sub;
        Hologram background;
        double bg_alpha;
        double bg_beta;
        double bg_scale1;
        double bg_min_grey;
        double bg_mean;

        size_t total_count;
        size_t num_chunks;
        size_t current_chunk;
        int start_image;
        int image_stepsize;
    };

} // namespace umnholo

#endif // HOLO_SEQUENCE_H
