#ifndef DECONVOLUTION_H
#define DECONVOLUTION_H

// SYSTEM INCLUDES
//

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "nppi.h"

// LOCAL INCLUDES
#include "optical_field.h"
#include "reconstruction.h"

namespace umnholo {

    /** @brief Short description of the class
    */
    class CV_EXPORTS Deconvolution : public Reconstruction
    {
    public:
        // LIFECYCLE

        Deconvolution(Hologram holo);

        void destroy();

        // OPERATORS
        // OPERATIONS

        /**
        * @brief Deconvolve reconstructed 3D volume to improve SNR
        * @param Calling object must be reconstructed (result of reconstruct)
        * @returns Internal data is modified. State is changed
        */
        void deconvolve(Hologram holo);

        /**
        * @brief Returns the XY minimum intensity image of the 3D volume
        * @returns Result of min in z direction for each XY pixel. Returned as
        *          a CuMat containing a CV_8U Mat that can be saved with imwrite
        */
        CuMat combinedXY();
        
        /**
         * @brief Rescales data to match first iteration
         *        Equivalent to data = mean1 + (data - mean2).*(std1/std2)
         *        mean1 and std1 are calculated within deconvolve
         * @returns If mean1 and std1 are uninitialized, data will be unchanged
         */
        void matchMeanStd();

        // ACCESS
        
        void setIterativeMode(bool val) { iterative_mode = val; }
        
        void setInitialVolumeMeanStd(float mean, float std);
        
        void getVolumeMeanStd(float* mean, float* std);
        
        // INQUIRY

    protected:
    private:
        
        /**
         * @brief Fills the first two planes with the mean of the volume
         */
        void eliminateCaustics();
        
        bool iterative_mode;
        float current_volume_mean;
        float current_volume_std;
        float initial_volume_mean;
        float initial_volume_std;
        bool initial_volume_stats_set;
        
        cufftHandle fft_plan_3d;
        Reconstruction psf;
        bool is_created_fft_plan_3d;
        bool is_created_psf;
    };

} // namespace umnholo

#endif // DECONVOLUTION_H
