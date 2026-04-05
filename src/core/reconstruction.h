#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

// SYSTEM INCLUDES
//

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "npp.h"

// LOCAL INCLUDES
#include "optical_field.h"
#include "focus_metric.h"
#include "holo_sequence.h"
#include "hologram.h"

namespace umnholo {

    enum ReconstructionMode
    {
        RECON_MODE_BASIC = 0,
        RECON_MODE_DECONVOLVE = 1,
        RECON_MODE_FSP = 2,
        RECON_MODE_COMPLEX = 3,
        RECON_MODE_COMPLEX_CONSTANT_SUM = 4,
        RECON_MODE_COMPLEX_ABS = 5
    };

    enum ReconstructionKernel
    {
        KERNEL_KF = 0,
        KERNEL_RS = 1
    };

    /** @brief Short description of the class
    */
    class CV_EXPORTS Reconstruction : public OpticalField
    {
    public:
        // LIFECYCLE

        Reconstruction(Hologram holo) : OpticalField(holo), plan_created(false){}
        
        /**
         * @brief Allows avoiding allocation of entire volume
         * @param holo Hologram used primarily to pass other parameters
         * @param init_method Initialization method. 
         *      Use OPTICALFIELD_DO_NOT_ALLOCATE to avoid allocating the volume
         */
        Reconstruction(Hologram holo, OpticalFieldAllocation init_method) :
            OpticalField(holo, init_method), plan_created(false){}

        void destroy();

        // OPERATORS
        // OPERATIONS

        /**
        * @brief Reconstruct 2D hologram to 3D volume
        * @param holo Hologram must be of state HOLOGRAM_STATE_NORM_MEAND
        *             Throws HOLO_ERROR_INVALID_STATE if not true
        * @param mode Specifier for which type of reconstruction is to be used
        * @returns Internal data is modified. State is changed
        */
        void reconstruct(Hologram holo, ReconstructionMode mode = RECON_MODE_BASIC);

        using OpticalField::combinedXY;
        /**
        * @brief Returns the XY minimum intensity of the reconstruction
        *        Does not require OpticalField to be reconstructed already
        * @param holo Hologram to reconstruct
        * @returns Same as OpticalField::combinedXY()
        */
        CuMat combinedXY(Hologram& holo);

        /**
        * @brief Returns the XY maximum intensity of the reconstruction
        *        Does not require OpticalField to be reconstructed already
        * @param holo Hologram to reconstruct
        * @returns Value similar to combinedXY for inverted but squared
        */
        CuMat projectMaxIntensity(Hologram& holo);

        /**
         * @brief Returns the XY maximum intensity of the reconstruction and
         *        the plane index where that maximum ocurred.
         *        Does not require OpticalField to be reconstructed already
         * @param holo Hologram to reconstruct
         * @param cmb_out The xy maximum intensity projection
         * @param arg_out The z plane index indicating the source of the maxima
         */
        void projectMaxIntensity(
            Hologram& holo,
            CuMat cmb_out,
            CuMat arg_out);

        /**
        * @brief Multiply data by complex conjugate
        * @param Initial state: OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX
        * @returns Modifies internal data
        *          Final state: OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX_REAL
        */
        void multiply_conjugate();

        /**
         * @brief Calculate metric for determining in-focus plane
         * @param holo Hologram of state HOLOGRAM_STATE_NORM_MEAND
         * @returns Filled raw FocusMetric
         */
        FocusMetric calcFocusMetric(Hologram* holo);

        /**
         * @brief Calculate metric for determining in-focus plane
         * @param holo Sequence of holograms of state HOLOGRAM_STATE_NORM_MEAND
         * @param metrics Pointer to array of FocusMetric objects to be filled
         *                Array must declared to the maximum length of the 
         *                HoloSequence but need not be allocated yet
         * @param count Returns the number of metrics filled
         */
        void calcFocusMetric(HoloSequence* holo, FocusMetric* metrics, int* count);

        /**
        * @brief Reconstruct the hologram to a particular plane
        * @param holo Input hologram image. Must be in fourier domain
        * @param z Distance to reconstruct to (units must be consistent with
        *          params.resolution and params.wavelength)
        * @param mode Specifier for which type of reconstruction is to be used
        * @returns holo Modified, state set to HOLOGRAM_STATE_RECONSTRUCTED
        *               can be reversed by calling again with z = -z
        */
        void reconstructTo(Hologram& holo, float z, ReconstructionMode mode = RECON_MODE_BASIC);

        /**
        * @brief Reconstruct the hologram to a particular plane
        * @param holo Input hologram image. Must not be in fourier domain
        * @param z Distance to reconstruct to (units must be consistent with
        *          params.resolution and params.wavelength)
        * @param mode Specifier for which type of reconstruction is to be used
        * @returns out_plane Data storage location for output reconstruction
        */
        void reconstructTo(CuMat* out_plane, Hologram holo, float z, ReconstructionMode mode = RECON_MODE_COMPLEX_ABS);

        /**
        * @brief Intensity reconstruction is square of reconstructTo result
        * @param holo Input hologram image. Must be in fourier domain
        * @param z Distance to reconstruct to (units must be consistent with
        *          params.resolution and params.wavelength)
        * @returns out_plane Data storage location for output reconstruction
        */
        void reconstructIntensityTo(CuMat* out_plane, Hologram holo, float z);
        
        /**
         * @brief Reverse reconstruction, from 3D volume to 2D plane
         * @param holo Output data destination
         * @param buffer Buffer data, will be allocated if necessary, if
         *      preallocated, must be size of reconstruction volume
         * @param mode Specifier for which reconstruction method is to 
         *      be used. Default is RECON_MODE_BASIC. Only 
         *      RECON_MODE_COMPLEX is implemented.
         * @returns holo Modifies internal data
         */
        void backPropagate(Hologram* holo, CuMat* buffer, ReconstructionMode mode = RECON_MODE_BASIC);
        
        /**
         * @brief Reverse reconstruction, from 3D volume to 2D plane
         * @param holo Output data destination
         * @param bg Background intensity image
         * @param buffer Buffer data, will be allocated if necessary, if
         *      preallocated, must be size of reconstruction volume
         * @param mode Specifier for which reconstruction method is to 
         *      be used. Default is RECON_MODE_BASIC. Only 
         *      RECON_MODE_COMPLEX is implemented.
         * @param State must be OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX
         * @returns holo Modifies internal data
         */
        //void backPropagate(Hologram* holo, Hologram& bg, CuMat* buffer, ReconstructionMode mode = RECON_MODE_BASIC);

        // ACCESS
        // INQUIRY

    protected:
    private:
        cufftHandle fft_plan;
        bool plan_created;
        CuMat buffer_data;

        // RECONSTRUCTION METHODS
        // Defined in reconstruction.cu
        /**
        * @brief Rayleigh-Sommefeld reconstruction
        * @param holo see reconstruct
        * @param kernel Identify whether to use a Rayleigh-Sommerfeld or
        *               Kirchoff-Fresnel reconstruction kernel
        */
        void recon_basic(Hologram holo, ReconstructionKernel kernel = KERNEL_KF);

        /**
        * @brief Rayleigh-Sommefeld reconstruction
        *        Result is the complex reconstruction
        * @param see reconstruct
        */
        void recon_rs_complex(Hologram holo);

        /**
        * @brief Rayleigh-Sommefeld reconstruction
        *        Result is the complex reconstruction with shift for
        *        accurate phase estimate
        * @param see reconstruct
        */
        void recon_rs_phase_complex(Hologram holo);

        /**
        * @brief Rayleigh-Sommefeld reconstruction using free space propagation
        *        Does not use the hologram data
        * @param see reconstruct
        */
        void recon_rs_fsp(Hologram holo);
        
        /**
         * @brief Multiply data uniformly by scale factor
         * @param scale_factor Scalar value by which to multiply all data
         * @returns Internal data modified, no state change
         */
        void scaleData(double scale_factor);
    };

} // namespace umnholo

#endif // RECONSTRUCTION_H
