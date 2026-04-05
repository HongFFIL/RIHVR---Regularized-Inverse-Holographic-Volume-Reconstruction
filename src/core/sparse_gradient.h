#ifndef SPARSE_GRADIENT_H
#define SPARSE_GRADIENT_H

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
#include "reconstruction.h"

namespace umnholo {
    
    /** 
     * @brief Sparse storage structure for reconstruction gradient data
     */
    class CV_EXPORTS SparseGradient
    {
    public:
    // LIFECYCLE
    
    void initialize(Hologram holo, cufftHandle* fft_plan_p);
    
    void destroy();
    
    // OPERATORS
    // OPERATIONS
    
    /**
     * @brief Indicate that volume is reconstructed from the hologram
     */
    void reconstruct(Hologram holo, ReconstructionMode mode);
    
    // ACCESS
    
    /*
     * @brief Returns a single 2D plane of the volume
     * @param plane Returned plane data, will overwrite any old data
     *        Size is equal to the size of the hologram
     * @param plane_idx Identifier for which plane to get. Less than getNumPlanes.
     */
    void getPlane(CuMat* plane, size_t plane_idx);
    
    // INQUIRY

    protected:
    private:
        
        Parameters params;
        
        size_t width;
        size_t height;
        
        bool use_reconstruction;
        Hologram holo;
        
        cufftHandle fft_plan;
        float2* holo_fft_d;
        
        CuMat exponent_data;
    };
} // namespace umnholo

#endif // SPARSE_GRADIENT_H