#ifndef SPARSE_COMPRESSIVE_HOLO_H
#define SPARSE_COMPRESSIVE_HOLO_H

// SYSTEM INCLUDES
#include <math.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"

// LOCAL INCLUDES
#include "umnholo.h"
#include "holo_ui.h"
#include "hologram.h"
#include "cumat.h"
#include "compressive_holo.h"
#include "sparse_volume.h"
#include "sparse_gradient.h"

namespace umnholo {
    
    /** @brief Short description of the class
    */
    class CV_EXPORTS SparseCompressiveHolo
    {
    public:
        // LIFECYCLE

        SparseCompressiveHolo(Hologram holo);

        void destroy();

        // OPERATORS
        // OPERATIONS
        
        /**
         * @brief Iterative inverse reconstruction of the volume
         * @param holo Hologram
         * @param mode Specifier for regularization method to be used
         * @returns Internal data is modified, state is changed
         */
        void inverseReconstruct(Hologram holo, CompressiveHoloMode mode);
        
        void calcResidual(Hologram* residual);
        
        void calcResidual(Hologram* residual, Hologram* estimate, ResidualMode mode);
        
        void calcInitialResidual(Hologram* residual, Hologram* estimate, ResidualMode mode);
        
        void calcResidualFrom(Hologram* residual, Hologram* estimate, ResidualMode mode);
        
        double calcResidualObjective(Hologram residual);
        
        double calcRegularizationObjective(CompressiveHoloMode mode);
        
        double calcObjectiveFunction(Hologram residual, CompressiveHoloMode mode);
        
        double calcInitialObjectiveFunction(CompressiveHoloMode mode);
        
        /**
         * @brief Save voxel locations and values to file
         * @param prefix Prefix of output file name
         */
        void saveSparse(char* prefix);
        
        /**
         * @brief Writes XY, YZ, and XZ max projections to tiff files
         * @param prefix Prefix to be appended to filename. Format will
         *        be [output_path]/[prefix]_[xy|xz|yz].tif
         * @param method Indicates method for taking projection (max or min)
         *        This is included only to match the OpticalField method.
         */
        void saveProjections(char* prefix, CmbMethod method = MAX_CMB, bool prefix_as_suffix = false);
        
        /**
         * @brief Save an image of each plane in the volume
         * @param prefix Prefix to be used for filename. Format will be
         *        [output_path]/[prefix]_[planeID].tif
         */
        //void savePlanes(char* prefix);
        void savePlanes(const char* prefix);

        // ACCESS

        /**
         * @brief Access a single plane from the volume
         * @param plane_idx Index value for the plane. Allowable range is 0 to 
         *                  num_planes. Other values will throw error
         * @returns CuMat object containing the plane data
         */
        CuMat getPlane(size_t plane_idx);
        
        void setStepsize(double step) { stepsize = step; force_stepsize = true; }
        
        // INQUIRY
        // TESTS
        
        double test_estimateLipschitz() {return this->estimateLipschitz();}
        CuMat test_denoised();
        Hologram test_estimatedHoloOfDenoised();
        Hologram test_residualOfDenoised();
        double test_objectiveOfDenoised();
        void test_fistaUpdate_results(CuMat* x1_p2, CuMat* d1);
        void test_denoise_it1(CuMat* x1_p2);
        CuMat test_FL_prox_it1(int num_tv_its);
        void test_latest_error(CuMat* x1_p2);
        
    protected:
    private:
        
        Parameters params;
        std::vector<double> objective_values;
        
        double stepsize;
        double stepsize_shrinkage;
        double regularization;
        double regularization_TV;
        bool force_stepsize;
        int num_tv_iterations;
        double lipschitz;
        
        int iteration;
        
        size_t width;
        size_t height;
        size_t depth;
        int plane_subsampling;
        
        SparseVolume x1;
        SparseVolume x0;
        SparseVolume y0;
        SparseVolume y1;
        
        // Used for Total Variation and Fused Lasso only
        SparseVolume gx;
        SparseVolume gy;
        SparseVolume gz;
        
        SparseGradient gradient;
        // SparseGradient bg_rec;
        
        CuMat plane;
        float2* buffer_plane_d;
        
        CuMat x1_plane;
        CuMat x0_plane;
        CuMat grad_plane;
        // CuMat bg_plane;
        CuMat y0_plane;
        CuMat y1_plane;
        CuMat gx_plane;
        CuMat gy_plane;
        CuMat gz_plane;
        
        Hologram measured;
        // Hologram bg_holo;
        cufftHandle fft_plan;
        bool fft_plan_created;
        double min_measured;
        double max_measured;
        // double min_residual;
        // double max_residual;
        
        CuMat exponent_data;
        
        bool use_fista;
        
        // TODO: Add function documentation. For now, just see compressive_holo.h
        void backPropagate(Hologram* estimate, ReconstructionMode mode);
        void backPropagateInitial(Hologram* estimate, ReconstructionMode mode);
        double estimateLipschitz(int seed = 1234);
        bool testAdjoint(int seed = 1234);
        //void denoise(SparseVolume x1, SparseVolume x0, SparseGradient grad, CompressiveHoloMode mode);
        void denoise(CompressiveHoloMode mode);
        void proximalL1();
        void enforceSparsity(); // applies threshold to volumes
        void proximalFusedLasso();
        void proximalFusedLasso2d();
        void proximalFusedLasso1dZ();
        void proximalAlternatingFusedLasso2d();
        double calcLineSearchLimit(SparseVolume* x1, SparseVolume* x0, SparseGradient* grad);
        bool fistaRestart(SparseVolume* x0, SparseVolume* x1, SparseVolume* y0);
        void fistaUpdate(SparseVolume* x1, SparseVolume* y0, SparseVolume* y1, double step);
        void fistaUpdate(Hologram* est, Hologram* prev, Hologram* current, double step);
        
        void totalVariationDenoisedEstimate();
        void totalVariationProjection();
        void totalVariationGradientOperator();
    };
    
} // namespace umnholo

#endif // SPARSE_COMPRESSIVE_HOLO_H