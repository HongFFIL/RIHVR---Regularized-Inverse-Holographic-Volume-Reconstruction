#ifndef COMPRESSIVE_HOLO_H
#define COMPRESSIVE_HOLO_H

// SYSTEM INCLUDES
#include <math.h>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "nppi.h"
#include "npps.h"
#include "curand.h"

// LOCAL INCLUDES
#include "optical_field.h"
#include "reconstruction.h"
#include "hologram.h"

namespace umnholo {
    
    enum CompressiveHoloMode
    {
        COMPRESSIVE_MODE_BRADY_TV_2D = 1,
        COMPRESSIVE_MODE_BRADY_TV_3D = 2,
        COMPRESSIVE_MODE_BRADY_L1 = 3,
        COMPRESSIVE_MODE_FISTA_L1 = 4,
        COMPRESSIVE_MODE_FASTA_L1 = 5,
        COMPRESSIVE_MODE_FASTA_FUSED_LASSO = 6,
        COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D = 7
    };
    
    enum ResidualMode
    {
        RM_EST_TRUE = 0,
        RM_TRUE_EST = 1
    };

    /** @brief Short description of the class
    */
    class CV_EXPORTS CompressiveHolo : public Reconstruction
    {
    public:
        // LIFECYCLE

        CompressiveHolo(Hologram holo);

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
        
        /**
         * @brief Iterative inverse reconstruction using TwIST algorithm
         * @param holo Hologram
         * @param mode Specifier for regularization method to be used
         * @returns Internal data is modified, state is changed
         */
        void inverseReconstruct_TwIST(Hologram holo, CompressiveHoloMode mode);
        
        /**
         * @brief Iterative inverse reconstruction using FISTA algorithm
         * @param holo Hologram
         * @param mode Specifier for regularization method to be used
         * @returns Internal data is modified, state is changed
         */
        void inverseReconstruct_FISTA(Hologram holo, CompressiveHoloMode mode);
        
        /**
         * @brief Iterative inverse reconstruction using FASTA algorithm
         * @param holo Hologram
         * @param mode Specifier for regularization method to be used
         * @param objective Optional returned final objective function value
         * @returns Internal data is modified, state is changed
         */
        void inverseReconstruct_FASTA(Hologram holo, CompressiveHoloMode mode, double* objective = NULL);
        
        /**
         * @brief Iterative inverse reconstruction of the volume
         * @param holo Hologram
         * @param bg Background intensity image
         * @param mode Specifier for regularization method to be used
         * @returns Internal data is modified, state is changed
         */
        //void inverseReconstruct(Hologram holo, Hologram bg, CompressiveHoloMode mode);
        
        /**
         * @brief Compute residual, difference of measured and estimate
         * @returns Hologram with state HOLOGRAM_STATE_NORM_MEAND_ZERO
         *      That state is artificial, no guarantees are made on the 
         *      bounds of the returned data.
         */
        void calcResidual(Hologram* residual);
        
        /**
         * @brief Compute residual, difference of measured and estimate
         * @returns Hologram with state HOLOGRAM_STATE_NORM_MEAND_ZERO
         *      That state is artificial, no guarantees are made on the 
         *      bounds of the returned data.
         * @returns estimate The estimated hologram
         */
        void calcResidual(Hologram* residual, Hologram* estimate, ResidualMode mode = RM_TRUE_EST);
        
        /**
         * @brief Compute residual, difference of measured and estimate
         * @param estimate The estimated hologram
         * @param measured The measured hologram
         * @param mode Sets which input is subtracted from the other
         * @returns Hologram with state HOLOGRAM_STATE_NORM_MEAND_ZERO
         *      That state is artificial, no guarantees are made on the 
         *      bounds of the returned data.
         */
        void calcResidualFrom(Hologram* residual, Hologram* estimate, 
            ResidualMode mode = RM_TRUE_EST);
        
        /**
         * @brief Compute residual, difference of measured and estimate
         * @returns Hologram with state HOLOGRAM_STATE_NORM_MEAND_ZERO
         *      That state is artificial, no guarantees are made on the 
         *      bounds of the returned data.
         */
        //void calcResidual(Hologram* residual, Hologram& bg);
        
        /**
         * @brief Computes the cost of the current estimate. The cost 
         *      is minimized for the optimal solution.
         * @param residual Result of calling calcResidual
         * @returns Value of the objective.
         */
        double calcObjectiveFunction(Hologram residual, CompressiveHoloMode mode);
        
        /**
         * @brief Denoising function ('Psi') 
         * @param x_d Device pointer to output data
         * @param xm1_d Device pointer to previous iteration estimate
         * @param xm2_d Device pointer to N-2 iteration estimate
         * @param grad_d Device pointer to gradient of the solution
         *              (i.e. reconstructed residual)
         * @returns x_d New estimated solution
         */
        void denoise(void* x_d, void* xm1_d, void* grad_d, CompressiveHoloMode mode);
        
        // ACCESS
        
        void setStepsize(double step) { stepsize = step; force_stepsize = true; }
        
        // INQUIRY
        
        // TESTS
        
        double test_lineSearchLimit();
        void test_lineSearch_predicates(OpticalField* x0, OpticalField* grad0);
        double test_estimateLipschitz() {return this->estimateLipschitz();}
        void test_fistaUpdate_results(OpticalField* x1, CuMat* d1);
        CuMat test_denoised();
        Hologram test_estimatedHoloOfDenoised();
        Hologram test_residualOfDenoised();
        double test_objectiveOfDenoised();
        void test_denoise_it1(OpticalField* x1);
        void test_latest_error(CuMat* x1_p2);

    protected:
    private:
        
        Parameters params;
        double tau;
        double projection_tau;
        int num_projection_iterations;
        double convergence_threshold;
        int num_twist_iterations;
        double lambda1;
        double lambdaN;
        double max_svd;
        double rho0;
        double alpha;
        double beta;
        size_t width;
        size_t height;
        size_t depth;
        size_t volume_size;
        
        double stepsize;
        bool force_stepsize;
        double regularization;
        double stepsize_shrinkage;
        std::vector<double> objective_values;
        
        Hologram measured;
        
        CuMat data_nm1;
        CuMat data_nm2;
        Reconstruction gradient;
        CuMat buffer;
        
        // Optional buffers to be used only if extra memory available
        CuMat opt_buffer2;
        CuMat opt_buffer3;
        CuMat opt_buffer4;
        
        /**
         * @brief Update solution using previous steps via the equation
         *        out = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x
         * @param out_d Device pointer to output data
         * @param x_d Device pointer to initial estimate (via denoise)
         * @param xm1_d Device pointer to previous iteration estimate
         * @param xm2_d Device pointer to N-2 iteration estimate
         */
        void twoStepUpdate(void* out_d, void* x_d, void* xm1_d, void* xm2_d);
        
        /**
         * @brief 2D variant of denoising function ('Psi') 
         *        This is to match the version used by Brady where only
         *        the 2D total variation was used
         */
        void denoise2d(void* x_d, void* xm1_d, void* grad_d);
        
        /**
         * @brief 3D variant of denoising function ('Psi') 
         *        This is is the "true" version where all dimensions are
         *        used to compute the total variation
         */
        void denoise3d(void* x_d, void* xm1_d, void* grad_d);
        
        /**
         * @brief Soft thresholding operation for L1-norm regularization
         */
        void softThreshold(void* x_d, void* xm1_d, void* grad_d);

        /**
         * @brief Soft thresholding for FASTA implementation
         */
        void softThreshold(void* x_d, void* xm1_d, void* grad_d, double stepsize, double regularization);
        
        /**
         * @brief Denoising for FISTA (softThreshold)
         *        Differs for TwIST methods by not adding xm1 and grad
         */
        void denoise_FISTA(void* out_d, void* in_d);
        
        /**
         * @brief Returns Y = Y - 2/L * AT(resid);
         */
        void updateY(void* y_d, void* grad_d);
        
        /**
         * @brief Final update step of FISTA
         *        Y=X_iter+t_old/t_new*(Z_iter-X_iter)+(t_old-1)/t_new*(X_iter-X_old);
         */
        void fista_update(void* y_iter_d, void* x_iter_d, void* x_old_d,
            void* z_iter_d, double t_new, double t_old, bool mono_failed);
        
        /**
         * @brief Variant of final FISTA update step used by FASTA
         *        x1 = y1 + step*(y1-y0);
         */
        void fistaUpdate(void* x1_d, void* y0_d, void* y1_d, double step, size_t size);
        
        /**
         * @brief Limit for monotonicity line search
         *        Dx = x1 - x0;
         *        real(dot(Dx(:),gradf0(:)))+norm(Dx(:))^2/(2*tau0);
         */
        double calcLineSearchLimit(void* x1_d, void* x0_d, void* grad_d);

        /**
         * @brief Determine whether to reset the FISTA acceleration parameter
         *        Evaluates (x0(:)-x1(:))'*(x1(:)-x_accel0(:))>0
         * @returns True if the parameter should be reset to 1
         */
        bool fistaRestart(void* x0_d, void* x1_d, void* x_accel_d);
        
        /**
         * @brief Estimate the Lipschitz constant for the smoothing term
         * @warning Requires all internal data as buffer, must only run
         *          at beginning of execution
         * @returns All internal data is modified
         */
        double estimateLipschitz();
    };

} // namespace umnholo

// Kernels are only declared here to be used in tests
__global__ void projection_2d_kernel
    (float2* xm1, float2* grad, 
     float2* div_pn, float2* pn_x, float2* pn_y,
     double tau, double proj_tau, double max_svd,
     int Nx, int Ny, int Nz);

__global__ void div_pn_2d_kernel
    (float2* div_pn, float2* pn_x, float2* pn_y, int Nx, int Ny, int Nz);

__global__ void denoise_output_kernel
    (float2* x_d, float2* xm1, float2* grad, float2* div_pn,
     double max_svd, double tau, int Nx, int Ny, int Nz);

#endif // COMPRESSIVE_HOLO_H
