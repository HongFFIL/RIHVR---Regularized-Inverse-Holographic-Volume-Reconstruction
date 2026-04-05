#include "compressive_holo.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

CompressiveHolo::CompressiveHolo(Hologram holo) : gradient(holo), 
                                                  Reconstruction(holo)
{
    params = holo.getParams();
    
    // Default parameter values
    tau = params.regularization_param;
    projection_tau = 0.05;
    num_projection_iterations = 4;
    convergence_threshold = 1e-6;
    num_twist_iterations = params.num_inverse_iterations;
    lambda1 = 1e-4;
    lambdaN = 1;
    max_svd = params.max_svd;
    
    stepsize = 1 / params.max_svd;
    force_stepsize = false;
    regularization = params.regularization_param;
    stepsize_shrinkage = 0.5;
    
    rho0 = (1 - lambda1/lambdaN) / (1 + lambda1/lambdaN);
    alpha = 2 / (1 + sqrtf(1 - rho0*rho0));
    beta = alpha*2 / (lambda1 + lambdaN);
    
    measured = holo;
    
    width = data.getWidth();
    height = data.getHeight();
    depth = data.getDepth();
    volume_size = width*height*depth;
    this->data.allocateCuData(width, height, depth, 8);
    
    data_nm1.allocateCuData(width, height, depth, 8);
    data_nm2.allocateCuData(width, height, depth, 8);
    gradient.getData().allocateCuData(width, height, depth, 8);
    buffer.allocateCuData(width, height, depth, 8);
    
}// CompressiveHolo

void CompressiveHolo::destroy()
{
    data.destroy();
    data_nm1.destroy();
    data_nm2.destroy();
    gradient.destroy();
    buffer.destroy();
    return;
}


//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

/*void CompressiveHolo::inverseReconstruct(Hologram holo, CompressiveHoloMode mode)
{
    Hologram bg(params);
    cv::Mat holo_mat = holo.getData().getMatData();
    CuMat bg_data;
    bg_data.setMatData(holo_mat);
    float2* bg_d = (float2*)bg_data.getCuData();
    CUDA_SAFE_CALL(cudaMemset(bg_d, 0, width*height*sizeof(float2)));
    bg_data.setCuData((void*)bg_d);
    bg.setData(bg_data);
    
    inverseReconstruct(holo, bg, mode);
    bg.destroy();
}*/

void CompressiveHolo::inverseReconstruct(Hologram holo, CompressiveHoloMode mode)
{
    switch (mode)
    {
    case COMPRESSIVE_MODE_BRADY_TV_2D: {}
    case COMPRESSIVE_MODE_BRADY_TV_3D: {}
    case COMPRESSIVE_MODE_BRADY_L1:
    {
        this->inverseReconstruct_TwIST(holo, mode);
        break;
    }
    case COMPRESSIVE_MODE_FISTA_L1:
    {
        this->inverseReconstruct_FISTA(holo, mode);
        break;
    }
    case COMPRESSIVE_MODE_FASTA_L1:
    {
        this->inverseReconstruct_FASTA(holo, mode);
        break;
    }
    default:
    {
        std::cout << "CompressiveHolo::inverseReconstruct: unknown mode" << std::endl;
        std::cout << "  mode was " << mode << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
}

/** 
 * Iterative reconstruction using the TwIST algorithm
 * This code was adapted from Brady et al 2009
 * Their source code can be found online at
 * http://www.disp.duke.edu/projects/ComputationalHolography/CompressiveHolography/index.ptml
 */
void CompressiveHolo::inverseReconstruct_TwIST(Hologram holo, CompressiveHoloMode mode)
{
    DECLARE_TIMING(INV_INVERSE_RECONSTRUCT);
    START_TIMING(INV_INVERSE_RECONSTRUCT);
    
    // Initialize guess by reconstructing recorded hologram
    this->reconstruct(holo, RECON_MODE_COMPLEX);
    Hologram residual(params);
    calcResidual(&residual);
    double prev_f = calcObjectiveFunction(residual, mode);
    double f = prev_f;
    std::cout << "Initial objective = " << f << std::endl;
    
    void* x_d = this->data.getCuData();
    void* xm1_d = data_nm1.getCuData();
    void* xm2_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(xm1_d, x_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(xm2_d, x_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    
    bool solution_converged = false;
    int it_count = 0;
    bool twist_it = false;
    
    printf("Inverse alpha = %f, beta = %f\n", alpha, beta);
    printf("lambda1 = %f, lambdaN = %f\n", lambda1, lambdaN);
    printf("max_svd = %f\n", max_svd);
    printf("params.max_svd = %f\n", params.max_svd);
    printf("stepsize = %f\n", stepsize);
    
    while (!solution_converged)
    {
        DECLARE_TIMING(INV_REC_ITERATION);
        START_TIMING(INV_REC_ITERATION);
        DECLARE_TIMING(INV_RECONSTRUCT_RESID);
        START_TIMING(INV_RECONSTRUCT_RESID);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX);
        void* grad_d = gradient.getData().getCuData();
        SAVE_TIMING(INV_RECONSTRUCT_RESID);
        
        while (true)
        {
            this->denoise(x_d, xm1_d, grad_d, mode);
            if (twist_it)
            {
                //this->twoStepUpdate(x_d, x_d, xm1_d, xm2_d);
                //cudaMemcpy(xm2_d, x_d, width*height*depth*sizeof(float2), cudaMemcpyDeviceToDevice);
                this->twoStepUpdate(xm2_d, x_d, xm1_d, xm2_d);
                
                // Test monotonicity
                calcResidual(&residual);
                f = calcObjectiveFunction(residual, mode);
                if (f > prev_f)
                {
                    twist_it = false;
                }
                else
                {
                    twist_it = true;
                    break; // while (true)
                }
            }
            else
            {
                calcResidual(&residual);
                f = calcObjectiveFunction(residual, mode);
                if (f > prev_f)
                {
                    max_svd = 2*max_svd;
                    printf("max_svd = %f\n", max_svd);
                    twist_it = false;
                    std::cout << "Increasing max_svd = " << max_svd << std::endl;
                    if (max_svd > 1e10)
                    {
                        solution_converged = true;
                        break;
                    }
                }
                else
                {
                    twist_it = true;
                    break; // while (true)
                }
            }
        } // while (true)
        
        void* temp = xm2_d;
        xm2_d = xm1_d;
        xm1_d = x_d;
        x_d = temp;
        this->data.setCuData(x_d);
        
        if (abs((f - prev_f) / prev_f) < convergence_threshold)
            solution_converged = true;
        
        prev_f = f;
        
        it_count++;
        std::cout << "Iteration " << it_count << " of " 
            << num_twist_iterations << " cost = " << f << std::endl;
        if (it_count >= num_twist_iterations)
            solution_converged = true;
        
        if (solution_converged)
        {
            x_d = this->data.getCuData();
            CUDA_SAFE_CALL( cudaMemcpy(x_d, xm1_d, 
                volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
        }
        SAVE_TIMING(INV_REC_ITERATION);
    } // while (!solution_converged)
    
    this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct");
    SAVE_TIMING(INV_INVERSE_RECONSTRUCT);
    
    return;
}

/** 
 * Iterative reconstruction using the FISTA algorithm
 * This method is similar to that of Endo et al. 2016
 * Their source code can be found online at https://sites.google.com/site/amirbeck314/software
 */
void CompressiveHolo::inverseReconstruct_FISTA(Hologram holo, CompressiveHoloMode mode)
{
    DECLARE_TIMING(INV_INVERSE_RECONSTRUCT);
    START_TIMING(INV_INVERSE_RECONSTRUCT);
    
    // Initialize guess by reconstructing recorded hologram
    this->reconstruct(holo, RECON_MODE_COMPLEX);
    Hologram residual(params);
    calcResidual(&residual);
    double prev_f = calcObjectiveFunction(residual, mode);
    double f = prev_f;
    std::cout << "Initial objective = " << f << std::endl;
    
    void* x_iter_d = this->data.getCuData();
    void* x_old_d = data_nm1.getCuData();
    void* y_iter_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x_old_d, x_iter_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y_iter_d, x_iter_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    
    float t_new = 1.0;
    
    for (int it = 0; it < num_twist_iterations; ++it)
    {
        float t_old = t_new;
        void* temp = x_old_d;
        x_old_d = x_iter_d;
        x_iter_d = temp;
        
        this->data.setCuData(y_iter_d);
        this->calcResidual(&residual);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX);
        void* grad_d = gradient.getData().getCuData();
        void* z_iter_d = grad_d;
        
        this->updateY(y_iter_d, grad_d);
        
        this->denoise_FISTA(x_iter_d, y_iter_d);
        
        // Test monotonicity
        this->data.setCuData(x_iter_d);
        this->calcResidual(&residual);
        f = calcObjectiveFunction(residual, mode);
        bool mono_failure = false;
        if (f > prev_f)
        {
            std::cout << "Monotonicity failed";// << std::endl;
            std::cout << " old=" << prev_f << ", new=" << f << std::endl;
            cudaMemcpy(z_iter_d, x_iter_d, width*height*depth*sizeof(float2), cudaMemcpyDeviceToDevice);
            cudaMemcpy(x_iter_d, x_old_d, width*height*depth*sizeof(float2), cudaMemcpyDeviceToDevice);
            f = prev_f;
            mono_failure = true;
            //t_old = 1;
        }
        prev_f = f;
        
        // Updated t and Y
        t_new = (1.0 + sqrt(1.0 + 4.0*t_old*t_old)) / 2.0;
        fista_update(y_iter_d, x_iter_d, x_old_d, z_iter_d, t_new, t_old, mono_failure);
        
        std::cout << "Iteration " << it << " of " 
            << num_twist_iterations << " cost = " << f << std::endl;
    }
    
    this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FISTA");
    SAVE_TIMING(INV_INVERSE_RECONSTRUCT);
}

/** 
 * Iterative reconstruction using the FASTA algorithm
 * FASTA is a variant of FISTA as implemented by Goldstein 2014
 * This method is similar to that of Endo et al. 2016
 * The FASTA paper is 
 *      Goldstein, T., Studer, C., & Baraniuk, R. (2014). A Field Guide to 
 *      Forward-Backward Splitting with a FASTA Implementation. arXiv:1411.3406, 25. 
 *      Retrieved from http://arxiv.org/abs/1411.3406
 * Their source code can be found online at https://www.cs.umd.edu/~tomg/projects/fasta/
 */
void CompressiveHolo::inverseReconstruct_FASTA(Hologram holo, CompressiveHoloMode mode, double* objective)
{
    double L = estimateLipschitz();
    stepsize = (2.0 / L) / 10.0;
    printf("L = %f, stepsize = %f\n", L, stepsize);
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    //this->reconstruct(holo, RECON_MODE_COMPLEX);
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    //double tau1 = stepsize;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    //Hologram previous_estimate;
    //this->backPropagate(&holo_estimate, &buffer, RECON_MODE_COMPLEX);
    //CuMat estimate = holo_estimate.getData();
    //void* d1_d = estimate.getCuData();
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
    for (int it = 0; it < num_twist_iterations; ++it)
    {
        std::cout << "FASTA iteration " << it << " of " << num_twist_iterations << ": ";
        std::cout << std::endl;
        
        // May be able to increase stepsize to speed up convergence
        /*if (its_without_backtrack >= 10)
        {
            stepsize /= stepsize_shrinkage;
            std::cout << std::endl << "Attempting to change stepsize: " << stepsize << std::endl;
        }*/
        
        // Rename iterates 
        CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
        //double tau0 = tau1;
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(x1_d, x0_d, grad_d, mode);
        
        // Non-monotone backtracking line search
        int backtrack_count = 0;
        this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
        f = calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1);
        //std::cout << "    lb = " << look_back << std::endl;
        //std::cout << "  objective_values: ";
        //for (std::vector<double>::const_iterator i = objective_values.begin(); i != objective_values.end(); ++i)
        //    std::cout << *i << ' ';
        //std::cout << std::endl;
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
        //std::cout << ", M = " << M << ", lim = " << lim << std::endl;
        //std::cout << "M+lim = " << M << "+" << lim << "=" << M+lim << " ";
        //printf("\n  before backtrack f = %e\n", f);
        double ratio = f / (M+lim);
        printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
        while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
        {
            its_without_backtrack = 0;
            //printf("    f = %f, M+lim = %f+%f = %f, comparison failed\n", f, M, lim, M+lim);
            stepsize *= stepsize_shrinkage;
            this->denoise(x1_d, x0_d, grad_d, mode);
            this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
            f = calcObjectiveFunction(residual, mode);
            lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
            ratio = f / (M+lim);
            backtrack_count++;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
            return;
        }
        its_without_backtrack++;
        printf("  after backtracks f = %f\n", f);
        
        // Skip implementing stopping criteria check
        
        // Begin FISTA acceleration steps

        // Update acceleration parameters
        if (fistaRestart(x0_d, x1_d, y0_d))
        {
            std::cout << "restarted FISTA parameter" << std::endl << "  ";
            alpha0 = 1;
        }
        alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;

        // Update x1
        buffer.allocateCuData(width,height,depth,sizeof(float2));
        void* y1_d = buffer.getCuData();
        cudaMemcpy(y1_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);
        double step = (alpha0-1) / alpha1;
        fistaUpdate(x1_d, y0_d, y1_d, step, volume_size);
        cudaMemcpy(y0_d, y1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);

        // Update d1
        CuMat estimate = holo_estimate.getData();
        void* d1_d = estimate.getCuData();
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accel1_d, d1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        fistaUpdate(d1_d, d_accel0_d, d_accel1_d, step, width*height);
        
        // Update d_accel for next iteration
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        // Compute new gradient and cost function
        estimate.setCuData(d1_d);
        holo_estimate.setData(estimate);
        this->calcResidualFrom(&residual, &holo_estimate, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        grad_d = gradient.getData().getCuData(); // grad_d = gradf0
        f = calcObjectiveFunction(residual, mode);
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        //std::cout << "FASTA iteration " << it;
        printf(" new f = %e", f);
        //std::cout << ": f = " << f;// << std::endl;
        std::cout << ", backtracks " << backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << std::endl;
        
        //CHECK_MEMORY("end FASTA iteration");
        CHECK_FOR_ERROR("end FASTA_ITERATION");
    }
    
    if (objective != NULL) *objective = f;

    this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
}

__global__ void calc_residual_kernel
    (float2* resid, float2* measured, float2* estimate, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        /*
         * resid[idx] = measured[idx] - estimate[idx];
         * if (measured[idx] == high_sat)
         *      if (resid[idx] < 0) resid[idx] = 0;
         * if (measured[idx] == low_sat)
         *      if (resid[idx] > 0) resid[idx] = 0;
         */
        
        resid[idx].x = measured[idx].x - estimate[idx].x;
        resid[idx].y = measured[idx].y - estimate[idx].y;
        
        /*
        high_sat = high_sat - 1e-5;
        low_sat = low_sat + 1e-5;
        
        resid[idx].x = (measured[idx].x >= high_sat)? 
            (resid[idx].x < 0)? resid[idx].x = 0 : resid[idx].x
            : resid[idx].x;
        resid[idx].y = (measured[idx].y >= high_sat)? 
            (resid[idx].y < 0)? resid[idx].y = 0 : resid[idx].y
            : resid[idx].y;
        resid[idx].x = (measured[idx].x <= low_sat)? 
            (resid[idx].x > 0)? resid[idx].x = 0 : resid[idx].x
            : resid[idx].x;
        resid[idx].y = (measured[idx].y <= low_sat)? 
            (resid[idx].y > 0)? resid[idx].y = 0 : resid[idx].y
            : resid[idx].y;
        */
        
        //float2 temp;
        //temp.x = measured[idx].x - estimate[idx].x;
        //temp.y = measured[idx].y - estimate[idx].y;
        //temp.x = (temp.x<low_sat)? low_sat : temp.x;
        //temp.y = (temp.y<low_sat)? low_sat : temp.y;
        //temp.x = (temp.x>high_sat)? high_sat : temp.x;
        //temp.y = (temp.y>high_sat)? high_sat : temp.y;
        //resid[idx] = temp;
    }
}

__global__ void calc_residual_inverse_kernel
    (float2* resid, float2* measured, float2* estimate, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        resid[idx].x = estimate[idx].x - measured[idx].x;
        resid[idx].y = estimate[idx].y - measured[idx].y;
        //printf("%d) %f = %f - %f\n", idx, resid[idx].x, estimate[idx].x, measured[idx].x);
    }
}

__global__ void check_data_kernel(float2* out, float2* in, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx].x = in[idx].x;
        out[idx].y = in[idx].y;
    }
}

void CompressiveHolo::calcResidual(Hologram* residual)
{
    Hologram estimate(params);
    calcResidual(residual, &estimate);
    estimate.destroy();
}

void CompressiveHolo::calcResidual(Hologram* residual, Hologram* estimate, ResidualMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcResidual");
    DECLARE_TIMING(INV_CALC_RESIDUAL);
    START_TIMING(INV_CALC_RESIDUAL);
    this->backPropagate(estimate, &buffer, RECON_MODE_COMPLEX);
    //estimate.setState(HOLOGRAM_STATE_LOADED);
    //estimate.show();
    
    //measured.setState(HOLOGRAM_STATE_LOADED);
    //measured.show();
    
    /*estimate.applySaturation(measured);
    
    CuMat mdata = measured.getData();//.getReal();
    CuMat edata = estimate.getData();//.getReal();
    //std::cout << "estimate data isAllocated(): " << edata.isAllocated() << std::endl;
    CuMat rdata = residual->getData();
    subtract(mdata, edata, rdata);*/
    
    
    CuMat mdata = measured.getData();
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)estimate->getData().getCuData();
    CuMat rdata = residual->getData();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    float2* rdata_d = (float2*)rdata.getCuData();
    float lowsat, highsat;
    measured.getSaturationLimits(&lowsat, &highsat);
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    if (mode == RM_TRUE_EST)
    {
        calc_residual_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else if (mode == RM_EST_TRUE)
    {
        calc_residual_inverse_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else
    {
        std::cout << "CompressiveHolo::calcResidual error unknown mode" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    rdata.setCuData((void*)rdata_d, height, width, CV_32FC2);
    
    residual->setData(rdata);
    residual->setState(HOLOGRAM_STATE_RESIDUAL);
    
    mdata.destroy();
    //edata.destroy();
    //mdata.destroy();
    //rdata.destroy();
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcResidual");
    SAVE_TIMING(INV_CALC_RESIDUAL);
}

void CompressiveHolo::calcResidualFrom(Hologram* residual,
    Hologram* estimate, ResidualMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcResidual");
    DECLARE_TIMING(INV_CALC_RESIDUAL);
    START_TIMING(INV_CALC_RESIDUAL);
    
    CuMat mdata = measured.getData();
    CuMat edata = estimate->getData();
    CuMat rdata = residual->getData();
    
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)edata.getCuData();
    float2* rdata_d = (float2*)rdata.getCuData();
    
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    
    if (mode == RM_TRUE_EST)
    {
        calc_residual_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else if (mode == RM_EST_TRUE)
    {
        calc_residual_inverse_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else
    {
        std::cout << "CompressiveHolo::calcResidual error unknown mode" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    
    rdata.setCuData((void*)rdata_d, height, width, CV_32FC2);
    edata.setCuData((void*)edata_d, height, width, CV_32FC2);
    mdata.setCuData((void*)mdata_d, height, width, CV_32FC2);
    
    residual->setData(rdata);
    estimate->setData(edata);
    residual->setState(HOLOGRAM_STATE_RESIDUAL);
    
    mdata.destroy();
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcResidual");
    SAVE_TIMING(INV_CALC_RESIDUAL);
}

/*void CompressiveHolo::calcResidual(Hologram* residual, Hologram& bg)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcResidual");
    DECLARE_TIMING(INV_CALC_RESIDUAL);
    START_TIMING(INV_CALC_RESIDUAL);
    Hologram estimate(params);
    this->backPropagate(&estimate, bg, &buffer, RECON_MODE_COMPLEX);
    //estimate.setState(HOLOGRAM_STATE_LOADED);
    //estimate.show();
    
    //measured.setState(HOLOGRAM_STATE_LOADED);
    //measured.show();
    
    //estimate.applySaturation(measured);
    //
    //CuMat mdata = measured.getData();//.getReal();
    //CuMat edata = estimate.getData();//.getReal();
    ////std::cout << "estimate data isAllocated(): " << edata.isAllocated() << std::endl;
    //CuMat rdata = residual->getData();
    //subtract(mdata, edata, rdata);
    
    
    CuMat mdata = measured.getData();
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)estimate.getData().getCuData();
    CuMat rdata = residual->getData();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    float2* rdata_d = (float2*)rdata.getCuData();
    float lowsat, highsat;
    measured.getSaturationLimits(&lowsat, &highsat);
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    calc_residual_kernel<<<dim_grid, dim_block>>>
        (rdata_d, mdata_d, edata_d, lowsat, highsat, numel);
    rdata.setCuData((void*)rdata_d, height, width, CV_32FC2);
    
    residual->setData(rdata);
    residual->setState(HOLOGRAM_STATE_RESIDUAL);
    
    estimate.destroy();
    mdata.destroy();
    //edata.destroy();
    //mdata.destroy();
    //rdata.destroy();
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcResidual");
    SAVE_TIMING(INV_CALC_RESIDUAL);
}*/

__global__ void total_variation_2d_kernel
    (float2* rec, float2* out, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    
    size_t xs = 1;
    size_t ys = Nx;
    
    float2 dx, dy;
    dx.x = (x<Nx-1)? rec[idx+xs].x - rec[idx].x : 0;
    dx.y = (x<Nx-1)? rec[idx+xs].y - rec[idx].y : 0;
    dy.x = (y<Ny-1)? rec[idx+ys].x - rec[idx].x : 0;
    dy.y = (y<Ny-1)? rec[idx+ys].y - rec[idx].y : 0;
    
    float2 dif;
    dif.x = sqrt(dx.x*dx.x + dy.x*dy.x);
    dif.y = sqrt(dx.y*dx.y + dy.y*dy.y);
    
    // Take square root again so we can sum with magnitude
    out[idx].x = sqrt(dif.x);
    out[idx].y = sqrt(dif.y);
}

__global__ void total_variation_3d_kernel
    (float2* rec, float2* out, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    
    size_t xs = 1;
    size_t ys = Nx;
    size_t zs = Nx*Ny;
    
    float2 dx, dy, dz;
    dx.x = (x<Nx-1)? rec[idx+xs].x - rec[idx].x : 0;
    dx.y = (x<Nx-1)? rec[idx+xs].y - rec[idx].y : 0;
    dy.x = (y<Ny-1)? rec[idx+ys].x - rec[idx].x : 0;
    dy.y = (y<Ny-1)? rec[idx+ys].y - rec[idx].y : 0;
    dz.x = (z<Nz-1)? rec[idx+zs].x - rec[idx].x : 0;
    dz.y = (z<Nz-1)? rec[idx+zs].y - rec[idx].y : 0;
    
    float2 dif;
    dif.x = sqrt(dx.x*dx.x + dy.x*dy.x + dz.x*dz.x);
    dif.y = sqrt(dx.y*dx.y + dy.y*dy.y + dz.y*dz.y);
    
    // Take square root again so we can sum with magnitude
    out[idx].x = sqrt(dif.x);
    out[idx].y = sqrt(dif.y);
}

double CompressiveHolo::calcObjectiveFunction(Hologram residual, CompressiveHoloMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcObjectiveFunction");
    
    DECLARE_TIMING(INV_CALC_OBJECTIVE);
    START_TIMING(INV_CALC_OBJECTIVE);
    
    // Compute the squared L2-norm of the residual
    void* resid_d = residual.getData().getCuData();
    
    buffer.allocateCuData(width,height,depth,sizeof(float2));
    //void* buffer1_d = buffer.getCuData();
    //cudaMemcpy(buffer1_d, resid_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    
    CuMat compute_data;
    compute_data.setCuData(resid_d, height, width, 1, CV_32FC2);
    OpticalField compute;
    compute.setData(compute_data);
    
    float resid_norm = -1;
    compute.calcSumMagnitude(resid_norm);
    //if (mode == COMPRESSIVE_MODE_FASTA_L1) return 0.5 * resid_norm;
    //std::cout << "  residual norm = " << resid_norm << ", ";//<< std::endl;
    
    // Compute the total variation norm
    buffer.allocateCuData(width,height,depth,sizeof(float2));
    float2* buffer2_d = (float2*)buffer.getCuData();
    float2* recon_d = (float2*)this->data.getCuData();
    
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(ceil(width/block_dim.x), ceil(height/block_dim.y), depth);
    switch (mode)
    {
    case COMPRESSIVE_MODE_BRADY_TV_2D:
    {
        total_variation_2d_kernel<<<grid_dim,block_dim>>>
            (recon_d, buffer2_d, width, height, depth);
        break;
    }
    case COMPRESSIVE_MODE_BRADY_TV_3D:
    {
        total_variation_3d_kernel<<<grid_dim,block_dim>>>
            (recon_d, buffer2_d, width, height, depth);
        break;
    }
    case COMPRESSIVE_MODE_BRADY_L1: {}
    case COMPRESSIVE_MODE_FISTA_L1:
    case COMPRESSIVE_MODE_FASTA_L1:
    {
        cudaMemcpy(buffer2_d, recon_d, width*height*depth*sizeof(float2), cudaMemcpyDeviceToDevice);
        break;
    }
    default:
    {
        std::cerr << "Error: CompressiveHolo::calcObjectiveFunction: Unknown mode" << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
    }
    }
    
    CuMat compute_data2;
    compute_data2.setCuData((void*)buffer2_d, height, width, depth, CV_32FC2);
    
    //std::cout << "nnz = " << compute_data2.countNonZeros() << std::endl;
    
    OpticalField compute2;
    compute2.setData(compute_data2);
    float tv_norm = -1;
    switch (mode)
    {
    case COMPRESSIVE_MODE_BRADY_L1:
    {
        compute2.calcL1Norm(tv_norm);
        break;
    }
    case COMPRESSIVE_MODE_FASTA_L1:
    {
        compute2.calcL1NormComplex(tv_norm);
        break;
    }
    default:
    {
        compute2.calcSumMagnitude(tv_norm);
    }
    }
    
    assert(tau == regularization);
    double objective = 0.5*resid_norm + tau*tv_norm;
    
    //buffer.destroy();
    //compute.destroy();
    //compute_data.destroy();
    //compute2.destroy();
    //compute_data2.destroy();
    
    SAVE_TIMING(INV_CALC_OBJECTIVE);
    CHECK_FOR_ERROR("end CompressiveHolo::calcObjectiveFunction");
    return objective;
}

//============================= ACCESS     ===================================
//============================= INQUIRY    ===================================
//============================= TESTS      ===================================

double CompressiveHolo::test_lineSearchLimit()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    double L = estimateLipschitz();
    stepsize = (2.0 / L) / 10.0;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    Hologram previous_estimate;
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    //double tau0 = tau1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    // Non-monotone backtracking line search
    int backtrack_count = 0;
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    f = calcObjectiveFunction(residual, mode);
    int look_back = std::min(10+1, 0+1);
    double M = *std::max_element(objective_values.end()-look_back, objective_values.end()-1);
    double lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
    
    return lim;
}

void CompressiveHolo::test_lineSearch_predicates(OpticalField* x0, OpticalField* grad0)
{
    printf("\ndefault stepsize was %f\n", stepsize);
    double L = estimateLipschitz();
    stepsize = (2.0 / L) / 10.0;
    printf("\nnew stepsize is %f\n", stepsize);
    
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    Hologram previous_estimate;
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    //double tau0 = tau1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    this->data.setCuData(x1_d);
    
    CuMat x0_data = x0->getData();
    x0_data.setCuData(x0_d, width, height, depth, CV_32FC2);
    x0->setData(x0_data);
    
    CuMat grad0_data =grad0->getData();
    grad0_data.setCuData(grad_d, width, height, depth, CV_32FC2);
    grad0->setData(grad0_data);
}

void quick_print_data(void* data_d, char* str)
{
    size_t w = 256;
    size_t h = 256;
    float2* data_h = (float2*)malloc(w*h*sizeof(float2));
    cudaMemcpy(data_h, data_d, w*h*sizeof(float2), cudaMemcpyDeviceToHost);
    printf("%s:\n[", str);
    for (size_t y = 180; y < 189; ++y)
    {
        for (size_t x = 126; x < 135; ++x)
        {
            size_t idx = y*w + x;
            printf("%f, ", data_h[idx].x);
        }
        printf("\b\b;\n");
    }
    printf("\b]\n");
    free(data_h);
}

void CompressiveHolo::test_fistaUpdate_results(OpticalField* x1, CuMat* d1)
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    double L = estimateLipschitz();
    stepsize = (2.0 / L) / 10.0;
    printf("CompressiveHolo::test_fistaUpdate_results: stepsize = %f\n", stepsize);
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    //double tau0 = tau1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    f = calcObjectiveFunction(residual, mode);
    
    // Skip implementing stopping criteria check
    
    // Begin FISTA acceleration steps

    // Update acceleration parameters
    alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;

    // Update x1
    buffer.allocateCuData(width,height,depth,sizeof(float2));
    void* y1_d = buffer.getCuData();
    cudaMemcpy(y1_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);
    double step = (alpha0-1) / alpha1;
    fistaUpdate(x1_d, y0_d, y1_d, step, volume_size);
    cudaMemcpy(y0_d, y1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);

    // Update d1
    CuMat estimate = holo_estimate.getData();
    void* d1_d = estimate.getCuData();
    
    printf("\nCompressiveHolo test_fistaUpdates\n");
    printf("\nBefore swaps:\n");
    quick_print_data(d_accel0_d, "previous");
    quick_print_data(d_accel1_d, "current");
    quick_print_data(d1_d, "estimate");
    
    cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_accel1_d, d1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    
    printf("\nAfter swaps:\n");
    quick_print_data(d_accel0_d, "previous");
    quick_print_data(d_accel1_d, "current");
    quick_print_data(d1_d, "estimate");
    
    //void* d_accel0_d;
    //void* d_accel1_d;
    //cudaMalloc((void**)&d_accel0_d, width*height*sizeof(float2));
    //cudaMalloc((void**)&d_accel1_d, width*height*sizeof(float2));
    //cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    //cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    //cudaMemcpy(d_accel1_d, d1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    
    printf("fistaUpdate step = %f\n", step);
    fistaUpdate(d1_d, d_accel0_d, d_accel1_d, step, width*height);
    
    printf("\nAfter fistaUpdate:\n");
    quick_print_data(d_accel0_d, "previous");
    quick_print_data(d_accel1_d, "current");
    quick_print_data(d1_d, "estimate");
    
    CuMat x1_data = x1->getData();
    x1_data.setCuData(x1_d, width, height, depth, CV_32FC2);
    x1->setData(x1_data);
    
    d1->setCuData(d1_d, width, height, CV_32FC2);
    
    return;
}

CuMat CompressiveHolo::test_denoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    stepsize = 0.017536;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    //this->reconstruct(holo, RECON_MODE_COMPLEX);
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
        
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    CuMat return_data;
    return_data.allocateCuData(width, height, 1, sizeof(float2));
    void* return_data_d = return_data.getCuData();
    cudaMemcpy(return_data_d, x1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
    
    return return_data;
}

Hologram CompressiveHolo::test_estimatedHoloOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    stepsize = 0.017536;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    //this->reconstruct(holo, RECON_MODE_COMPLEX);
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
        
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    // Non-monotone backtracking line search
    int backtrack_count = 0;
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    //f = calcObjectiveFunction(residual, mode);
    
    return holo_estimate;
}

Hologram CompressiveHolo::test_residualOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    stepsize = 0.017536;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    //this->reconstruct(holo, RECON_MODE_COMPLEX);
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
        
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    // Non-monotone backtracking line search
    int backtrack_count = 0;
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    //f = calcObjectiveFunction(residual, mode);
    
    return residual;
}

double CompressiveHolo::test_objectiveOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    stepsize = 0.017536;
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    //this->reconstruct(holo, RECON_MODE_COMPLEX);
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
        
    // Rename iterates 
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(x1_d, x0_d, grad_d, mode);
    
    // Non-monotone backtracking line search
    int backtrack_count = 0;
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    f = calcObjectiveFunction(residual, mode);
    
    return f;
}

void CompressiveHolo::test_denoise_it1(OpticalField* x1)
{
    printf("Begin CompressiveHolo::test_denoise_it1\n");
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    if (!force_stepsize)
    {
        printf("Estimating stepsize with Lipschitz\n");
        double L = this->estimateLipschitz();
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
    }
    printf("stepsize = %f\n", stepsize);
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
    for (int it = 0; it < num_twist_iterations; ++it)
    {
        std::cout << "FASTA iteration " << it << " of " << num_twist_iterations << ": ";
        std::cout << std::endl;
        
        // Rename iterates 
        CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(x1_d, x0_d, grad_d, mode);
        if (it == 1)
        {
            CuMat x1_data = x1->getData();
            x1_data.setCuData(x1_d, width, height, depth, CV_32FC2);
            x1->setData(x1_data);
            return;
        }
        
        // Non-monotone backtracking line search
        int backtrack_count = 0;
        this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
        f = calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1);
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
        double ratio = f / (M+lim);
        printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
        while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
        {
            its_without_backtrack = 0;
            stepsize *= stepsize_shrinkage;
            this->denoise(x1_d, x0_d, grad_d, mode);
            this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
            f = calcObjectiveFunction(residual, mode);
            lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
            ratio = f / (M+lim);
            backtrack_count++;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
            return;
        }
        its_without_backtrack++;
        printf("  after backtracks f = %f\n", f);

        // Update acceleration parameters
        if (fistaRestart(x0_d, x1_d, y0_d))
        {
            std::cout << "restarted FISTA parameter" << std::endl << "  ";
            alpha0 = 1;
        }
        alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;

        // Update x1
        buffer.allocateCuData(width,height,depth,sizeof(float2));
        void* y1_d = buffer.getCuData();
        cudaMemcpy(y1_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);
        double step = (alpha0-1) / alpha1;
        fistaUpdate(x1_d, y0_d, y1_d, step, volume_size);
        cudaMemcpy(y0_d, y1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);

        // Update d1
        CuMat estimate = holo_estimate.getData();
        void* d1_d = estimate.getCuData();
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accel1_d, d1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        fistaUpdate(d1_d, d_accel0_d, d_accel1_d, step, width*height);
        
        // Update d_accel for next iteration
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        // Compute new gradient and cost function
        estimate.setCuData(d1_d);
        holo_estimate.setData(estimate);
        this->calcResidualFrom(&residual, &holo_estimate, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        grad_d = gradient.getData().getCuData(); // grad_d = gradf0
        f = calcObjectiveFunction(residual, mode);
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << std::endl;
        
        //CHECK_MEMORY("end FASTA iteration");
        CHECK_FOR_ERROR("end FASTA_ITERATION");
    }
    
    this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
}

void CompressiveHolo::test_latest_error(CuMat* x1_p2)
{
    printf("Begin CompressiveHolo::test_denoise_it1\n");
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    if (!force_stepsize)
    {
        printf("Estimating stepsize with Lipschitz\n");
        double L = this->estimateLipschitz();
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
    }
    printf("stepsize = %f\n", stepsize);
    
    // Reconstruct to initialize volume
    void* x1_d = this->data.getCuData();
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize FISTA iteration variables
    void* x0_d = data_nm1.getCuData();
    void* y0_d = data_nm2.getCuData();
    CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(y0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
    double alpha1 = 1.0;
    
    Hologram holo_estimate(params); // holo_estimate = d1
    CuMat prev_estimate;
    CuMat current_estimate;
    prev_estimate.allocateCuData(width, height, 1, CV_32FC2);
    current_estimate.allocateCuData(width, height, 1, CV_32FC2);
    void* d_accel0_d = prev_estimate.getCuData();
    void* d_accel1_d = current_estimate.getCuData();
    cudaMemset(d_accel0_d, 0, width*height*sizeof(float2));
    cudaMemset(d_accel1_d, 0, width*height*sizeof(float2));
    
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    double f = calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData(); // grad_d = gradf0
    
    CHECK_FOR_ERROR("FASTA before iterations");
    int its_without_backtrack = 0;
    
    for (int it = 0; it < num_twist_iterations; ++it)
    {
        std::cout << "FASTA iteration " << it << " of " << num_twist_iterations << ": ";
        std::cout << std::endl;
        
        // Rename iterates 
        CUDA_SAFE_CALL( cudaMemcpy(x0_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice) );
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(x1_d, x0_d, grad_d, mode);
        
        // Non-monotone backtracking line search
        int backtrack_count = 0;
        this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
        
        f = calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1);
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
        double ratio = f / (M+lim);
        printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
        while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
        {
            its_without_backtrack = 0;
            stepsize *= stepsize_shrinkage;
            this->denoise(x1_d, x0_d, grad_d, mode);
            this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
            f = calcObjectiveFunction(residual, mode);
            lim = calcLineSearchLimit(x1_d, x0_d, grad_d);
            ratio = f / (M+lim);
            backtrack_count++;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
            return;
        }
        its_without_backtrack++;
        printf("  after backtracks f = %f\n", f);

        // Update acceleration parameters
        if (fistaRestart(x0_d, x1_d, y0_d))
        {
            std::cout << "restarted FISTA parameter" << std::endl << "  ";
            alpha0 = 1;
        }
        alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;

        // Update x1
        buffer.allocateCuData(width,height,depth,sizeof(float2));
        void* y1_d = buffer.getCuData();
        cudaMemcpy(y1_d, x1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);
        double step = (alpha0-1) / alpha1;
        
        /*if (it == 1)
        {
            x1_p2->setCuData(y1_d + width*height*2*sizeof(float2), width, height, 1, CV_32FC2);
            //*x1_p2 = residual.getData();
            return;
        }//*/
        
        fistaUpdate(x1_d, y0_d, y1_d, step, volume_size);
        cudaMemcpy(y0_d, y1_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice);
        
		// Original (incorrect pointer arithmetic on void*):
		// x1_p2->setCuData(x1_d + width*height*2*sizeof(float2), width, height, 1, CV_32FC2);

		if (it == 1)
		{
			// Cast void* to float2* before doing pointer arithmetic
			float2* x1_d_f2 = static_cast<float2*>(x1_d);
			x1_p2->setCuData(x1_d_f2 + width * height * 2, width, height, 1, CV_32FC2);
			//*x1_p2 = residual.getData();
			printf("fistaUpdate step = %f\n", step);
			return;
		}//*/

        // Update d1
        CuMat estimate = holo_estimate.getData();
        void* d1_d = estimate.getCuData();
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_accel1_d, d1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        fistaUpdate(d1_d, d_accel0_d, d_accel1_d, step, width*height);
        
        // Update d_accel for next iteration
        cudaMemcpy(d_accel0_d, d_accel1_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice);
        
        // Compute new gradient and cost function
        estimate.setCuData(d1_d);
        holo_estimate.setData(estimate);
        this->calcResidualFrom(&residual, &holo_estimate, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        grad_d = gradient.getData().getCuData(); // grad_d = gradf0
        f = calcObjectiveFunction(residual, mode);
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << std::endl;
        
        //CHECK_MEMORY("end FASTA iteration");
        CHECK_FOR_ERROR("end FASTA_ITERATION");
    }
    
    this->state = OPTICALFIELD_STATE_FULL_COMPLEX;
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
}

/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////

void CompressiveHolo::denoise(void* x_d, void* xm1_d, void* grad_d, CompressiveHoloMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::denoise");
    DECLARE_TIMING(INV_DENOISE);
    START_TIMING(INV_DENOISE);
    switch (mode)
    {
    case COMPRESSIVE_MODE_BRADY_TV_2D:
    {
        denoise2d(x_d, xm1_d, grad_d);
        break;
    }
    case COMPRESSIVE_MODE_BRADY_TV_3D:
    {
        denoise3d(x_d, xm1_d, grad_d);
        break;
    }
    case COMPRESSIVE_MODE_BRADY_L1:
    {
        softThreshold(x_d, xm1_d, grad_d);
        break;
    }
    case COMPRESSIVE_MODE_FASTA_L1:
    {
        softThreshold(x_d, xm1_d, grad_d, stepsize, regularization);
        break;
    }
    default:
    {
        std::cerr << "CompressiveHolo::denoise: Unknown mode" << std::endl;
        throw HOLO_ERROR_UNKNOWN_MODE;
    }
    }
    SAVE_TIMING(INV_DENOISE);
    CHECK_FOR_ERROR("end CompressiveHolo::denoise");
}

__global__ void projection_2d_kernel
    (float2* xm1, float2* grad, 
     float2* div_pn, float2* pn_x, float2* pn_y,
     double tau, double proj_tau, double max_svd,
     int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    size_t idx_zwrap = (z+1)*Nx*Ny + y*Nx + 0;
    if (idx < Nx*Ny*Nz)
    {
        // First convert inputs
        float2 g;
        g.x = xm1[idx].x + grad[idx].x/max_svd;
        g.y = xm1[idx].y + grad[idx].y/max_svd;
        float lam = 0.5 * tau / max_svd;
        
        float2 ax, ay, b;
        
        /// First-order derivative of (div_pn - g/lam) = v
        float2 v1, v2;
        v1.x = div_pn[idx].x - g.x/lam;
        v1.y = div_pn[idx].y - g.y/lam;
        
        // X direction
        v2.x = (x<Nx-1)? 
               div_pn[idx+1].x - (xm1[idx+1].x + grad[idx+1].x/max_svd)/lam 
               : 0;
        v2.y = (x<Nx-1)? 
               div_pn[idx+1].y - (xm1[idx+1].y + grad[idx+1].y/max_svd)/lam 
               : 0;
        v2.x = (x==Nx-1)? (z>0 && z<Nz-1)? div_pn[idx_zwrap].x - (xm1[idx_zwrap].x + grad[idx_zwrap].x/max_svd)/lam
                               : 0
                        : v2.x;
        v2.y = (x==Nx-1)? (z>0 && z<Nz-1)? div_pn[idx_zwrap].y - (xm1[idx_zwrap].y + grad[idx_zwrap].y/max_svd)/lam
                               : 0
                        : v2.y;
        ax.x = (x<Nx-1)? v2.x - v1.x : 0;
        ax.y = (x<Nx-1)? v2.y - v1.y : 0;
        //ax.x = v2.x - v1.x;
        //ax.y = v2.y - v1.y;
        ax.x = (z>0 && z<Nz-1)? v2.x - v1.x : 0;
        ax.y = (z>0 && z<Nz-1)? v2.y - v1.y : 0;
        
        // Y direction
        v2.x = (y<Ny-1)? 
               div_pn[idx+Nx].x - (xm1[idx+Nx].x + grad[idx+Nx].x/max_svd)/lam 
               : 0;
        v2.y = (y<Ny-1)? 
               div_pn[idx+Nx].y - (xm1[idx+Nx].y + grad[idx+Nx].y/max_svd)/lam 
               : 0;
        ay.x = (y<Ny-1)? v2.x - v1.x : 0;
        ay.y = (y<Ny-1)? v2.y - v1.y : 0;
        
        b.x = sqrt(ax.x*ax.x + ay.x*ay.x);
        b.y = sqrt(ax.y*ax.y + ay.y*ay.y);
        
        /// Update pn
        pn_x[idx].x = (pn_x[idx].x + proj_tau*ax.x) / (1 + proj_tau*b.x);
        pn_x[idx].y = (pn_x[idx].y + proj_tau*ax.y) / (1 + proj_tau*b.y);
        pn_y[idx].x = (pn_y[idx].x + proj_tau*ay.x) / (1 + proj_tau*b.x);
        pn_y[idx].y = (pn_y[idx].y + proj_tau*ay.y) / (1 + proj_tau*b.y);
    }
}

__global__ void div_pn_2d_kernel
    (float2* div_pn, float2* pn_x, float2* pn_y, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    size_t idx_zwrap = (z-1)*Nx*Ny + y*Nx + (Nx-1);
    if (idx < Nx*Ny*Nz)
    {
        float2 yx, yy;
        yx.x = (x>0)? 
               (x<Nx-1)? pn_x[idx].x - pn_x[idx - 1].x : -pn_x[idx-1].x
               : pn_x[idx].x;
        yx.y = (x>0)? 
               (x<Nx-1)? pn_x[idx].y - pn_x[idx - 1].y : -pn_x[idx-1].y
               : pn_x[idx].y;
        yx.x = (x==0 && z>0)? pn_x[idx].x - pn_x[idx_zwrap].x : yx.x;
        yx.y = (x==0 && z>0)? pn_x[idx].y - pn_x[idx_zwrap].y : yx.y;
        yx.x = (x==Nx-1 && z>0)? pn_x[idx].x - pn_x[idx - 1].x : yx.x;
        yx.y = (x==Nx-1 && z>0)? pn_x[idx].y - pn_x[idx - 1].y : yx.y;
        
        yy.x = (y>0)? 
               (y<Ny-1)? pn_y[idx].x - pn_y[idx-Nx].x : -pn_y[idx-Nx].x
               : pn_y[idx].x;
        yy.y = (y>0)? 
               (y<Ny-1)? pn_y[idx].y - pn_y[idx-Nx].y : -pn_y[idx-Nx].y
               : pn_y[idx].y;
        
        div_pn[idx].x = yx.x + yy.x;
        div_pn[idx].y = yx.y + yy.y;
    }
}

__global__ void denoise_output_kernel
    (float2* x_d, float2* xm1, float2* grad, float2* div_pn,
     double max_svd, double tau, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    if (idx < Nx*Ny*Nz)
    {
        double lam = 0.5 * tau / max_svd;
        x_d[idx].x = xm1[idx].x + grad[idx].x/max_svd - lam*div_pn[idx].x;
        x_d[idx].y = xm1[idx].y + grad[idx].y/max_svd - lam*div_pn[idx].y;
    }
}

void CompressiveHolo::denoise2d(void* x_d, void* xm1_d, void* grad_d)
{
    buffer.allocateCuData(width, height, depth, sizeof(float2));
    float2* div_pn_d = (float2*)buffer.getCuData();
    cudaMemset(div_pn_d, 0, width*height*depth*sizeof(float2));
    
    // Knowing that this TV method only uses 2D data, can process chunks
    // of slices independently to save memory. 
    // If the data is very small and extra space is available, allocate
    // 3 more buffers. Otherwise, break the single buffer into 4
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    if (free_byte/2 > buffer.getDataSize())
    {
        opt_buffer2.allocateCuData(width, height, depth, sizeof(float2));
        opt_buffer3.allocateCuData(width, height, depth, sizeof(float2));
    }
    float2 *pn_x_d, *pn_y_d;
    size_t tile_depth;
    if (opt_buffer2.isAllocated() && opt_buffer3.isAllocated())
    {
        tile_depth = depth;
        pn_x_d = (float2*)opt_buffer2.getCuData();
        pn_y_d = (float2*)opt_buffer3.getCuData();
    }
    else
    {
        tile_depth = depth / 3;
        pn_x_d = div_pn_d + 1*width*height*tile_depth;
        pn_y_d = div_pn_d + 2*width*height*tile_depth;
    }
    
    // Initialize to zeros
    size_t buffer_size = width*height*tile_depth*sizeof(float2);
    cudaMemset(div_pn_d, 0, buffer_size);
    cudaMemset(pn_x_d, 0, buffer_size);
    cudaMemset(pn_y_d, 0, buffer_size);
    
    dim3 block_dim(16,16,1);
    dim3 grid_dim(ceil((double)width  / (double)block_dim.x), 
                  ceil((double)height / (double)block_dim.y), 
                  ceil((double)tile_depth  / (double)block_dim.z));
    
    for (int ztile = 0; ztile < depth; ztile += tile_depth)
    {
        for (int i = 0; i < num_projection_iterations; ++i)
        {
            projection_2d_kernel<<<grid_dim, block_dim>>>
                ((float2*)xm1_d, (float2*)grad_d, div_pn_d, pn_x_d, pn_y_d,
                 tau, projection_tau, max_svd, width, height, tile_depth);
            cudaDeviceSynchronize();
            div_pn_2d_kernel<<<grid_dim, block_dim>>>
                (div_pn_d, pn_x_d, pn_y_d, width, height, tile_depth);
            cudaDeviceSynchronize();
        }
        
        denoise_output_kernel<<<grid_dim, block_dim>>>
            ((float2*)x_d, (float2*)xm1_d, (float2*)grad_d, div_pn_d,
             max_svd, tau, width, height, tile_depth);
        cudaDeviceSynchronize();
        // x = (xm1-grad/max_svd) - lam*div_pn
    }
    
    CHECK_FOR_ERROR("end CompressiveHolo::denoise");
}

__global__ void projection_3d_kernel
    (float2* xm1, float2* grad, 
     float2* div_pn, float2* pn_x, float2* pn_y, float2* pn_z,
     double tau, double proj_tau, double max_svd,
     int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    
    size_t xs = 1;
    size_t ys = Nx;
    size_t zs = Nx*Ny;
    
    if (idx < Nx*Ny*Nz)
    {
        // First convert inputs
        float2 g;
        g.x = xm1[idx].x + grad[idx].x/max_svd;
        g.y = xm1[idx].y + grad[idx].y/max_svd;
        float lam = 0.5 * tau / max_svd;
        
        float2 ax, ay, az, b;
        
        /// First-order derivative of (div_pn - g/lam) = v
        float2 v1, v2;
        v1.x = div_pn[idx].x - g.x/lam;
        v1.y = div_pn[idx].y - g.y/lam;
        
        // X direction
        v2.x = (x<Nx-1)? 
               div_pn[idx+xs].x - (xm1[idx+xs].x + grad[idx+xs].x/max_svd)/lam 
               : 0;
        v2.y = (x<Nx-1)? 
               div_pn[idx+xs].y - (xm1[idx+xs].y + grad[idx+xs].y/max_svd)/lam 
               : 0;
        ax.x = (x<Nx-1)? v2.x - v1.x : 0;
        ax.y = (x<Nx-1)? v2.y - v1.y : 0;
        
        // Y direction
        v2.x = (y<Ny-1)? 
               div_pn[idx+ys].x - (xm1[idx+ys].x + grad[idx+ys].x/max_svd)/lam 
               : 0;
        v2.y = (y<Ny-1)? 
               div_pn[idx+ys].y - (xm1[idx+ys].y + grad[idx+ys].y/max_svd)/lam 
               : 0;
        ay.x = (y<Ny-1)? v2.x - v1.x : 0;
        ay.y = (y<Ny-1)? v2.y - v1.y : 0;
        
        // Z direction
        v2.x = (z<Nz-1)? 
               div_pn[idx+zs].x - (xm1[idx+zs].x + grad[idx+zs].x/max_svd)/lam 
               : 0;
        v2.y = (z<Nz-1)? 
               div_pn[idx+zs].y - (xm1[idx+zs].y + grad[idx+zs].y/max_svd)/lam 
               : 0;
        az.x = (z<Nz-1)? v2.x - v1.x : 0;
        az.y = (z<Nz-1)? v2.y - v1.y : 0;
        
        b.x = sqrt(ax.x*ax.x + ay.x*ay.x + az.x*az.x);
        b.y = sqrt(ax.y*ax.y + ay.y*ay.y + az.y*az.y);
        
        /// Update pn
        pn_x[idx].x = (pn_x[idx].x + proj_tau*ax.x) / (1 + proj_tau*b.x);
        pn_x[idx].y = (pn_x[idx].y + proj_tau*ax.y) / (1 + proj_tau*b.y);
        pn_y[idx].x = (pn_y[idx].x + proj_tau*ay.x) / (1 + proj_tau*b.x);
        pn_y[idx].y = (pn_y[idx].y + proj_tau*ay.y) / (1 + proj_tau*b.y);
        pn_z[idx].x = (pn_z[idx].x + proj_tau*az.x) / (1 + proj_tau*b.x);
        pn_z[idx].y = (pn_z[idx].y + proj_tau*az.y) / (1 + proj_tau*b.y);
    }
}

__global__ void div_pn_3d_kernel
    (float2* div_pn, float2* pn_x, float2* pn_y, float2* pn_z, int Nx, int Ny, int Nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t idx = z*Nx*Ny + y*Nx + x;
    
    size_t xs = 1;
    size_t ys = Nx;
    size_t zs = Nx*Ny;
    
    if (idx < Nx*Ny*Nz)
    {
        float2 yx, yy, yz;
        yx.x = (x>0)? 
               (x<Nx-1)? pn_x[idx].x - pn_x[idx-xs].x : -pn_x[idx-xs].x
               : pn_x[idx].x;
        yx.y = (x>0)? 
               (x<Nx-1)? pn_x[idx].y - pn_x[idx-xs].y : -pn_x[idx-xs].y
               : pn_x[idx].y;
        
        yy.x = (y>0)? 
               (y<Ny-1)? pn_y[idx].x - pn_y[idx-ys].x : -pn_y[idx-ys].x
               : pn_y[idx].x;
        yy.y = (y>0)? 
               (y<Ny-1)? pn_y[idx].y - pn_y[idx-ys].y : -pn_y[idx-ys].y
               : pn_y[idx].y;
        
        yz.x = (z>0)? 
               (z<Nz-1)? pn_z[idx].x - pn_z[idx-zs].x : -pn_z[idx-zs].x
               : pn_z[idx].x;
        yz.y = (z>0)? 
               (z<Nz-1)? pn_z[idx].y - pn_z[idx-zs].y : -pn_z[idx-zs].y
               : pn_z[idx].y;
        
        div_pn[idx].x = yx.x + yy.x + yz.x;
        div_pn[idx].y = yx.y + yy.y + yz.y;
    }
}

void CompressiveHolo::denoise3d(void* x_d, void* xm1_d, void* grad_d)
{
    buffer.allocateCuData(width, height, depth, sizeof(float2));
    float2* div_pn_d = (float2*)buffer.getCuData();
    cudaMemset(div_pn_d, 0, width*height*depth*sizeof(float2));
    
    // Knowing that this TV method only uses 2D data, can process chunks
    // of slices independently to save memory. 
    // If the data is very small and extra space is available, allocate
    // 3 more buffers. Otherwise, break the single buffer into 4
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    bool buffers_allocated = false;
    if (free_byte/3 > buffer.getDataSize())
    {
        opt_buffer2.allocateCuData(width, height, depth, sizeof(float2));
        opt_buffer3.allocateCuData(width, height, depth, sizeof(float2));
        opt_buffer4.allocateCuData(width, height, depth, sizeof(float2));
        buffers_allocated = true;
    }
    float2 *pn_x_d, *pn_y_d, *pn_z_d;
    size_t tile_depth;
    if (buffers_allocated)
    {
        tile_depth = depth;
        pn_x_d = (float2*)opt_buffer2.getCuData();
        pn_y_d = (float2*)opt_buffer3.getCuData();
        pn_z_d = (float2*)opt_buffer4.getCuData();
    }
    else
    {
        std::cerr << "CompressiveHolo::denoise3d: Required memory is too large" << std::endl;
        throw HOLO_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize to zeros
    size_t buffer_size = width*height*tile_depth*sizeof(float2);
    cudaMemset(div_pn_d, 0, buffer_size);
    cudaMemset(pn_x_d, 0, buffer_size);
    cudaMemset(pn_y_d, 0, buffer_size);
    cudaMemset(pn_z_d, 0, buffer_size);
    
    dim3 block_dim(16,16,1);
    dim3 grid_dim(ceil((double)width  / (double)block_dim.x), 
                  ceil((double)height / (double)block_dim.y), 
                  ceil((double)tile_depth  / (double)block_dim.z));
    
    for (int ztile = 0; ztile < depth; ztile += tile_depth)
    {
        for (int i = 0; i < num_projection_iterations; ++i)
        {
            projection_3d_kernel<<<grid_dim, block_dim>>>
                ((float2*)xm1_d, (float2*)grad_d, div_pn_d, pn_x_d, pn_y_d, pn_z_d,
                 tau, projection_tau, max_svd, width, height, tile_depth);
            cudaDeviceSynchronize();
            div_pn_3d_kernel<<<grid_dim, block_dim>>>
                (div_pn_d, pn_x_d, pn_y_d, pn_z_d, width, height, tile_depth);
            cudaDeviceSynchronize();
        }
        
        denoise_output_kernel<<<grid_dim, block_dim>>>
            ((float2*)x_d, (float2*)xm1_d, (float2*)grad_d, div_pn_d,
             max_svd, tau, width, height, tile_depth);
        cudaDeviceSynchronize();
        // x = (xm1-grad/max_svd) - lam*div_pn
    }
    
    CHECK_FOR_ERROR("end CompressiveHolo::denoise");
}

__global__ void soft_threshold_kernel
    (float2* out_x, float2* xm1, float2* grad, double tau, double max_svd, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        float2 temp_out;
        in.x = xm1[idx].x + grad[idx].x/max_svd;
        in.y = xm1[idx].y + grad[idx].y/max_svd;
        float T = tau / max_svd;
        
        //float absx = sqrt(in.x*in.x + in.y*in.y);
        float absx = (in.x*in.x + in.y*in.y);
        float y = max(absx - T, 0.0);
        y = y / (y + T);
        
        temp_out.x = y * in.x;
        temp_out.y = y * in.y;
        
        /*
         * This is how it was done in Brady's code
        float y = max(abs(in.x) - T, 0.0);
        y = y / (y + T);
        temp_out.x = y * in.x;
        y = max(abs(in.y) - T, 0.0);
        y = y / (y + T);
        temp_out.y = y * in.y;
        */
        
        /*//atan2(im, re);
        float amp = sqrt(temp_out.x*temp_out.x + temp_out.y*temp_out.y);
        float phase = atan2(temp_out.y, temp_out.x);
        phase = (phase > M_PI)? phase - 2*M_PI : phase;
        float T2 = 0;
        y = max(phase - T2, 0.0);
        //y = y / (y + T2);
        
        temp_out.x = amp * cos(y);
        temp_out.y = amp * sin(y);*/
        
        out_x[idx].x = temp_out.x;
        out_x[idx].y = temp_out.y;
    }
}

__global__ void soft_threshold_kernel
    (float2* out, float2* in, double thr, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        //float absx = sqrt(in[idx].x*in[idx].x + in[idx].y*in[idx].y);
        float absx = (in[idx].x*in[idx].x + in[idx].y*in[idx].y);
        float y = max(absx - thr, 0.0);
        y = y / (y + thr);
        
        out[idx].x = y * in[idx].x;
        out[idx].y = y * in[idx].y;
    }
}

__global__ void soft_threshold_FASTA_kernel
    (float2* out_x, float2* xm1, float2* grad, double stepsize, double regularization, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = xm1[idx].x - stepsize*grad[idx].x;
        in.y = xm1[idx].y - stepsize*grad[idx].y;
        
        float T = stepsize * regularization;
        
        //float absx = sqrt(in.x*in.x + in.y*in.y);
        float absx = abs(in.x);
        float absy = abs(in.y);
        float2 y;
        y.x = max(absx - T, 0.0);
        y.y = max(absy - T, 0.0);
        y.x = y.x / (y.x + T);
        y.y = y.y / (y.y + T);
        
        out_x[idx].x = y.x * in.x;
        out_x[idx].y = y.y * in.y;
    }
}

void CompressiveHolo::softThreshold(void* x_d, void* xm1_d, void* grad_d)
{
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    //printf("softThreshold: T = %f\n", tau/max_svd);
    soft_threshold_kernel<<<grid_dim, block_dim>>>
        ((float2*)x_d, (float2*)xm1_d, (float2*)grad_d,
         tau, max_svd, width*height*depth);
    
    CHECK_FOR_ERROR("end CompressiveHolo::softThreshold");
}

void CompressiveHolo::softThreshold(void* x_d, void* xm1_d, void* grad_d, double stepsize, double regularization)
{
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
        ((float2*)x_d, (float2*)xm1_d, (float2*)grad_d,
         stepsize, regularization, width*height*depth);
    
    CHECK_FOR_ERROR("end CompressiveHolo::softThreshold");
}

void CompressiveHolo::denoise_FISTA(void* out_d, void* in_d)
{
    double thr = this->tau / this->max_svd;
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    soft_threshold_kernel<<<grid_dim, block_dim>>>
        ((float2*)out_d, (float2*)in_d, thr, width*height*depth);
}


__global__ void two_step_update_kernel
    (float2* out, float2* x, float2* xm1, float2* xm2, 
     float alpha, float beta, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx].x = (alpha-beta)*xm1[idx].x + (1-alpha)*xm2[idx].x + beta*x[idx].x;
        out[idx].y = (alpha-beta)*xm1[idx].y + (1-alpha)*xm2[idx].y + beta*x[idx].y;
    }
}

void CompressiveHolo::twoStepUpdate(void* out_d, void* x_d, void* xm1_d, void* xm2_d)
{
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    two_step_update_kernel<<<grid_dim, block_dim>>>
        ((float2*)out_d, (float2*)x_d, (float2*)xm1_d, (float2*)xm2_d,
         alpha, beta, width*height*depth);
    
    CHECK_FOR_ERROR("end CompressiveHolo::twoStepUpdate");
}

__global__ void updateY_kernel(float2* y_d, float2* grad_d, double L, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        y_d[idx].x = y_d[idx].x - (2/L) * grad_d[idx].x;
        y_d[idx].y = y_d[idx].y - (2/L) * grad_d[idx].y;
    }
}

void CompressiveHolo::updateY(void* y_d, void* grad_d)
{
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    updateY_kernel<<<grid_dim, block_dim>>>
        ((float2*)y_d, (float2*)grad_d, max_svd, width*height*depth);
}

__global__ void fista_update_1_kernel(float2* y, float2* x, float2* z,
    double t_new, double t_old, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        //y[idx].x = x[idx].x + (t_old/t_new)*(z[idx].x - x[idx].x);
        //y[idx].y = x[idx].y + (t_old/t_new)*(z[idx].y - x[idx].y);
        y[idx].x = x[idx].x + ((t_old-1.0)/t_new)*(z[idx].x - x[idx].x);
        y[idx].y = x[idx].y + ((t_old-1.0)/t_new)*(z[idx].y - x[idx].y);
        if (t_old == 1.0)
        {
            y[idx].x = x[idx].x;
            y[idx].y = x[idx].y;
        }
    }
}

__global__ void fista_update_2_kernel(float2* y, float2* x, float2* x_old,
    double t_new, double t_old, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        y[idx].x = x[idx].x + ((t_old-1.0)/t_new)*(x[idx].x - x_old[idx].x);
        y[idx].y = x[idx].y + ((t_old-1.0)/t_new)*(x[idx].y - x_old[idx].y);
        if (t_old == 1.0)
        {
            y[idx].x = x[idx].x;
            y[idx].y = x[idx].y;
        }
    }
}

void CompressiveHolo::fista_update(void* y_iter_d, void* x_iter_d,
    void* x_old_d, void* z_iter_d,
    double t_new, double t_old, bool mono_failed)
{
    // Y=X_iter+t_old/t_new*(Z_iter-X_iter)+(t_old-1)/t_new*(X_iter-X_old);
    
    int block_dim = 256;
    int grid_dim = ceil(width*height*depth / (double)block_dim);
    
    // if mono_failed = false, z_iter_d should be identical to x_iter_d
    // This isn't actaully true for speed so skip subtraction
    // When mono_failed == true, x_iter_d = x_old_d
    if (mono_failed)
    {
        fista_update_1_kernel<<<grid_dim, block_dim>>>
            ((float2*)y_iter_d, (float2*)x_iter_d, (float2*)z_iter_d,
             t_new, t_old, width*height*depth);
    }
    else
    {
        fista_update_2_kernel<<<grid_dim, block_dim>>>
            ((float2*)y_iter_d, (float2*)x_iter_d, (float2*)x_old_d,
             t_new, t_old, width*height*depth);
    }
}

__global__ void fistaUpdate_kernel(float2* x1, float2* y0, float2* y1, double step, size_t size)
{
    // x1 = y1 + step*(y1-y0);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        x1[idx].x = y1[idx].x + step*(y1[idx].x - y0[idx].x);
        x1[idx].y = y1[idx].y + step*(y1[idx].y - y0[idx].y);
    }
}

void CompressiveHolo::fistaUpdate(void* x1_d, void* y0_d, void* y1_d, double step, size_t size)
{
    int block_dim = 256;
    int grid_dim = ceil(size / (double)block_dim);

    fistaUpdate_kernel<<<grid_dim, block_dim>>>
        ((float2*)x1_d, (float2*)y0_d, (float2*)y1_d, step, size);
    
    CHECK_FOR_ERROR("CompressiveHolo::fistaUpdate");
}

template <unsigned int blockSize>
__global__ void linesearch_1_kernel(float2* x1, float2* x0, float2* grad, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        float2 temp;
        temp.x = (x1[i].x - x0[i].x) * grad[i].x;
        temp.y = (x1[i].y - x0[i].y) * grad[i].y;
        sdata[tid] += temp.x + temp.y;
        temp.x = (x1[i+blockSize].x - x0[i+blockSize].x) * grad[i+blockSize].x;
        temp.y = (x1[i+blockSize].y - x0[i+blockSize].y) * grad[i+blockSize].y;
        sdata[tid] += temp.x + temp.y;
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void linesearch_2_kernel(float2* x1, float2* x0, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        float2 temp;
        temp.x = (x1[i].x - x0[i].x);
        temp.y = (x1[i].y - x0[i].y);
        sdata[tid] += temp.x*temp.x + temp.y*temp.y;
        temp.x = (x1[i+blockSize].x - x0[i+blockSize].x);
        temp.y = (x1[i+blockSize].y - x0[i+blockSize].y);
        sdata[tid] += temp.x*temp.x + temp.y*temp.y;
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

double CompressiveHolo::calcLineSearchLimit(void* x1_d, void* x0_d, void* grad_d)
{
    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    
    linesearch_1_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)x1_d, (float2*)x0_d, (float2*)grad_d, buffer_d, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum1 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum1 += buffer_h[i];
    }
    
    linesearch_2_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)x1_d, (float2*)x0_d, buffer_d, size);
    
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum2 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum2 += buffer_h[i];
    }
    
    free(buffer_h);
    cudaFree(buffer_d);
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcLineSearchLimit");
    return sum1 + sum2/(2*stepsize);
}

template <unsigned int blockSize>
__global__ void fistaRestart_kernel(float2* x0, float2* x1, float2* x_accel, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    // Multiply complex number by conjugate for magnitude
    
    while (i < size)
    {
        float2 temp;
        temp.x = (x0[i].x - x1[i].x) * (x1[i].x - x_accel[i].x);
        temp.y = (x0[i].y - x1[i].y) * (x1[i].y - x_accel[i].y);
        sdata[tid] += temp.x + temp.y;
        unsigned int i2 = i + blockSize;
        temp.x = (x0[i2].x - x1[i2].x) * (x1[i2].x - x_accel[i2].x);
        temp.x = (x0[i2].y - x1[i2].y) * (x1[i2].y - x_accel[i2].y);
        sdata[tid] += temp.x + temp.y;
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

bool CompressiveHolo::fistaRestart(void* x0_d, void* x1_d, void* x_accel_d)
{
    // Evaluates (x0(:)-x1(:))'*(x1(:)-x_accel0(:))>0

    int size = width*height*depth;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    
    fistaRestart_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)x0_d, (float2*)x1_d, (float2*)x_accel_d, buffer_d, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum1 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum1 += buffer_h[i];
    }

    free(buffer_h);
    cudaFree(buffer_d);

    CHECK_FOR_ERROR("after CompressiveHolo::fistaRestart");
    return sum1 > 0;
}

template <unsigned int blockSize>
__global__ void subtractionNorm_kernel(float2* data1, float2* data2, float* buffer, int size)
{
    // Adapted from <http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf>
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0.0;
    
    while (i < size)
    {
        float2 temp;
        temp.x = (data1[i].x - data2[i].x) * (data1[i].x - data2[i].x);
        temp.y = (data1[i].y - data2[i].y) * (data1[i].y - data2[i].y);
        sdata[tid] += temp.x + temp.y;
        unsigned int i2 = i + blockSize;
        temp.x = (data1[i2].x - data2[i2].x) * (data1[i2].x - data2[i2].x);
        temp.y = (data1[i2].y - data2[i2].y) * (data1[i2].y - data2[i2].y);
        sdata[tid] += temp.x + temp.y;
        i += gridSize;
    }
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; __syncthreads();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; __syncthreads();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; __syncthreads();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; __syncthreads();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2]; __syncthreads();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; __syncthreads();
    }
    
    if (tid == 0) buffer[blockIdx.x] = sdata[0];
}

double subtractionNorm(void* data1_d, void* data2_d, size_t size)
{
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    
    subtractionNorm_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)data1_d, (float2*)data2_d, buffer_d, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum1 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum1 += buffer_h[i];
    }
    
    sum1 = sqrt(sum1);

    free(buffer_h);
    cudaFree(buffer_d);

    CHECK_FOR_ERROR("after CompressiveHolo::subtractionNorm");
    return sum1;
}

double CompressiveHolo::estimateLipschitz()
{
    // Check allocated sizes
    size_t min_size = volume_size*sizeof(float2);
    bool failure = false;
    if ((this->data.getDataSize() < min_size) || !this->data.isAllocated()) failure = true;
    if ((data_nm1.getDataSize() < min_size) || !data_nm1.isAllocated()) failure = true;
    if ((data_nm2.getDataSize() < min_size) || !data_nm2.isAllocated()) failure = true;
    if ((buffer.getDataSize() < min_size) || !buffer.isAllocated()) failure = true;
    if (failure)
    {
        std::cout << "CompressiveHolo::estimateLipschitz:: Error: Data not allocated" << std::endl;
        std::cout << "Min size is " << min_size << std::endl;
        std::cout << "x1_d size = " << this->data.getDataSize()
            << ", allocation "      << this->data.isAllocated() << std::endl;
        std::cout << "x2_d size = " << data_nm1.getDataSize()
            << ", allocation "      << data_nm1.isAllocated() << std::endl;
        std::cout << "grad1_d size = " << data_nm2.getDataSize()
            << ", allocation "         << data_nm2.isAllocated() << std::endl;
        std::cout << "grad2_d size = " << buffer.getDataSize()
            << ", allocation "         << buffer.isAllocated() << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    void* x1_d = this->data.getCuData();
    void* x2_d = data_nm1.getCuData();
    void* grad1_d = data_nm2.getCuData();
    void* grad2_d = buffer.getCuData();
    this->state = OPTICALFIELD_STATE_RECONSTRUCTED_COMPLEX;
    
    // Initialize x1_d and x2_d using random data
    curandGenerator_t gen;
    CURAND_SAFE_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234));
    double mean = 0;
    double std = 1; // Standard normal distribution
    CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)x1_d, volume_size*2, mean, std));
    CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)x2_d, volume_size*2, mean, std));
    CURAND_SAFE_CALL(curandDestroyGenerator(gen));
    
    // Compute the norm of x2 - x1
    double norm_x = subtractionNorm(x2_d, x1_d, volume_size);
    //printf("norm_x = %f\n", norm_x);
    
    // Reconstruct the residual estimated from each x1_d and x2_d
    Hologram holo_estimate(params);
    Hologram residual(params);
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    void* grad_d = gradient.getData().getCuData();
    CUDA_SAFE_CALL(cudaMemcpy(grad1_d, grad_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice));
    
    CUDA_SAFE_CALL(cudaMemcpy(x1_d, x2_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice));
    this->calcResidual(&residual, &holo_estimate, RM_EST_TRUE);
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    grad_d = gradient.getData().getCuData();
    CUDA_SAFE_CALL(cudaMemcpy(grad2_d, grad_d, volume_size*sizeof(float2), cudaMemcpyDeviceToDevice));
    
    // Compute norm of grad1 - grad2
    double norm_grad = subtractionNorm(grad1_d, grad2_d, volume_size);
    //printf("norm_grad = %f\n", norm_grad);
    
    // Wipe data to preserve future calls
    cudaMemset(x1_d, 0, volume_size*sizeof(float2));
    cudaMemset(x2_d, 0, volume_size*sizeof(float2));
    cudaMemset(grad1_d, 0, volume_size*sizeof(float2));
    cudaMemset(grad2_d, 0, volume_size*sizeof(float2));
    
    // Compute and return final value
    //printf("CompressiveHolo::estimateLipschitz: norm_grad = %f, norm_x = %f\n", norm_grad, norm_x);
    double L = norm_grad / norm_x;
    L = std::max(L, 1e-6);
    CHECK_FOR_ERROR("end CompressiveHolo::estimateLipschitz");
    return L;
}
