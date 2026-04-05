#include "sparse_compressive_holo.h"  // class implemented
#include <chrono>
#include <thread>

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

void check_isnan_mdata(Hologram* mdata)
{
    CuMat data = mdata->getData();
    size_t width = data.getWidth();
    size_t height = data.getHeight();
    float2* mdata_d = (float2*)data.getCuData();
    float2* temp_mdata_h = (float2*)malloc(width*height*sizeof(float2));
    cudaMemcpy(temp_mdata_h, mdata_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool bad_mdata = false;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            size_t idx = i*width + j;
            if (isnan(temp_mdata_h[idx].x) || isnan(temp_mdata_h[idx].y))
            {
                bad_mdata = true;
                printf("Warning: hologram has bad data at [%d, %d]: [%f, %f]\n",
                    i, j, temp_mdata_h[idx].x, temp_mdata_h[idx].y);
            }
        }
    }
    //printf("Is mdata bad: %d\n", bad_mdata);
    free(temp_mdata_h);
}

//============================= LIFECYCLE ====================================

__global__ void 
sparse_rs_exponent_kernel(double* out_d, int Nx, int Ny, 
               double lambda, double reso)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    // Calculate f2
    // For non-shifted FFT
    // Corrected off-by-one error if rs_mult_kernel that matched matlab
    double fx = (double)(((x + Nx / 2) % Nx)   - ((Nx / 2) )) / (double)Nx;
    double fy = (double)(((y + Ny / 2) % Ny)   - ((Ny / 2) )) / (double)Ny;
    double f2 = fx*fx + fy*fy;

    double sqrt_input = 1 - f2*(lambda / reso)*(lambda / reso);
    sqrt_input = (sqrt_input < 0) ? 0 : sqrt_input;
    out_d[idx] = -2 * M_PI*sqrt(sqrt_input) / lambda;
}

SparseCompressiveHolo::SparseCompressiveHolo(Hologram holo)
{
    params = holo.getParams();
    measured = holo;
    holo.getMinMax(&min_measured, &max_measured);
    // min_residual = -1;
    // max_residual = 1;
    
    stepsize = 1.0;
    stepsize_shrinkage = 0.5;
    force_stepsize = false;
    
    width = holo.getWidth();
    height = holo.getHeight();
    depth = params.num_planes;
    regularization = params.regularization_param;
    regularization_TV = params.regularization_TV_param;
    num_tv_iterations = params.num_tv_iterations;
    
    plane_subsampling = 1; // 1 means no subsampling (all planes)
    //if (depth >= 1000)
    //{
    //    // TODO: Make this more robust
    //    plane_subsampling = 10;
    //}
    
    x1.initialize(holo, plane_subsampling);
    x0.initialize(holo, plane_subsampling);
    y0.initialize(holo, plane_subsampling);
    y1.initialize(holo, plane_subsampling);
    
    gx.initialize(holo, plane_subsampling);
    gy.initialize(holo, plane_subsampling);
    gz.initialize(holo, plane_subsampling);
    
    depth = x1.getNumPlanes();
    
    fft_plan_created = false;
    if (!fft_plan_created)
    {
        cufftPlan2d(&fft_plan, height, width, CUFFT_C2C);
        fft_plan_created = true;
    }
    
    gradient.initialize(holo, &fft_plan);
    
    // holo.copyTo(&bg_holo);
    // bg_holo.setData(holo.getBgData());
    // bg_rec.initialize(bg_holo, &fft_plan);
    
    plane.allocateCuData(width, height, 1, sizeof(float2));
    
    x1_plane.allocateCuData(width, height, 1, sizeof(float2));
    x0_plane.allocateCuData(width, height, 1, sizeof(float2));
    grad_plane.allocateCuData(width, height, 1, sizeof(float2));
    // bg_plane.allocateCuData(width, height, 1, sizeof(float2));
    y0_plane.allocateCuData(width, height, 1, sizeof(float2));
    y1_plane.allocateCuData(width, height, 1, sizeof(float2));
    gx_plane.allocateCuData(width, height, 1, sizeof(float2));
    gy_plane.allocateCuData(width, height, 1, sizeof(float2));
    gz_plane.allocateCuData(width, height, 1, sizeof(float2));
    
    cudaMalloc((void**)&buffer_plane_d, width*height*sizeof(float2));
    
    // Pre-calculate some of the rs exponent data to speed up computation
    exponent_data.allocateCuData(width, height, 1, sizeof(double));
    double* exp_d = (double*)exponent_data.getCuData();
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    sparse_rs_exponent_kernel<<<grid_dim, block_dim>>>
        (exp_d, width, height, params.wavelength, params.resolution);
    
    iteration = 0;
    
    use_fista = true;
    
    CHECK_FOR_ERROR("SparseCompressiveHolo::SparseCompressiveHolo");
}

void SparseCompressiveHolo::destroy()
{
    x1.destroy();
    x0.destroy();
    y0.destroy();
    y1.destroy();
    cudaFree(buffer_plane_d);
    exponent_data.destroy();
    return;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

double computeStepsize(int it, int num_its, int nz, double stepsize)
{
    double initial_step = 1.0 / (double)nz;
    double final_step = 0.2;
    //double increment_log = log10(final_step / initial_step) / (num_its - 1);
    //double step = initial_step + pow(10, it*increment_log);
    
    double increment = (final_step - initial_step) / (num_its-1);
    // double step = initial_step + it*increment;
    double step = stepsize + increment;
    
    if (it == 0) return initial_step;
    if (it == num_its-1) return final_step;
    return initial_step;
    //return step;
}

/**
 * This is the same algorithm as CompressiveHolo::invreseReconstruct_FASTA
 * The only (intended) difference is that the data is stored using a sparse
 * representation to allow larger volumes to be processed
 */
void SparseCompressiveHolo::inverseReconstruct(Hologram holo, CompressiveHoloMode mode)
{
    printf("begin SparseCompressiveHolo::inverseReconstruct\n");
    DECLARE_TIMING(sparse_inverseReconstruct_total);
    START_TIMING(sparse_inverseReconstruct_total);
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
        break;
    default:
    {
        std::cout << "SparseCompressiveHolo::inverseReconstruct: Unsupported mode: " << mode << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    // No need to waste processing time if TV regularization isn't used
    if (params.regularization_TV_param == 0.0)
    {
        mode = COMPRESSIVE_MODE_FASTA_L1;
    }
    
    char metrics_filename[FILENAME_MAX];
    sprintf(metrics_filename, "%s/all_metrics.csv", params.output_path);
    FILE* metricsfid = fopen(metrics_filename, "a");
    if (metricsfid == NULL)
    {
        std::cout << "Unable to open file: " << metrics_filename << std::endl;
        throw(HOLO_ERROR_INVALID_FILE);
    }
    fprintf(metricsfid, "it, f, sparsity, stepsize, total_backtracks\n");
    
    printf("Begin inverseReconstruct\n");
    /*
    for (int s = 0; s < 10; ++s)
    {
        printf("\ntestAdjoint seed = %d:\n", s);
        bool result = this->testAdjoint(s);
        if (result)
            printf("result = true!\n");
        else
            printf("result = false\n");
    }
    bool is_valid_adjoint = this->testAdjoint();
    if (!is_valid_adjoint)
    {
        std::cout << "Reconstruction functions do not satisfy adjoint properties! Check code!" << std::endl;
        //throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    std::cout << "Adjoint check passed" << std::endl;
    */
    double L = -1;
    if (!force_stepsize)
    {
        printf("begin estimateLipschitz\n");
        DECLARE_TIMING(estimateLipschitz);
        START_TIMING(estimateLipschitz);
        
        // for (int s = 0; s < 10; ++s)
        // {
        //     L = this->estimateLipschitz(s);
        //     stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
        //     printf("Seed %d: Lipschitz L = %f, stepsize = %f\n", s, L, stepsize);
        // }
        
        L = this->estimateLipschitz();
        //stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
        stepsize = 1 / L; // TODO: remove magic numbers
        printf("Lipschitz L = %f, stepsize = %f\n", L, stepsize);
        printf("  num_planes = %d, 1/num_planes = %f\n", params.num_planes, 1.0/(double)params.num_planes);
        SAVE_TIMING(estimateLipschitz);
    }
    stepsize = 1.0/(double)params.num_planes;
    // stepsize = stepsize * 0.9;
    float original_stepsize = stepsize;
    
    printf("stepsize = %f, 1/step = %f\n", stepsize, 1/stepsize);
    
    // Initialize volume
    //printf("begin initializations\n");
    //x1.initializeZeros();
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    // residual and estimated_holo are both outputs
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    //printf("calcObjectiveFunction\n");
    double f = this->calcObjectiveFunction(residual, mode);
    double f_resid = this->calcResidualObjective(residual);
    double prev_f_resid = f_resid;
    objective_values.push_back(f);
    //printf("Done with calcObjective\n");
    std::cout << "Initial f = " << f << std::endl;
    
    this->calcInitialObjectiveFunction(mode);
    
    // Reconstruct residual to get the gradient
    //printf("begin reconstruct gradient\n");
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    //printf("done\n");
    // Hologram bg_mean_holo(params);
    // bg_holo.copyTo(&bg_mean_holo);
    // bg_mean_holo.setToMean();
    // bg_rec.reconstruct(bg_holo, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    //std::cout << "Number of elements = " << x1.countNonZeros() << std::endl;
    //std::cout << "Sparsity = " << x1.getSparsity() << std::endl;
    CHECK_FOR_ERROR("sparse inverse before iterations");
    //printf("Will use %d inverse iterations\n", params.num_inverse_iterations);
    int its_since_backtrack = 0;
    int total_backtracks = 0;
    for (int it = 0; it < params.num_inverse_iterations; ++it)
    {
        iteration = it;
        /*if (it == 10)
        {
            double old_reg = regularization;
            regularization = regularization * 1.2;
            printf("Changed L1 regularization from %f to %f\n",
                old_reg, regularization);
        }*/
        // if (it >= params.num_inverse_iterations/2)
        // {
        //     double old_reg = regularization;
        //     regularization = 0.05;
        //     printf("WARNING: changed L1 regularization from %f to %f\n",
        //         old_reg, regularization);
        // }
        DECLARE_TIMING(sparse_inverseReconstruct_iteration);
        START_TIMING(sparse_inverseReconstruct_iteration);
        //std::cout << "FASTA iteration " << it << " of " << params.num_inverse_iterations << ": ";
        printf("FASTA iteration %d of %d\n", it+1, params.num_inverse_iterations);
        //std::cout << "Number of elements = " << x1.countNonZeros() << std::endl;
        //std::cout << "Sparsity = " << x1.getSparsity() << std::endl;
        
        /*if (its_since_backtrack > 10 && stepsize != original_stepsize)
        {
            double new_stepsize = stepsize / stepsize_shrinkage;
            // double new_stepsize = original_stepsize;
            printf("Resetting stepsize from %f to %f\n",
                stepsize, new_stepsize);
            stepsize = new_stepsize;
        }*/
        if (it+1 >= (params.num_inverse_iterations * 8/10))
        {
            if (plane_subsampling != 1)
            {
                printf("Changing subsampling from %d to %d\n",
                    plane_subsampling, 1);
                plane_subsampling = 1;
                
                printf("  before sparsities: x1=%f, x0=%f, y0=%f, y1=%f\n",
                    x1.getSparsity(), x0.getSparsity(),
                    y0.getSparsity(), y1.getSparsity());
                
                x1.setPlaneSubsampling(plane_subsampling);
                x0.setPlaneSubsampling(plane_subsampling);
                y0.setPlaneSubsampling(plane_subsampling);
                y1.setPlaneSubsampling(plane_subsampling);
                
                gx.setPlaneSubsampling(plane_subsampling);
                gy.setPlaneSubsampling(plane_subsampling);
                gz.setPlaneSubsampling(plane_subsampling);
                
                this->enforceSparsity();
                
                printf("Resetting stepsize from %f to %f\n",
                    stepsize, original_stepsize);
                stepsize = original_stepsize;
                //if (L != -1)
                //    stepsize = (2.0 / L) / 10.0;
                
                printf("  after sparsities: x1=%f, x0=%f, y0=%f, y1=%f\n",
                    x1.getSparsity(), x0.getSparsity(),
                    y0.getSparsity(), y1.getSparsity());
            }
        }
        
        //stepsize = computeStepsize(it, params.num_inverse_iterations, params.num_planes, stepsize);
        
        x0 = x1;
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(mode); // x1 = prox(x0, grad, step, reg)
        
        char cmb_iter_filename[FILENAME_MAX];
        /*if (params.output_steps)
        {
            sprintf(cmb_iter_filename, "cmb_iter%03d_backtracks%02d", it, 0);
            this->saveProjections(cmb_iter_filename, MAX_CMB, true);
            
            sprintf(cmb_iter_filename, "%s/estimate_iter%03d.tif", params.output_path, it);
            estimated_holo.write(cmb_iter_filename);
            sprintf(cmb_iter_filename, "%s/residual_iter%03d.tif", params.output_path, it);
            residual.write(cmb_iter_filename);
        }*/
        
        // Non-monotone backtracking line search (enforce monotonicity)
        int backtrack_count = 0;
        this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
        f = this->calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1); // TODO: remove magic number
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
        double ratio = f / (M + lim);
        //printf("  old backtrack ratio = %f, f = %f, M = %f, lim = %f\n", ratio, f, M, lim);
        // if (it == 0)
        // {
        //     printf("WARNING: First it force backtrack ratio to be small\n");
        //     ratio = 0.01;
        // }
        
        // New backtracking method
        f_resid = this->calcResidualObjective(residual);
        ratio = f_resid / (prev_f_resid + lim);
        //printf("  new backtrack ratio = %f, f = %f, M = %f, lim = %f\n",
        //    f_resid/(prev_f_resid+lim), f_resid, prev_f_resid, lim);
        /*if (ratio > 1.0)
        {
            printf("WARNING: Ignored backtracking value = %f\n", ratio);
            ratio = 0.1;
        }*/

        // if (it > 2 && (ratio > 1.0) || (ratio < 0))
        // {
        //     printf("\n\nWarning: Ignoring backtracking entirely!!!\n\n\n");
        //     ratio = 0.1;
        // }
        // if (stepsize*stepsize_shrinkage < 1e-4)
        // {
        //     printf("Ignoring backtracking because of small stepsize\n");
        //     ratio = 0.1;
        // }
        printf("  BT test: f1 = %f, M = %f, lim = %f, ratio = %f\n", f, prev_f_resid, lim, ratio);
        while (((ratio > 1.0) || (ratio < 0)) && (backtrack_count < 2))
        {
            printf("  ratio = %f, backtracking\n", ratio);
            printf("    f_resid = %f, prev_f = %f, lim = %f\n", f_resid, prev_f_resid, lim);
            stepsize *= stepsize_shrinkage;
            //this->denoise(x1, x0, gradient, mode);
            this->denoise(mode);
            
            /*if (params.output_steps)
            {
                sprintf(cmb_iter_filename, "cmb_iter%03d_backtracks%02d", it, backtrack_count+1);
                this->saveProjections(cmb_iter_filename, MAX_CMB, true);
            }*/
            
            this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
            f = this->calcObjectiveFunction(residual, mode);
            lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
            ratio = f / (M+lim);
            f_resid = this->calcResidualObjective(residual);
            ratio = f_resid / (prev_f_resid + lim);
            backtrack_count++;
            total_backtracks++;
            its_since_backtrack = 0;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e, sparsity = %f\n",
                backtrack_count, f, stepsize, x1.getSparsity());
            //printf("    backtrack ratio = %f, f = %f, M = %f, lim = %f\n", ratio, f, M, lim);
            printf("    after ratio = %f, f_resid = %f, prev_f = %f, lim = %f\n", ratio, f_resid, prev_f_resid, lim);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            return;
        }
        if ((ratio > 1.0) || (ratio < 0)) 
            printf("Warning: Ignoring backtrack condition because count too high\n");
        its_since_backtrack++;
        //printf("Done with backtracking\n");
        //printf("objective function = %f\n", f);
        
        // Skip implementing stopping criteria check
        
        // Begin FISTA acceleration steps
        double f_ista = f;
        if (use_fista)
        {
            // Update acceleration parameters
            //printf("Update acceleration parameters\n");
            if (this->fistaRestart(&x0, &x1, &y0))
            {
                std::cout << "restarted FISTA parameter" << std::endl << "  ";
                alpha0 = 1;
            }
            alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;
            
            // Update x1
            //printf("Update x1\n");
            y1 = x1;
            double step = (alpha0 - 1) / alpha1;
            //printf("before fistaUpdate\n");
            this->fistaUpdate(&x1, &y0, &y1, step);
            //printf("after fistaUpdate\n");
            y0 = y1;
            
            // Update d1
            //printf("Update d1\n");
            //printf("current_estimate: width = %d, height = %d\n", current_estimate.getWidth(), current_estimate.getHeight());
            //printf("previous_estimate: width = %d, height = %d\n", previous_estimate.getWidth(), previous_estimate.getHeight());
            //printf("estimated_holo: width = %d, height = %d\n", estimated_holo.getWidth(), estimated_holo.getHeight());
            DECLARE_TIMING(FISTA_UPDATE_COPYTO);
            START_TIMING(FISTA_UPDATE_COPYTO);
            current_estimate.copyTo(&previous_estimate);
            estimated_holo.copyTo(&current_estimate);
            //printf("after update previous and current estimates\n");
            this->fistaUpdate(&estimated_holo, &previous_estimate, &current_estimate, step);
            current_estimate.copyTo(&previous_estimate);
            SAVE_TIMING(FISTA_UPDATE_COPYTO);
        }
        
        // Compute new gradient and cost function
        //printf("Calculate new gradient and cost function\n");
        
        /*if (it == params.num_inverse_iterations-1)
        {
            // double resid_minmax_val = 0.15;
            // double est_minmax_val = 0.30;
            double resid_minmax_val = 0.25;
            double est_minmax_val = 0.25;
            printf("\n\nForcing sparsity\n");
            this->saveProjections("cmb_before_sparsity", MAX_CMB, true);
            //this->savePlanes("before_sparsity");
            this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
            char temp_filename[FILENAME_MAX];
            sprintf(temp_filename, "%s/residual_before_sparsity.tif", params.output_path);
            printf("writing residual before sparsity\n");
            residual.write(temp_filename, -resid_minmax_val, resid_minmax_val);
            sprintf(temp_filename, "%s/estimated_before_sparsity.tif", params.output_path);
            printf("writing estimated before sparsity\n");
            estimated_holo.write(temp_filename, -est_minmax_val, est_minmax_val);
            
            double old_reg = regularization;
            regularization = 0.05;
            printf("WARNING: changed L1 regularization from %f to %f\n",
                old_reg, regularization);
            this->enforceSparsity();
            this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
            f_resid = this->calcResidualObjective(residual);
            ratio = f_resid / (prev_f_resid + lim);
            printf("  new backtrack ratio = %f, f = %f, M = %f, lim = %f\n",
                f_resid/(prev_f_resid+lim), f_resid, prev_f_resid, lim);
            this->saveProjections("cmb_afer_sparsity", MAX_CMB, true);
            //this->savePlanes("after_sparsity");
            sprintf(temp_filename, "%s/residual_after_sparsity.tif", params.output_path);
            printf("writing residual after sparsity\n");
            residual.write(temp_filename, -resid_minmax_val, resid_minmax_val);
            sprintf(temp_filename, "%s/estimated_after_sparsity.tif", params.output_path);
            printf("writing estimated after sparsity\n");
            estimated_holo.write(temp_filename, -est_minmax_val, est_minmax_val);
            printf("\n\n");
        }
        else*/
        {
        this->calcResidualFrom(&residual, &estimated_holo, RM_EST_TRUE);
        }
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        f = this->calcObjectiveFunction(residual, mode);
        prev_f_resid = this->calcResidualObjective(residual);
        
        double f_fista = f;
        // if (f_fista > f_ista)
        // {
        //     use_fista = false;
        //     std::cout << "Turned FISTA off, now using ISTA" << std::endl;
        // }
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << total_backtracks;//backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << ", sparsity " << x1.getSparsity();
        std::cout << std::endl;
        
        //char plane_count_filename[FILENAME_MAX];
        //sprintf(plane_count_filename, "plane_counts_%03d", it);
        //x1.savePlaneNonZeros(plane_count_filename);
        
        if (params.output_steps)
        {
            printf("begin saving step outputs\n");
            sprintf(cmb_iter_filename, "cmb_iter%03d_end", it);
            this->saveProjections(cmb_iter_filename, MAX_CMB, true);
            printf("end saveProjections\n");
            
            // if (max_residual == 1.0)
            // {
            //     residual.getMinMax(&min_residual, &max_residual);
            // }
            // Hologram est_enh(params);
            // estimated_holo.copyTo(&est_enh);
            // CuMat est_enh_data = est_enh.getData();
            // CuMat est_data = estimated_holo.getData();
            // CuMat bg_data = bg_holo.getData();
            // subtract(est_data, bg_data, est_enh_data);
            // est_enh.setData(est_enh_data);
            sprintf(cmb_iter_filename, "%s/estimate_iter%03d.tif", params.output_path, it);
            estimated_holo.write(cmb_iter_filename, min_measured, max_measured);
            // est_enh.write(cmb_iter_filename, -0.25, 0.25);
            sprintf(cmb_iter_filename, "%s/residual_iter%03d.tif", params.output_path, it);
            residual.write(cmb_iter_filename, min_measured, max_measured);
            // est_enh.destroy();
            
            fprintf(metricsfid, "%d, %f, %f, %f, %d\n", it, f, x1.getSparsity(), stepsize, total_backtracks);
            printf("end saving step outputs\n");
        }
        
        STOP_TIMING(sparse_inverseReconstruct_iteration);
        PRINT_TIMING(sparse_inverseReconstruct_iteration);
        SAVE_TIMING(sparse_inverseReconstruct_iteration);
        
        //printf("end of iteration\n");
        CHECK_FOR_ERROR("end sparse FASTA iteration");
        //CHECK_MEMORY("end sparse FASTA iteration");
    }
    
    char resid_filename[FILENAME_MAX];
    // if (max_residual == 1.0)
    // {
    //     residual.getMinMax(&min_residual, &max_residual);
    // }
    // Hologram est_enh(params);
    // estimated_holo.copyTo(&est_enh);
    // CuMat est_enh_data = est_enh.getData();
    // CuMat est_data = estimated_holo.getData();
    // CuMat bg_data = bg_holo.getData();
    // subtract(est_data, bg_data, est_enh_data);
    // est_enh.setData(est_enh_data);
    sprintf(resid_filename, "%s/estimate_final.tif", params.output_path);
    estimated_holo.write(resid_filename, min_measured, max_measured);
    sprintf(resid_filename, "%s/residual_final.tif", params.output_path);
    residual.write(resid_filename, min_measured, max_measured);
    // est_enh.destroy();
    
    fclose(metricsfid);
    
    SAVE_TIMING(sparse_inverseReconstruct_total);
    
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
    return;
}

/**
 * This is the same algorithm as CompressiveHolo::invreseReconstruct_FASTA
 * The only (intended) difference is that the data is stored using a sparse
 * representation to allow larger volumes to be processed
 */
/*
void SparseCompressiveHolo::inverseReconstruct(Hologram holo, CompressiveHoloMode mode)
{
    printf("begin SparseCompressiveHolo::inverseReconstruct\n");
    DECLARE_TIMING(sparse_inverseReconstruct_total);
    START_TIMING(sparse_inverseReconstruct_total);
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
        break;
    default:
    {
        std::cout << "SparseCompressiveHolo::inverseReconstruct: Unsupported mode: " << mode << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    // No need to waste processing time if TV regularization isn't used
    if (params.regularization_TV_param == 0.0)
    {
        mode = COMPRESSIVE_MODE_FASTA_L1;
    }
    
    printf("Begin test inverseReconstruct\n");
    
    double L = -1;
    if (!force_stepsize)
    {
        L = this->estimateLipschitz();
        stepsize = 1 / L; // TODO: remove magic numbers
        printf("Lipschitz L = %f, stepsize = %f\n", L, stepsize);
        printf("  num_planes = %d, 1/num_planes = %f\n", params.num_planes, 1.0/(double)params.num_planes);
    }
    float original_stepsize = stepsize;
    
    printf("stepsize = %f, 1/step = %f\n", stepsize, 1/stepsize);
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    // residual and estimated_holo are both outputs
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    double f_resid = this->calcResidualObjective(residual);
    double prev_f_resid = f_resid;
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    this->calcInitialObjectiveFunction(mode);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    int its_since_backtrack = 0;
    int total_backtracks = 0;
    for (int it = 0; it < params.num_inverse_iterations; ++it)
    {
        iteration = it;
        printf("FASTA iteration %d of %d\n", it+1, params.num_inverse_iterations);
        
        x0 = x1;
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(mode); // x1 = prox(x0, grad, step, reg)
        
        char cmb_iter_filename[FILENAME_MAX];
        
        // Non-monotone backtracking line search (enforce monotonicity)
        int backtrack_count = 0;
        this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
        f = this->calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1); // TODO: remove magic number
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
        double ratio = f / (M + lim);
        
        //if (it == 0)
        {
            // Golden-section search to find stepsize to minimize residual
            // Following https://en.wikipedia.org/wiki/Golden-section_search
            double invphi = (sqrt(5.0) - 1.0) / 2.0;
            double invphi2 = (3.0 - sqrt(5.0)) / 2.0;
            double a = 0.0; // Initial lower bound
            double b = 2/L; // Initial upper bound
            double h = b - a;
            double c = a + invphi2 * h;
            double d = a + invphi * h;
            double yc = 0;
            {
                stepsize = c;
                this->denoise(mode);
                this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
                f = this->calcObjectiveFunction(residual, mode);
                // f_resid = this->calcResidualObjective(residual);
                yc = f;//_resid;
                printf("test stepsize = %f, resid = %f\n", stepsize, f_resid);
            }
            double yd = 0;
            {
                stepsize = d;
                this->denoise(mode);
                this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
                f = this->calcObjectiveFunction(residual, mode);
                // f_resid = this->calcResidualObjective(residual);
                yd = f;//_resid;
                printf("test stepsize = %f, resid = %f\n", stepsize, f_resid);
            }
            while (true)
            {
                int modvalue = 0;
                if (yc < yd)
                {
                    b = d;
                    d = c;
                    yd = yc;
                    h = invphi*h;
                    c = a + invphi2 * h;
                    stepsize = c;
                    modvalue = 1;
                }
                else
                {
                    a = c;
                    c = d;
                    yc = yd;
                    h = invphi*h;
                    d = a + invphi * h;
                    stepsize = d;
                    modvalue = 2;
                }
                this->denoise(mode);
                this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
                f = this->calcObjectiveFunction(residual, mode);
                // f_resid = this->calcResidualObjective(residual);
                printf("test stepsize = %f, resid = %f\n", stepsize, f_resid);
                if (modvalue == 1)
                {
                    yc = f;//_resid;
                }
                else
                {
                    yd = f;//_resid;
                }
                if (h < 0.000001)
                    break;
            }
        }
        
        // Skip implementing stopping criteria check
        
        // Begin FISTA acceleration steps
        double f_ista = f;
        use_fista = false;
        if (use_fista)
        {
            // Update acceleration parameters
            if (this->fistaRestart(&x0, &x1, &y0))
            {
                std::cout << "restarted FISTA parameter" << std::endl << "  ";
                alpha0 = 1;
            }
            alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;
            
            // Update x1
            y1 = x1;
            double step = (alpha0 - 1) / alpha1;
            this->fistaUpdate(&x1, &y0, &y1, step);
            y0 = y1;
            
            // Update d1
            current_estimate.copyTo(&previous_estimate);
            estimated_holo.copyTo(&current_estimate);
            this->fistaUpdate(&estimated_holo, &previous_estimate, &current_estimate, step);
            current_estimate.copyTo(&previous_estimate);
        }
        
        this->calcResidualFrom(&residual, &estimated_holo, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        f = this->calcObjectiveFunction(residual, mode);
        prev_f_resid = this->calcResidualObjective(residual);
        
        double f_fista = f;
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << total_backtracks;//backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << ", sparsity " << x1.getSparsity();
        std::cout << std::endl;
        
        if (params.output_steps)
        {
            sprintf(cmb_iter_filename, "cmb_iter%03d_end", it);
            this->saveProjections(cmb_iter_filename, MAX_CMB, true);
            
            sprintf(cmb_iter_filename, "%s/estimate_iter%03d.tif", params.output_path, it);
            estimated_holo.write(cmb_iter_filename, min_measured, max_measured);
            sprintf(cmb_iter_filename, "%s/residual_iter%03d.tif", params.output_path, it);
            residual.write(cmb_iter_filename, min_measured, max_measured);
        }
        
        CHECK_FOR_ERROR("end sparse FASTA iteration");
    }
    
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
    return;
}
*/

void SparseCompressiveHolo::calcResidual(Hologram* residual)
{
    Hologram estimate(params);
    calcResidual(residual, &estimate, RM_TRUE_EST);
    estimate.destroy();
}

void SparseCompressiveHolo::saveSparse(char* prefix)
{
    x1.saveData(prefix);
    
    return;
}



void SparseCompressiveHolo::saveProjections(char* prefix, CmbMethod method, bool prefix_as_suffix)
{
    x1.saveProjections(prefix, method, prefix_as_suffix);
    
    // int num_subsample_planes = (depth / plane_subsampling);// + 1;
    
    // // Initialize the cmb images
    // cv::Mat xycmb = cv::Mat::zeros(height, width, CV_32F);
    // cv::Mat xzcmb = cv::Mat::zeros(depth,  width, CV_32F);
    // cv::Mat yzcmb = cv::Mat::zeros(height, depth, CV_32F);
    // cv::Mat sub_xzcmb = cv::Mat::zeros(num_subsample_planes, width, CV_32F);
    // cv::Mat sub_yzcmb = cv::Mat::zeros(height, num_subsample_planes, CV_32F);
    
    // cv::Mat xzvec = cv::Mat::zeros(1, width, CV_32F);
    // cv::Mat yzvec = cv::Mat::zeros(height, 1, CV_32F);
    
    // CuMat single_plane;
    // single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    // size_t plane_size = width * height;

    // cv::Mat plane;
    // cv::Mat absorption;
    
    // // Compute global min and max for normalization
    // // TODO: This should be a separate method
    // double global_min = 0;
    // double global_max = 0;
    // for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    // {
    //     x1.getPlane(&single_plane, zidx);
    //     bg_rec.getPlane(&bg_plane, zidx);
    //     plane = single_plane.getMatData();
    //     cv::Mat bg_data = bg_plane.getMatData();
    //     bg_data = bg_data / params.num_planes;
        
    //     cv::Mat plane_complex[2];
    //     cv::Mat bg_complex[2];
    //     cv::split(plane, plane_complex);
    //     cv::split(bg_data, bg_complex);
    //     plane_complex[0] = plane_complex[0] + bg_complex[0];
    //     plane_complex[1] = plane_complex[1] + bg_complex[1];
    //     cv::multiply(plane_complex[0], plane_complex[0], plane_complex[0]);
    //     cv::multiply(plane_complex[1], plane_complex[1], plane_complex[1]);
    //     cv::Mat amplitude = plane_complex[0] + plane_complex[1];
    //     cv::multiply(bg_complex[0], bg_complex[0], bg_complex[0]);
    //     cv::multiply(bg_complex[1], bg_complex[1], bg_complex[1]);
    //     cv::Mat bg_amplitude = bg_complex[0] + bg_complex[1];
    //     absorption = -(amplitude - bg_amplitude);
        
    //     double single_min = 0;
    //     double single_max = 0;
    //     minMaxLoc(absorption, &single_min, &single_max);
        
    //     if (zidx == 0) {global_min = single_min; global_max = single_max;}
    //     global_min = (single_min < global_min)? single_min : global_min;
    //     global_max = (single_max > global_max)? single_max : global_max;
    //     x1.unGetPlane(&single_plane, zidx);
    // }
    
    // printf("saveProjections absorption global min = %f, max = %f\n", global_min, global_max);
    // global_min = 0.0;
    
    // // Compute the combined images
    // // Start at zidx = 1 to exclude noisy first plane
    // int sub_idx = 0;
    // for (int zidx = 0; zidx < depth; zidx+=plane_subsampling)
    // //for (int zidx = 1; zidx < params.num_planes; ++zidx)
    // {
    //     x1.getPlane(&single_plane, zidx);
    //     bg_rec.getPlane(&bg_plane, zidx);
    //     cv::Mat x1_plane = single_plane.getMatData();
    //     cv::Mat bg_data = bg_plane.getMatData();
    //     bg_data = bg_data / params.num_planes;
        
    //     cv::Mat plane_complex[2];
    //     cv::Mat bg_complex[2];
    //     cv::split(x1_plane, plane_complex);
    //     cv::split(bg_data, bg_complex);
    //     plane_complex[0] = plane_complex[0] + bg_complex[0];
    //     plane_complex[1] = plane_complex[1] + bg_complex[1];
    //     cv::multiply(plane_complex[0], plane_complex[0], plane_complex[0]);
    //     cv::multiply(plane_complex[1], plane_complex[1], plane_complex[1]);
    //     cv::Mat amplitude = plane_complex[0] + plane_complex[1];
    //     cv::multiply(bg_complex[0], bg_complex[0], bg_complex[0]);
    //     cv::multiply(bg_complex[1], bg_complex[1], bg_complex[1]);
    //     cv::Mat bg_amplitude = bg_complex[0] + bg_complex[1];
    //     absorption = -(amplitude - bg_amplitude);
    //     plane = absorption;
        
    //     plane = (plane - global_min) / (global_max - global_min);
        
    //     cv::reduce(plane, xzvec, 0, cv::REDUCE_MAX);
    //     cv::reduce(plane, yzvec, 1, cv::REDUCE_MAX);
    //     if (zidx > 0) cv::max(plane, xycmb, xycmb);
        
    //     if (plane_subsampling > 1)
    //     {
    //         xzvec.copyTo(sub_xzcmb(cv::Rect(0, sub_idx, width, 1)));
    //         yzvec.copyTo(sub_yzcmb(cv::Rect(sub_idx, 0, 1, height)));
    //     }
    //     else
    //     {
    //         xzvec.copyTo(xzcmb(cv::Rect(0, zidx, width, 1)));
    //         yzvec.copyTo(yzcmb(cv::Rect(zidx, 0, 1, height)));
    //     }
        
    //     x1.unGetPlane(&single_plane, zidx);
    //     sub_idx++;
    // }
    
    // if (plane_subsampling > 1)
    // {
    //     cv::resize(sub_xzcmb, xzcmb, cv::Size(width, depth));
    //     cv::resize(sub_yzcmb, yzcmb, cv::Size(depth, height));
    // }
    
    // xycmb.convertTo(xycmb, CV_8U, 255);
    // xzcmb.convertTo(xzcmb, CV_8U, 255);
    // yzcmb.convertTo(yzcmb, CV_8U, 255);
    // sub_xzcmb.convertTo(sub_xzcmb, CV_8U, 255);
    // sub_yzcmb.convertTo(sub_yzcmb, CV_8U, 255);
    
    // // Save the images
    // char filename[FILENAME_MAX];
    // sprintf(filename, "%s%s_absorption_xy.tif", params.output_path, prefix);
    // if (prefix_as_suffix) sprintf(filename, "%sxy_absorption_%s.tif", params.output_path, prefix);
    // bool result1 = cv::imwrite(filename, xycmb);
    // sprintf(filename, "%s%s_absorption_xz.tif", params.output_path, prefix);
    // if (prefix_as_suffix) sprintf(filename, "%sxz_absorption_%s.tif", params.output_path, prefix);
    // bool result2 = cv::imwrite(filename, xzcmb);
    // sprintf(filename, "%s%s_absorption_yz.tif", params.output_path, prefix);
    // if (prefix_as_suffix) sprintf(filename, "%syz_absorption_%s.tif", params.output_path, prefix);
    // bool result3 = cv::imwrite(filename, yzcmb);
    // if (!result1 || !result2 || !result3)
    // {
    //     std::cerr << "OpticalField::saveProjections: imwrite failed" << std::endl;
    //     throw HOLO_ERROR_UNKNOWN_ERROR;
    // }
    
    // if (plane_subsampling > 1)
    // {
    //     printf("plane_subsampling = %d, saving subsampled images\n", plane_subsampling);
    //     sprintf(filename, "%s%s_xz_sub.tif", params.output_path, prefix);
    //     if (prefix_as_suffix) sprintf(filename, "%sxz_sub_%s.tif", params.output_path, prefix);
    //     cv::imwrite(filename, sub_xzcmb);
    //     sprintf(filename, "%s%s_yz_sub.tif", params.output_path, prefix);
    //     if (prefix_as_suffix) sprintf(filename, "%syz_sub_%s.tif", params.output_path, prefix);
    //     cv::imwrite(filename, sub_yzcmb);
    // }
    
    // single_plane.destroy();
    
    return;
}

//void SparseCompressiveHolo::savePlanes(char* prefix)
void SparseCompressiveHolo::savePlanes(const char* prefix)
{
    CuMat single_plane;
    single_plane.setMatData(cv::Mat::zeros(height, width, CV_32FC2));
    size_t plane_size = width * height;
    
    cv::Mat plane;
    
    // Compute global min and max for normalization
    // TODO: This should be a separate method
    double global_min = 0;
    double global_max = 0;
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        x1.getPlane(&single_plane, zidx);
        plane = single_plane.getMag().getMatData();
        
        double single_min = 0;
        double single_max = 0;
        minMaxLoc(plane, &single_min, &single_max);
        
        if (zidx == 0) {global_min = single_min; global_max = single_max;}
        global_min = (single_min < global_min)? single_min : global_min;
        global_max = (single_max > global_max)? single_max : global_max;
        x1.unGetPlane(&single_plane, zidx);
    }
    
    // Write the plane images
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        x1.getPlane(&single_plane, zidx);
        plane = single_plane.getMag().getMatData();
        plane = (plane - global_min) / (global_max - global_min);
        
        plane.convertTo(plane, CV_8U, 255);
        
        char filename[FILENAME_MAX];
        // Old code (always prepended params.output_path):
        // sprintf(filename, "%s%s_%04d.tif", params.output_path, prefix, zidx);
        
        // New code: Use prefix as full path+prefix, so no prepending output_path
        sprintf(filename, "%s_%04d.tif", prefix, zidx);
        
        bool is_success = cv::imwrite(filename, plane);
        if (!is_success)
        {
            std::cerr << "OpticalField::savePlanes: imwrite failed" << std::endl;
            throw HOLO_ERROR_UNKNOWN_ERROR;
        }
        
        x1.unGetPlane(&single_plane, zidx);
    }
    
    // The following commented-out block for phase min/max remains unchanged
    /*
    // global_min = 0;
    // global_max = 0;
    // for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    // {
    //     x1.getPlane(&single_plane, zidx);
    //     plane = single_plane.getPhase().getMatData();
        
    //     double single_min = 0;
    //     double single_max = 0;
    //     minMaxLoc(plane, &single_min, &single_max);
        
    //     if (zidx == 0) {global_min = single_min; global_max = single_max;}
    //     global_min = (single_min < global_min)? single_min : global_min;
    //     global_max = (single_max > global_max)? single_max : global_max;
    //     x1.unGetPlane(&single_plane, zidx);
    // }
    // printf("Plane phase global min = %f, max = %f\n", global_min, global_max);
    */
    
    // Write the phase images
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        x1.getPlane(&single_plane, zidx);
        plane = single_plane.getPhase().getMatData();
        plane = (plane) / (2*M_PI);
        //plane = (plane - global_min) / (global_max - global_min);
        
        plane.convertTo(plane, CV_8U, 255);
        
        char filename[FILENAME_MAX];
        // Old:
        // sprintf(filename, "%s%s_phase_%04d.tif", params.output_path, prefix, zidx);
        // New:
        sprintf(filename, "%s_phase_%04d.tif", prefix, zidx);
        
        bool is_success = cv::imwrite(filename, plane);
        if (!is_success)
        {
            std::cerr << "OpticalField::savePlanes: imwrite failed" << std::endl;
            throw HOLO_ERROR_UNKNOWN_ERROR;
        }
        
        x1.unGetPlane(&single_plane, zidx);
    }
    
    // Write the real component images
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        x1.getPlane(&single_plane, zidx);
        plane = single_plane.getReal().getMatData();
        double max_val = sqrt(global_max)/10;
        double min_val = -max_val;
        plane = (plane - min_val) / (max_val - min_val);
        
        plane.convertTo(plane, CV_8U, 255);
        
        char filename[FILENAME_MAX];
        // Old:
        // sprintf(filename, "%s%s_real_%04d.tif", params.output_path, prefix, zidx);
        // New:
        sprintf(filename, "%s_real_%04d.tif", prefix, zidx);
        
        bool is_success = cv::imwrite(filename, plane);
        if (!is_success)
        {
            std::cerr << "OpticalField::savePlanes: imwrite failed" << std::endl;
            throw HOLO_ERROR_UNKNOWN_ERROR;
        }
    }
    
    // Write the imaginary component images
    for (int zidx = 0; zidx < params.num_planes; zidx+=plane_subsampling)
    {
        x1.getPlane(&single_plane, zidx);
        plane = single_plane.getImag().getMatData();
        double max_val = sqrt(global_max)/10;
        double min_val = -max_val;
        plane = (plane - min_val) / (max_val - min_val);
        
        plane.convertTo(plane, CV_8U, 255);
        
        char filename[FILENAME_MAX];
        // Old:
        // sprintf(filename, "%s%s_imag_%04d.tif", params.output_path, prefix, zidx);
        // New:
        sprintf(filename, "%s_imag_%04d.tif", prefix, zidx);
        
        bool is_success = cv::imwrite(filename, plane);
        if (!is_success)
        {
            std::cerr << "OpticalField::savePlanes: imwrite failed" << std::endl;
            throw HOLO_ERROR_UNKNOWN_ERROR;
        }
        
        x1.unGetPlane(&single_plane, zidx);
    }
    
    single_plane.destroy();
    
    return;
}
//============================= ACCESS     ===================================

CuMat SparseCompressiveHolo::getPlane(size_t plane_idx)
{
    CuMat plane;
    x1.getPlane(&plane, plane_idx);
    return plane;
}

//============================= INQUIRY    ===================================
//============================= TESTS      ===================================

CuMat SparseCompressiveHolo::test_denoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    stepsize = 0.017536;
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    //std::cout << "FASTA iteration " << it << " of " << params.num_inverse_iterations << ": ";
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    //this->denoise(x1, x0, gradient, mode);
    this->denoise(mode);
    
    x1.getPlane(&x1_plane, 0);
    
    return x1_plane;
}

Hologram SparseCompressiveHolo::test_estimatedHoloOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    stepsize = 0.017536;
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    //std::cout << "FASTA iteration " << it << " of " << params.num_inverse_iterations << ": ";
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    //this->denoise(x1, x0, gradient, mode);
    this->denoise(mode);
    
    // Non-monotone backtracking line search (enforce monotonicity)
    int backtrack_count = 0;
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    
    return estimated_holo;
}

Hologram SparseCompressiveHolo::test_residualOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    stepsize = 0.017536;
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    //std::cout << "FASTA iteration " << it << " of " << params.num_inverse_iterations << ": ";
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    //this->denoise(x1, x0, gradient, mode);
    this->denoise(mode);
    
    // Non-monotone backtracking line search (enforce monotonicity)
    int backtrack_count = 0;
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    
    return residual;
}

double SparseCompressiveHolo::test_objectiveOfDenoised()
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    stepsize = 0.017536;
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    //std::cout << "FASTA iteration " << it << " of " << params.num_inverse_iterations << ": ";
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    //this->denoise(x1, x0, gradient, mode);
    this->denoise(mode);
    
    // Non-monotone backtracking line search (enforce monotonicity)
    int backtrack_count = 0;
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    f = this->calcObjectiveFunction(residual, mode);
    
    return f;
}

void SparseCompressiveHolo::test_fistaUpdate_results(CuMat* x1_p2, CuMat* d1)
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    if (!force_stepsize)
    {
        printf("begin estimateLipschitz\n");
        double L = this->estimateLipschitz();
        //printf("finished estimateLipschitz\n");
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
    }
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    //printf("calcResidual\n");
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    //printf("calcObjectiveFunction\n");
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    //printf("Done with calcObjective\n");
    std::cout << "Initial f = " << f << std::endl;
    
    // Reconstruct residual to get the gradient
    //printf("begin reconstruct gradient\n");
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    //printf("done\n");
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    int it = 0;
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    //printf("  denoise\n");
    //this->denoise(x1, x0, gradient, mode);
    this->denoise(mode);
    
    //printf("After denoise\n");
    //x1.getPlane(&x1_plane, 255);
    //std::cout << "Recovered last x1_plane: " << std::endl;
    //std::cout << x1_plane.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    
    // Non-monotone backtracking line search (enforce monotonicity)
    int backtrack_count = 0;
    //printf("  calcResidual\n");
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    //std::cout << "estimated hologram: " << std::endl;
    //std::cout << estimated_holo.getData().getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    //printf("  calcObjectiveFunction\n");
    f = this->calcObjectiveFunction(residual, mode);
    int look_back = std::min(10, it+1); // TODO: remove magic number
    double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
    //printf("  calcLineSearchLimit\n");
    double lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
    double ratio = f / (M + lim);
    printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
    while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
    {
        printf("  ratio = %f, backtracking\n", ratio);
        stepsize *= stepsize_shrinkage;
        //this->denoise(x1, x0, gradient, mode);
        this->denoise(mode);
        this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
        f = this->calcObjectiveFunction(residual, mode);
        lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
        ratio = f / (M+lim);
        backtrack_count++;
        if (backtrack_count == 1) printf("\n");
        printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
    }
    if (backtrack_count == 20)
    {
        std::cout << "Warning: excessive backtracking detected" << std::endl;
        return;
    }
    printf("Done with backtracking\n");
    printf("objective function = %f\n", f);
    
    // Skip implementing stopping criteria check
    
    // Begin FISTA acceleration steps

    // Update acceleration parameters
    //printf("Update acceleration parameters\n");
    if (this->fistaRestart(&x0, &x1, &y0))
    {
        std::cout << "restarted FISTA parameter" << std::endl << "  ";
        alpha0 = 1;
    }
    alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;
    
    // Update x1
    //printf("Update x1\n");
    y1 = x1;
    double step = (alpha0 - 1) / alpha1;
    //printf("before fistaUpdate\n");
    this->fistaUpdate(&x1, &y0, &y1, step);
    //printf("after fistaUpdate\n");
    y0 = y1;
    
    // Update d1
    //printf("Update d1\n");
    //printf("current_estimate: width = %d, height = %d\n", current_estimate.getWidth(), current_estimate.getHeight());
    //printf("previous_estimate: width = %d, height = %d\n", previous_estimate.getWidth(), previous_estimate.getHeight());
    //printf("estimated_holo: width = %d, height = %d\n", estimated_holo.getWidth(), estimated_holo.getHeight());
    
    //cv::Rect roi(126,180,9,9);
    //std::cout << "ROI: " << roi << std::endl;
    //std::cout << std::endl << "Before swaps" << std::endl;
    //std::cout << "previous_estimate:" << std::endl << previous_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "current_estimate:" << std::endl << current_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "estimated_holo:" << std::endl << estimated_holo.getData().getReal().getMatData()(roi) << std::endl;
    
    //previous_estimate = current_estimate;
    //current_estimate = estimated_holo;
    current_estimate.copyTo(&previous_estimate);
    estimated_holo.copyTo(&current_estimate);
    
    //std::cout << std::endl << "After swaps" << std::endl;
    //std::cout << "previous_estimate:" << std::endl << previous_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "current_estimate:" << std::endl << current_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "estimated_holo:" << std::endl << estimated_holo.getData().getReal().getMatData()(roi) << std::endl;
    
    //printf("after update previous and current estimates\n");
    //printf("fistaUpdate step = %f\n", step);
    this->fistaUpdate(&estimated_holo, &previous_estimate, &current_estimate, step);
    //previous_estimate = current_estimate;
    current_estimate.copyTo(&previous_estimate);
    
    //std::cout << std::endl << "After fistaUpdate" << std::endl;
    //std::cout << "previous_estimate:" << std::endl << previous_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "current_estimate:" << std::endl << current_estimate.getData().getReal().getMatData()(roi) << std::endl;
    //std::cout << "estimated_holo:" << std::endl << estimated_holo.getData().getReal().getMatData()(roi) << std::endl;
    
    x1.getPlane(x1_p2, 2);
    *d1 = estimated_holo.getData();
    
    return;
}

void SparseCompressiveHolo::test_denoise_it1(CuMat* x1_p2)
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    if (!force_stepsize)
    {
        printf("begin estimateLipschitz\n");
        double L = this->estimateLipschitz();
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
    }
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
   
    CHECK_FOR_ERROR("sparse inverse before iterations");
    for (int it = 0; it < 2; ++it)
    {
        printf("FASTA iteration %d of %d\n", it+1, params.num_inverse_iterations);
        
        x0 = x1;
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(mode);
        if (it == 1)
        {
            x1.getPlane(x1_p2, 2);
            return;
        }
        
        // Non-monotone backtracking line search (enforce monotonicity)
        int backtrack_count = 0;
        this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
        f = this->calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1); // TODO: remove magic number
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
        double ratio = f / (M + lim);
        printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
        while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
        {
            printf("  ratio = %f, backtracking\n", ratio);
            stepsize *= stepsize_shrinkage;
            this->denoise(mode);
            this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
            f = this->calcObjectiveFunction(residual, mode);
            lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
            ratio = f / (M+lim);
            backtrack_count++;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            return;
        }

        // Update acceleration parameters
        if (this->fistaRestart(&x0, &x1, &y0))
        {
            std::cout << "restarted FISTA parameter" << std::endl << "  ";
            alpha0 = 1;
        }
        alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;
        
        // Update x1
        y1 = x1;
        double step = (alpha0 - 1) / alpha1;
        this->fistaUpdate(&x1, &y0, &y1, step);
        y0 = y1;
        
        // Update d1
        current_estimate.copyTo(&previous_estimate);
        estimated_holo.copyTo(&current_estimate);
        this->fistaUpdate(&estimated_holo, &previous_estimate, &current_estimate, step);
        current_estimate.copyTo(&previous_estimate);
        
        // Compute new gradient and cost function
        this->calcResidualFrom(&residual, &estimated_holo, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        f = this->calcObjectiveFunction(residual, mode);
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << std::endl;
        
        CHECK_FOR_ERROR("end sparse FASTA iteration");
    }
    
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
    return;
}

CuMat SparseCompressiveHolo::test_FL_prox_it1(int num_tv_its)
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_FUSED_LASSO;
    this->num_tv_iterations = num_tv_its;
    
    if (!force_stepsize)
    {
        printf("begin estimateLipschitz\n");
        DECLARE_TIMING(estimateLipschitz);
        START_TIMING(estimateLipschitz);
        double L = this->estimateLipschitz();
        std::cout << "Estimated Lipschitz constant is " << L << std::endl;
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
        SAVE_TIMING(estimateLipschitz);
    }
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    CHECK_FOR_ERROR("sparse inverse before iterations");
    
    x0 = x1;
    double alpha0 = alpha1;
    
    // Compute proximal (FBS step)
    this->denoise(mode);
    
    x1.getPlane(&x1_plane, 49);
    
    return x1_plane;
}

void SparseCompressiveHolo::test_latest_error(CuMat* x1_p2)
{
    CompressiveHoloMode mode = COMPRESSIVE_MODE_FASTA_L1;
    
    if (!force_stepsize)
    {
        printf("begin estimateLipschitz\n");
        double L = this->estimateLipschitz();
        stepsize = (2.0 / L) / 10.0; // TODO: remove magic numbers
    }
    
    // Initialize FISTA iteration variables
    x0 = x1;
    y0 = x1;
    double alpha1 = 1.0; // TODO: remove magic numbers
    
    Hologram estimated_holo(params); // holo_estimate = d1
    Hologram previous_estimate(params);
    Hologram current_estimate(params);
    
    estimated_holo.setSize(width, height);
    previous_estimate.setSize(width, height);
    current_estimate.setSize(width, height);
    
    // Compute residual and initial objective value
    Hologram residual(params);
    this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
    double f = this->calcObjectiveFunction(residual, mode);
    objective_values.push_back(f);
    std::cout << "Initial f = " << f << std::endl;
    
    // Reconstruct residual to get the gradient
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
   
    CHECK_FOR_ERROR("sparse inverse before iterations");
    for (int it = 0; it < params.num_inverse_iterations; ++it)
    {
        printf("FASTA iteration %d of %d\n", it+1, params.num_inverse_iterations);
        
        x0 = x1;
        double alpha0 = alpha1;
        
        // Compute proximal (FBS step)
        this->denoise(mode);
        
        // Non-monotone backtracking line search (enforce monotonicity)
        int backtrack_count = 0;
        this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
        
        f = this->calcObjectiveFunction(residual, mode);
        int look_back = std::min(10, it+1); // TODO: remove magic number
        double M = *std::max_element(objective_values.end()-look_back, objective_values.end());
        double lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
        double ratio = f / (M + lim);
        printf("backtrack test: f = %f, M = %f, lim = %f\n", f, M, lim);
        while (((ratio > 1.01) || (ratio < 0)) && (backtrack_count < 20))
        {
            printf("  ratio = %f, backtracking\n", ratio);
            stepsize *= stepsize_shrinkage;
            this->denoise(mode);
            this->calcResidual(&residual, &estimated_holo, RM_EST_TRUE);
            f = this->calcObjectiveFunction(residual, mode);
            lim = this->calcLineSearchLimit(&x1, &x0, &gradient);
            ratio = f / (M+lim);
            backtrack_count++;
            if (backtrack_count == 1) printf("\n");
            printf("  backtrack %d: f = %e, step = %e\n", backtrack_count, f, stepsize);
        }
        if (backtrack_count == 20)
        {
            std::cout << "Warning: excessive backtracking detected" << std::endl;
            return;
        }
        printf("  after backtracks f = %f\n", f);

        // Update acceleration parameters
        if (this->fistaRestart(&x0, &x1, &y0))
        {
            std::cout << "restarted FISTA parameter" << std::endl << "  ";
            alpha0 = 1;
        }
        alpha1 = (1.0 +sqrt(1.0 + 4.0*alpha0*alpha0)) / 2.0;
        
        // Update x1
        y1 = x1;
        double step = (alpha0 - 1) / alpha1;
        
        /*if (it == 1)
        {
            y1.getPlane(x1_p2, 2);
            //*x1_p2 = residual.getData();
            return;
        }*/
        
        this->fistaUpdate(&x1, &y0, &y1, step);
        y0 = y1;
        
        if (it == 1)
        {
            x1.getPlane(x1_p2, 2);
            //*x1_p2 = residual.getData();
            printf("fistaUpdate step = %f\n", step);
            return;
        }
        
        // Update d1
        current_estimate.copyTo(&previous_estimate);
        estimated_holo.copyTo(&current_estimate);
        this->fistaUpdate(&estimated_holo, &previous_estimate, &current_estimate, step);
        current_estimate.copyTo(&previous_estimate);
        
        // Compute new gradient and cost function
        this->calcResidualFrom(&residual, &estimated_holo, RM_EST_TRUE);
        gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
        f = this->calcObjectiveFunction(residual, mode);
        
        if (it == 0) objective_values.pop_back();
        objective_values.push_back(f);
        
        printf(" new f = %e", f);
        std::cout << ", backtracks " << backtrack_count;
        std::cout << ", stepsize " << stepsize;
        std::cout << std::endl;
        
        CHECK_FOR_ERROR("end sparse FASTA iteration");
    }
    
    CHECK_FOR_ERROR("end CompressiveHolo::inverseReconstruct::FASTA");
    return;
}

/////////////////////////////// PRIVATE //////////////////////////////////////

__global__ void sparse_phase_scale_kernel
    (float2* out_d, float2* in_d, size_t Nx, size_t Ny, double lambda, double z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    
    double phase = 2*M_PI*z/lambda;
    
    //double p_x = cos(phase);
    //double p_y = sin(phase);
    float p_x = 0.0;
    float p_y = 0.0;
    sincosf(phase, &p_y, &p_x);
    
    float2 old_data = in_d[idx];
    
    out_d[idx].x = old_data.x*p_x - old_data.y*p_y;
    out_d[idx].y = old_data.y*p_x + old_data.x*p_y;
}

__global__ void 
sparse_rs_phase_mult_kernel(float2* plane_d, float2* fft_holo_d, int Nx, int Ny, 
               double* exp_d, double z, double scale = 1)
{
    scale = 1.0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;

    double exponent = z*exp_d[idx];

    // keep local copies of Hx[idx].{x|y}
    //double lHz_x = cos(exponent);
    //double lHz_y = sin(exponent);
    float lHz_x = 0.0;
    float lHz_y = 0.0;
    sincosf(exponent, &lHz_y, &lHz_x);
    
    float in_x = fft_holo_d[idx].x;
    float in_y = fft_holo_d[idx].y;

    // Result is complex multiplication of hologram fft with Hz kernel
    // Pre divide by size to scale correctly after inverse FFT
    plane_d[idx].x = scale * (lHz_x * in_x - lHz_y * in_y) / (Nx * Ny);
    plane_d[idx].y = scale * (lHz_y * in_x + lHz_x * in_y) / (Nx * Ny);
}

__global__ void sparse_add_plane_kernel(float2* out, float2* in, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((x < Nx) && (y < Ny))
    {
        int in_idx = y*Nx + x;
        int out_idx = y*Nx + x;
        
        out[out_idx].x += in[in_idx].x;
        out[out_idx].y += in[in_idx].y;
    }
}

__global__ void sparse_discard_imag_kernel(float2* data, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        // Keep only real component as is
        data[idx].y = 0;
    }
}

__global__ void back_propagate_finish_kernel(float2* data, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y*Nx + x;
    
    if (idx < Nx*Ny)
    {
        // float2 dataval = data[idx];
        // float mag = dataval.x*dataval.x + dataval.y*dataval.y;
        // float out = dataval.x + 0.5*mag;
        
        // data[idx].x = out;
        data[idx].y = 0;
    }
}

void SparseCompressiveHolo::backPropagate(Hologram* estimate, ReconstructionMode mode)
{
    DECLARE_TIMING(SP_INV_BACK_PROPAGATE);
    START_TIMING(SP_INV_BACK_PROPAGATE);
    //printf("SparseCompressiveHolo::backPropagate\n");
    CuMat holo_data = estimate->getData();
    holo_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* holo_d = (float2*)holo_data.getCuData();
    cudaMemset(holo_d, 0, width*height*sizeof(float2));
    double* exp_d = (double*)exponent_data.getCuData();

    double z = params.start_plane;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    double scale = 2 / std::sqrt(2 * x1.getNumPlanes());
    
    /*float2* temp_plane_h = (float2*)malloc(width*height*sizeof(float2));
    bool found_nan = false;
    cudaMemcpy(temp_plane_h, holo_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    found_nan = false;
    for (int i = 0; i < width*height; ++i)
    {
        if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
            found_nan = true;
            //printf("found nan in holo_d at %d\n", i);
    }
    printf("holo_d: found_nan = %d\n", found_nan);*/
    
    int planes_used = 0;
    for (size_t zid = 0; zid < x1.getNumPlanes(); zid+=plane_subsampling)
    {
        planes_used++;
        z = params.plane_list[zid];
        
        if (x1.getPlaneNnz(zid) > 0)
        {
            //if (zid == 0) printf("\n\nPlane %d of %d\n", zid, x1.getNumPlanes());
            x1.getPlane(&plane, zid);
            float2* plane_d = (float2*)plane.getCuData();
            /*cudaMemcpy(temp_plane_h, plane_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
            found_nan = false;
            for (int i = 0; i < width*height; ++i)
            {
                if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
                    found_nan = true;
                    //printf("found nan in plane_d at %d\n", i);
            }
            if (zid == 0) printf("plane_d: found_nan = %d\n", found_nan);
            if (zid == 0) printf("plane.hasNan = %d\n", plane.hasNan());*/
            
            DECLARE_TIMING(BACKPROPAGATE_SCALE_KERNEL);
            START_TIMING(BACKPROPAGATE_SCALE_KERNEL);
            sparse_phase_scale_kernel<<<grid_dim, block_dim>>>
                (buffer_plane_d, plane_d, width, height, params.wavelength, -z);
            SAVE_TIMING(BACKPROPAGATE_SCALE_KERNEL);
            
            DECLARE_TIMING(BACKPROPAGATE_FFT);
            START_TIMING(BACKPROPAGATE_FFT);
            cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
            SAVE_TIMING(BACKPROPAGATE_FFT);
            
            /*cudaMemcpy(temp_plane_h, buffer_plane_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
            found_nan = false;
            for (int i = 0; i < width*height; ++i)
            {
                if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
                    found_nan = true;
            }
            if (zid == 0) printf("after fft, buffer_plane_d: found_nan = %d\n", found_nan);*/
            
            DECLARE_TIMING(BACKPROPAGATE_PHASE_MULT_KERNEL);
            START_TIMING(BACKPROPAGATE_PHASE_MULT_KERNEL);
            sparse_rs_phase_mult_kernel<<<grid_dim, block_dim>>>
                (buffer_plane_d, buffer_plane_d, width, height, 
                exp_d, -z, scale);
            SAVE_TIMING(BACKPROPAGATE_PHASE_MULT_KERNEL);
                
            /*cudaMemcpy(temp_plane_h, buffer_plane_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
            found_nan = false;
            for (int i = 0; i < width*height; ++i)
            {
                if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
                    found_nan = true;
            }
            if (zid == 0) printf("after rs_phase_mult, buffer_plane_d: found_nan = %d\n", found_nan);
            */
            
            DECLARE_TIMING(BACKPROPAGATE_ADD_PLANE_KERNEL);
            START_TIMING(BACKPROPAGATE_ADD_PLANE_KERNEL);
            sparse_add_plane_kernel<<<grid_dim, block_dim>>>(holo_d, buffer_plane_d, width, height);
            SAVE_TIMING(BACKPROPAGATE_ADD_PLANE_KERNEL);
            
            /*cudaMemcpy(temp_plane_h, holo_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
            found_nan = false;
            for (int i = 0; i < width*height; ++i)
            {
                if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
                    found_nan = true;
                    //printf("found nan in holo_d at %d\n", i);
            }
            if (zid == 0) printf("holo_d: found_nan = %d\n", found_nan);*/
            
            x1.unGetPlane(&plane, zid);
        }
        
        //z += params.plane_stepsize;
    }
    
    /*cudaMemcpy(temp_plane_h, holo_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    found_nan = false;
    for (int i = 0; i < width*height; ++i)
    {
        if (isnan(temp_plane_h[i].x) || isnan(temp_plane_h[i].y))
            found_nan = true;
            //printf("found nan in holo_d at %d\n", i);
    }
    printf("\n\nafter planes holo_d: found_nan = %d\n", found_nan);*/
    
    cufftExecC2C(fft_plan, holo_d, holo_d, CUFFT_INVERSE);
    
    // Actual estimate is only real component
    // sparse_discard_imag_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    back_propagate_finish_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    
    holo_data.setCuData((void*)holo_d);
    // Account for other planes
    mult(holo_data, (double)depth/(double)planes_used, holo_data);
    
    // // Add in the background
    // CuMat bg_data = measured.getBgData();
    // sum(holo_data, bg_data, holo_data);
    
    estimate->setData(holo_data);
    
    //std::cout << "SparseCompressiveHolo::backPropagate: estimated hologram:" << std::endl;
    //std::cout << holo_data.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    //free(temp_plane_h);
    
    SAVE_TIMING(SP_INV_BACK_PROPAGATE);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::backPropagate");
    return;
}

void SparseCompressiveHolo::backPropagateInitial(Hologram* estimate, ReconstructionMode mode)
{
    CuMat holo_data = estimate->getData();
    holo_data.allocateCuData(width, height, 1, sizeof(float2));
    float2* holo_d = (float2*)holo_data.getCuData();
    cudaMemset(holo_d, 0, width*height*sizeof(float2));
    double* exp_d = (double*)exponent_data.getCuData();
    
    

    double z = params.start_plane;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    double scale = 1;
    
    double this_stepsize = stepsize;///30;
    printf("backPropagateInital: step = %f, 1/step = %f\n", stepsize, 1/stepsize);
    
    int planes_used = 0;
    for (size_t zid = 0; zid < x1.getNumPlanes(); zid+=plane_subsampling)
    {
        planes_used++;
        z = params.plane_list[zid];
        
        gradient.getPlane(&plane, zid);
        mult(plane, this_stepsize, plane);
        float2* plane_d = (float2*)plane.getCuData();
        
        sparse_phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, plane_d, width, height, params.wavelength, -z);
        
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        
        sparse_rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
            exp_d, -z, scale);
        
        sparse_add_plane_kernel<<<grid_dim, block_dim>>>(holo_d, buffer_plane_d, width, height);
    }
    
    cufftExecC2C(fft_plan, holo_d, holo_d, CUFFT_INVERSE);
    
    // Actual estimate is only real component
    sparse_discard_imag_kernel<<<grid_dim, block_dim>>>(holo_d, width, height);
    
    holo_data.setCuData((void*)holo_d);
    // Account for other planes
    mult(holo_data, (double)depth/(double)planes_used, holo_data);
    estimate->setData(holo_data);
    
    SAVE_TIMING(SP_INV_BACK_PROPAGATE);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::backPropagate");
    return;
}

template <unsigned int blockSize>
__global__ void sparseLipschitzSubtractionNorm_kernel(float2* data1, float2* data2, float* buffer, int size)
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

double sparseLipschitzSubtractionNorm(void* data1_d, void* data2_d, size_t size)
{
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    
    sparseLipschitzSubtractionNorm_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)data1_d, (float2*)data2_d, buffer_d, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum1 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum1 += buffer_h[i];
    }
    
    // IMPORTANT! Returned value is not the true L2-norm until sqrt taken
    //sum1 = sqrt(sum1);

    free(buffer_h);
    cudaFree(buffer_d);

    CHECK_FOR_ERROR("after CompressiveHolo::subtractionNorm");
    return sum1;
}

double SparseCompressiveHolo::estimateLipschitz(int seed)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::estimageLipschitz");
    // Check allocated sizes
    size_t plane_size = width*height;
    size_t min_size = plane_size*sizeof(float2);
    bool failure = false;
    if ((x1_plane.getDataSize() < min_size) || !x1_plane.isAllocated()) failure = true;
    if ((x0_plane.getDataSize() < min_size) || !x0_plane.isAllocated()) failure = true;
    if ((grad_plane.getDataSize() < min_size) || !grad_plane.isAllocated()) failure = true;
    if ((y0_plane.getDataSize() < min_size) || !y0_plane.isAllocated()) failure = true;
    if (failure)
    {
        std::cout << "CompressiveHolo::estimateLipschitz:: Error: Data not allocated" << std::endl;
        std::cout << "Min size is " << min_size << std::endl;
        std::cout << "x1_d size = " << x1_plane.getDataSize()
            << ", allocation "      << x1_plane.isAllocated() << std::endl;
        std::cout << "x2_d size = " << x0_plane.getDataSize()
            << ", allocation "      << x0_plane.isAllocated() << std::endl;
        std::cout << "grad1_d size = " << grad_plane.getDataSize()
            << ", allocation "         << grad_plane.isAllocated() << std::endl;
        std::cout << "grad2_d size = " << y0_plane.getDataSize()
            << ", allocation "         << y0_plane.isAllocated() << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    CHECK_FOR_ERROR("getting CuData from planes");
    float2* x1_d = (float2*)x1_plane.getCuData();
    float2* x2_d = (float2*)x0_plane.getCuData();
    float2* grad1_d = (float2*)grad_plane.getCuData();
    float2* grad2_d = (float2*)y0_plane.getCuData();
    double* exp_d = (double*)exponent_data.getCuData();
    
    // Aliases for convenience
    float2* holo1_d = grad1_d;
    float2* holo2_d = grad2_d;
    
    // Holograms must be initialized to zero before adding to them
    cudaMemset(holo1_d, 0, width*height*sizeof(float2));
    cudaMemset(holo2_d, 0, width*height*sizeof(float2));
    
    // Random number generator initialization
    CHECK_FOR_ERROR("Create random number generator");
    curandGenerator_t gen;
    CURAND_SAFE_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    double mean = 0;
    double std = 1; // Standard normal distribution
    
    double z = params.start_plane;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    double scale = 2 / std::sqrt(2 * params.num_planes);
    
    //float2* temp_h = (float2*)malloc(width*height*sizeof(float2));
    //bool bad_found = false;
    
    // Compute norm of x2 - x1 plane-by-plane
    // Also backpropagate to create the estimated hologram
    double norm_x = 0;
    CHECK_FOR_ERROR("begin x norm plane steps");
    for (size_t zid = 0; zid < x1.getNumPlanes(); ++zid)
    {
        //printf("\nplane %d of %d\n", zid, x1.getNumPlanes());
        //check_isnan_mdata(&measured);
        
        z = params.plane_list[zid];
        
        CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)x1_d, plane_size*2, mean, std));
        cudaDeviceSynchronize();
        CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)x2_d, plane_size*2, mean, std));
        cudaDeviceSynchronize();
        
        /*cudaMemcpy(temp_h, x1_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
        printf("after curandGenerateNormal:\n[");
        for (int i = 0; i < 10; ++i) printf("{%f, %f}, ", temp_h[i].x, temp_h[i].y);
        printf("]\n");
        bad_found = false;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                size_t idx = i*width + j;
                if (isnan(temp_h[idx].x) || isnan(temp_h[idx].y))
                {
                    bad_found = true;
                }
            }
        }
        printf("Found bad value: %d\n", bad_found);*/
        
        // Add the norm for this plane
        norm_x += sparseLipschitzSubtractionNorm(x2_d, x1_d, plane_size);
        
        // Update estimated hologram from x1
        sparse_phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, x1_d, width, height, params.wavelength, -z);
        cudaDeviceSynchronize();
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        
        /*cudaMemcpy(temp_h, buffer_plane_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
        printf("after cufftExecC2C:\n[");
        for (int i = 0; i < 10; ++i) printf("{%f, %f}, ", temp_h[i].x, temp_h[i].y);
        printf("]\n");
        bad_found = false;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                size_t idx = i*width + j;
                if (isnan(temp_h[idx].x) || isnan(temp_h[idx].y))
                {
                    bad_found = true;
                }
            }
        }
        printf("Found bad value: %d\n", bad_found);*/
        
        sparse_rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
            exp_d, -z, scale);
        cudaDeviceSynchronize();
        sparse_add_plane_kernel<<<grid_dim, block_dim>>>(holo1_d, buffer_plane_d, width, height);
        cudaDeviceSynchronize();
        
        /*cudaMemcpy(temp_h, holo1_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
        printf("after sparse_add_plane_kernel:\n[");
        for (int i = 0; i < 10; ++i) printf("{%f, %f}, ", temp_h[i].x, temp_h[i].y);
        printf("]\n");
        bad_found = false;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                size_t idx = i*width + j;
                if (isnan(temp_h[idx].x) || isnan(temp_h[idx].y))
                {
                    bad_found = true;
                }
            }
        }
        printf("Found bad value: %d\n", bad_found);*/
        
        // Update estimated hologram from x2
        sparse_phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, x2_d, width, height, params.wavelength, -z);
        cudaDeviceSynchronize();
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        sparse_rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
            exp_d, -z, scale);
        cudaDeviceSynchronize();
        sparse_add_plane_kernel<<<grid_dim, block_dim>>>(holo2_d, buffer_plane_d, width, height);
        cudaDeviceSynchronize();
        
        /*cudaMemcpy(temp_h, holo2_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
        printf("x2 after sparse_add_plane_kernel:\n[");
        for (int i = 0; i < 10; ++i) printf("{%f, %f}, ", temp_h[i].x, temp_h[i].y);
        printf("]\n");
        bad_found = false;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                size_t idx = i*width + j;
                if (isnan(temp_h[idx].x) || isnan(temp_h[idx].y))
                {
                    bad_found = true;
                }
            }
        }
        printf("Found bad value: %d\n", bad_found);*/
    }
    
    norm_x = sqrt(norm_x);
    
    // Finish estimating holograms
    CHECK_FOR_ERROR("Finish estimating holograms");
    check_isnan_mdata(&measured);
    
    CUFFT_SAFE_CALL(cufftExecC2C(fft_plan, holo1_d, holo1_d, CUFFT_INVERSE));
    CUFFT_SAFE_CALL(cufftExecC2C(fft_plan, holo2_d, holo2_d, CUFFT_INVERSE));
    cudaDeviceSynchronize();
    
    sparse_discard_imag_kernel<<<grid_dim, block_dim>>>(holo1_d, width, height);
    cudaDeviceSynchronize();
    sparse_discard_imag_kernel<<<grid_dim, block_dim>>>(holo2_d, width, height);
    cudaDeviceSynchronize();
    
    // Compute residuals
    CHECK_FOR_ERROR("Compute residuals");
    
    Hologram estimated_holo_1(params);
    Hologram estimated_holo_2(params);
    CuMat est_data1 = estimated_holo_1.getData();
    CuMat est_data2 = estimated_holo_2.getData();
    est_data1.setCuData(holo1_d, height, width, CV_32FC2);
    est_data2.setCuData(holo2_d, height, width, CV_32FC2);
    estimated_holo_1.setData(est_data1);
    estimated_holo_2.setData(est_data2);
    
    //double holo_norm = sqrt(sparseLipschitzSubtractionNorm(holo2_d, holo1_d, plane_size));
    
    Hologram resid_1(params);
    Hologram resid_2(params);
    //printf("before calcResidualFrom\n");
    //printf("test measured before compute residuals call\n");
    //check_isnan_mdata(&measured);
    calcResidualFrom(&resid_1, &estimated_holo_1, RM_EST_TRUE);
    calcResidualFrom(&resid_2, &estimated_holo_2, RM_EST_TRUE);
    
    float2* resid_1_d = (float2*)resid_1.getData().getCuData();
    float2* resid_2_d = (float2*)resid_2.getData().getCuData();
    //double resid_norm = sqrt(sparseLipschitzSubtractionNorm(resid_2_d, resid_1_d, plane_size));
    //printf("norm of difference between residual holograms: %f\n", resid_norm);
    
    // Reconstruct gradient and compute norm for each plane
    CHECK_FOR_ERROR("Initiate reconstructing residuals");
    //printf("Initiate reconstructing residuals\n");
    SparseGradient gradient2;
    gradient2.initialize(resid_2, &fft_plan);
    gradient.reconstruct(resid_1, RECON_MODE_COMPLEX_CONSTANT_SUM);
    gradient2.reconstruct(resid_2, RECON_MODE_COMPLEX_CONSTANT_SUM);
    double norm_grad = 0;
    for (size_t zid = 0; zid < x1.getNumPlanes(); ++zid)
    {
        gradient.getPlane(&grad_plane, zid);
        gradient2.getPlane(&y0_plane, zid);
        
        grad1_d = (float2*)grad_plane.getCuData();
        grad2_d = (float2*)y0_plane.getCuData();
        
        norm_grad += sparseLipschitzSubtractionNorm(grad2_d, grad1_d, plane_size);
    }
    CHECK_FOR_ERROR("Done reconstructing residuals");
    //printf("Done reconstructing residuals\n");
    
    norm_grad = sqrt(norm_grad);
    
    gradient2.destroy();
    CHECK_FOR_ERROR("after gradient2.destory");
    //printf("after gradient2.destory\n");
    
    // Wipe data to preserve future calls
    cudaMemset(x1_d, 0, plane_size*sizeof(float2));
    cudaMemset(x2_d, 0, plane_size*sizeof(float2));
    cudaMemset(grad1_d, 0, plane_size*sizeof(float2));
    cudaMemset(grad2_d, 0, plane_size*sizeof(float2));
    CHECK_FOR_ERROR("After wiping data");
    //printf("After wiping data\n");
    
    // Compute and return final value
    printf("SparseCompressiveHolo::estimateLipschitz: norm_grad = %f, norm_x = %f\n", norm_grad, norm_x);
    double L = norm_grad / norm_x;
    L = std::max(L, 1e-6);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::estimateLipschitz");
    this->lipschitz = L;
    return L;
}

template <unsigned int blockSize>
__global__ void adjointNorm_kernel(float2* data1, float2* data2, float* buffer, int size)
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
        temp.x = (data1[i].x * data2[i].x) - (data1[i].y * data2[i].y);
        temp.y = (data1[i].x - data2[i].y) + (data1[i].y * data2[i].x);
        sdata[tid] += temp.x;
        unsigned int i2 = i + blockSize;
        temp.x = (data1[i2].x * data2[i2].x) - (data1[i2].y - data2[i2].y);
        temp.y = (data1[i2].x - data2[i2].y) + (data1[i2].y - data2[i2].x);
        sdata[tid] += temp.x;
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

double adjointNorm(void* data1_d, void* data2_d, size_t size)
{
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    
    adjointNorm_kernel<512><<<dimGrid, dimBlock, smemSize>>>
        ((float2*)data1_d, (float2*)data2_d, buffer_d, size);
    
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    float sum1 = 0;
    for (int i = 0; i < dimGrid.x; ++i)
    {
        sum1 += buffer_h[i];
    }

    free(buffer_h);
    cudaFree(buffer_d);

    CHECK_FOR_ERROR("after CompressiveHolo::subtractionNorm");
    return sum1;
}

bool SparseCompressiveHolo::testAdjoint(int seed)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::testAdjoint");
    // Check allocated sizes
    size_t plane_size = width*height;
    size_t min_size = plane_size*sizeof(float2);
    bool failure = false;
    if ((x1_plane.getDataSize() < min_size) || !x1_plane.isAllocated()) failure = true;
    if ((x0_plane.getDataSize() < min_size) || !x0_plane.isAllocated()) failure = true;
    if ((grad_plane.getDataSize() < min_size) || !grad_plane.isAllocated()) failure = true;
    if ((y0_plane.getDataSize() < min_size) || !y0_plane.isAllocated()) failure = true;
    if (failure)
    {
        std::cout << "CompressiveHolo::estimateLipschitz:: Error: Data not allocated" << std::endl;
        std::cout << "Min size is " << min_size << std::endl;
        std::cout << "x1_d size = " << x1_plane.getDataSize()
            << ", allocation "      << x1_plane.isAllocated() << std::endl;
        std::cout << "x2_d size = " << x0_plane.getDataSize()
            << ", allocation "      << x0_plane.isAllocated() << std::endl;
        std::cout << "grad1_d size = " << grad_plane.getDataSize()
            << ", allocation "         << grad_plane.isAllocated() << std::endl;
        std::cout << "grad2_d size = " << y0_plane.getDataSize()
            << ", allocation "         << y0_plane.isAllocated() << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    CHECK_FOR_ERROR("getting CuData from planes");
    float2* x1_d = (float2*)x1_plane.getCuData();
    float2* x2_d = (float2*)x0_plane.getCuData();
    //float2* grad2_d = (float2*)grad_plane.getCuData();
    float2* grad1_d = (float2*)y0_plane.getCuData();
    double* exp_d = (double*)exponent_data.getCuData();
    
    // Aliases for convenience
    float2* holo1_d = grad1_d;
    //float2* holo2_d = grad2_d;
    
    // Holograms must be initialized to zero before adding to them
    cudaMemset(holo1_d, 0, width*height*sizeof(float2));
    //cudaMemset(holo2_d, 0, width*height*sizeof(float2));
    
    // Random number generator initialization
    CHECK_FOR_ERROR("Create random number generator");
    curandGenerator_t gen;
    CURAND_SAFE_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    double mean = 0;
    double std = 1; // Standard normal distribution
    
    double z = params.start_plane;
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    double scale = 2 / std::sqrt(2 * params.num_planes);
    
    // Create a random residual hologram
    Hologram residual(params);
    char read_filename[FILENAME_MAX];
    sprintf(read_filename, params.image_filename, params.start_image);
    residual.read(read_filename);
    residual = residual.crop();
    residual = residual.zeroPad(params.zero_padding);
    
    CuMat resid_data = residual.getData();
    float2* resid_data_d = (float2*)resid_data.getCuData();
    CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)resid_data_d, plane_size*2, mean, std));
    resid_data.setCuData((void*)resid_data_d);
    residual.setData(resid_data);
    gradient.reconstruct(residual, RECON_MODE_COMPLEX_CONSTANT_SUM);
    
    // Compute norm of reconstruction and propagate a random volume to a plane
    CHECK_FOR_ERROR("begin x norm plane steps");
    double volume_norm = 0;
    for (size_t zid = 0; zid < x1.getNumPlanes(); ++zid)
    {
        z = params.plane_list[zid];
        
        CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)x1_d, plane_size*2, mean, std));
        cudaDeviceSynchronize();
        
        // Update estimated hologram from x1
        sparse_phase_scale_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, x1_d, width, height, params.wavelength, -z);
        cudaDeviceSynchronize();
        cufftExecC2C(fft_plan, buffer_plane_d, buffer_plane_d, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        sparse_rs_phase_mult_kernel<<<grid_dim, block_dim>>>
            (buffer_plane_d, buffer_plane_d, width, height, 
            exp_d, -z, scale);
        cudaDeviceSynchronize();
        sparse_add_plane_kernel<<<grid_dim, block_dim>>>(holo1_d, buffer_plane_d, width, height);
        cudaDeviceSynchronize();
        
        // Computer inner product of random plane and reconstructed random gradient
        gradient.getPlane(&grad_plane, zid);
        float2* grad2_d = (float2*)grad_plane.getCuData();
        volume_norm += adjointNorm(grad2_d, x1_d, plane_size);
    }
    
    CUFFT_SAFE_CALL(cufftExecC2C(fft_plan, holo1_d, holo1_d, CUFFT_INVERSE));
    
    // Compute inner product of propagated plane and randomly generated one
    // CURAND_SAFE_CALL(curandGenerateNormal(gen, (float*)holo2_d, plane_size*2, mean, std));
    double planar_norm = adjointNorm(holo1_d, resid_data_d, plane_size);
    
    double diff = std::abs(planar_norm - volume_norm);
    double max_norm = std::max(std::abs(planar_norm), std::abs(volume_norm));
    bool is_valid_adjoint = (diff/max_norm) < 1e-9;
    
    residual.destroy();
    
    // if (!is_valid_adjoint)
    {
        if (!is_valid_adjoint) printf("Functions do not satisfy adjoint property\n");
        printf("planar_norm = %e, volume_norm = %e\n", planar_norm, volume_norm);
        printf("diff = %e, max_norm = %e, test = %e\n", diff, max_norm, diff/max_norm);
    }
    
    CHECK_FOR_ERROR("end SparseCompressiveHolo::testAdjoint");
    return is_valid_adjoint;
}

__global__ void sparse_calc_residual_kernel
    (float2* resid, float2* measured, float2* estimate, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        resid[idx].x = measured[idx].x - estimate[idx].x;
        resid[idx].y = measured[idx].y - estimate[idx].y;
    }
}

__global__ void sparse_calc_residual_inverse_kernel
    (float2* resid, float2* measured, float2* estimate, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        resid[idx].x = estimate[idx].x - measured[idx].x;
        resid[idx].y = estimate[idx].y - measured[idx].y;
    }
}

void SparseCompressiveHolo::calcResidual(Hologram* residual, Hologram* estimate, ResidualMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcResidual");
    DECLARE_TIMING(SP_INV_CALC_RESIDUAL);
    START_TIMING(SP_INV_CALC_RESIDUAL);
    
    // Generate the estimated hologram
    //printf("begin backPropagate estimate\n");
    this->backPropagate(estimate, RECON_MODE_COMPLEX);
    //printf("end backPropagate estimate\n");
    
    CuMat mdata = measured.getData();
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)estimate->getData().getCuData();
    CuMat rdata = residual->getData();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    float2* rdata_d = (float2*)rdata.getCuData();
    
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    if (mode == RM_TRUE_EST)
    {
        sparse_calc_residual_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else if (mode == RM_EST_TRUE)
    {
        sparse_calc_residual_inverse_kernel<<<dim_grid, dim_block>>>
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
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcResidual");
    SAVE_TIMING(SP_INV_CALC_RESIDUAL);
}

void SparseCompressiveHolo::calcInitialResidual
    (Hologram* residual, Hologram* estimate, ResidualMode mode)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcInitialResidual");
    DECLARE_TIMING(SP_INV_CALC_RESIDUAL);
    START_TIMING(SP_INV_CALC_RESIDUAL);
    
    // Generate the estimated hologram
    this->backPropagateInitial(estimate, RECON_MODE_COMPLEX);
    
    CuMat mdata = measured.getData();
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)estimate->getData().getCuData();
    CuMat rdata = residual->getData();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    float2* rdata_d = (float2*)rdata.getCuData();
    
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    if (mode == RM_TRUE_EST)
    {
        sparse_calc_residual_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else if (mode == RM_EST_TRUE)
    {
        sparse_calc_residual_inverse_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else
    {
        std::cout << "CompressiveHolo::calcInitialResidual error unknown mode" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    rdata.setCuData((void*)rdata_d, height, width, CV_32FC2);
    
    residual->setData(rdata);
    residual->setState(HOLOGRAM_STATE_RESIDUAL);
    
    mdata.destroy();
    
    CHECK_FOR_ERROR("end CompressiveHolo::calcInitialResidual");
    SAVE_TIMING(SP_INV_CALC_RESIDUAL);
}

double SparseCompressiveHolo::calcResidualObjective(Hologram residual)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::calcResidualObjective");
    DECLARE_TIMING(SP_INV_CALC_OBJ);
    START_TIMING(SP_INV_CALC_OBJ);
    
    // Compute the squared L2-norm of the residual
    void* resid_d = residual.getData().getCuData();
    CuMat compute_data;
    compute_data.setCuData(resid_d, height, width, 1, CV_32FC2);
    OpticalField compute;
    compute.setData(compute_data);
    float resid_norm = -1; // Just initialize, value won't be used
    compute.calcSumMagnitude(resid_norm);
    
    CHECK_FOR_ERROR("end SparseCompressiveHolo::calcResidualObjective");
    return 0.5 * resid_norm;
}

double SparseCompressiveHolo::calcRegularizationObjective(CompressiveHoloMode mode)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::calcRegularizationObjective");
    DECLARE_TIMING(SP_INV_CALC_OBJ);
    START_TIMING(SP_INV_CALC_OBJ);
    
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
        break;
    default:
    {
        std::cout << "SparseCompressiveHolo::calcRegularizationObjective error" << std::endl;
        std::cout << "Mode must be COMPRESSIVE_MODE_FASTA_L1" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    // Calculate norm of the estimate
    double est_norm = -1;
    x1.calcL1NormComplex(&est_norm);
    
    double objective = regularization*est_norm;
    
    if (mode == COMPRESSIVE_MODE_FASTA_FUSED_LASSO)
    {
        double tv_norm = -1;
        x1.calcTVNorm(&tv_norm);
        objective = objective + regularization_TV*tv_norm;
    }
    else if (mode == COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D)
    {
        double tv_norm = -1;
        x1.calcTVNorm2d(&tv_norm);
        objective = objective + regularization_TV*tv_norm;
    }
    
    SAVE_TIMING(SP_INV_CALC_OBJ);
    CHECK_FOR_ERROR("end SparseCompressiveHolo:calcRegularizationObjective");
    return objective;
}

double SparseCompressiveHolo::calcObjectiveFunction(Hologram residual, CompressiveHoloMode mode)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::calcObjectiveFunction");
    DECLARE_TIMING(SP_INV_CALC_OBJ);
    START_TIMING(SP_INV_CALC_OBJ);
    
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
        break;
    default:
    {
        std::cout << "SparseCompressiveHolo::calcObjectiveFunction error" << std::endl;
        std::cout << "Mode must be COMPRESSIVE_MODE_FASTA_L1" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    // Compute the squared L2-norm of the residual
    void* resid_d = residual.getData().getCuData();
    CuMat compute_data;
    compute_data.setCuData(resid_d, height, width, 1, CV_32FC2);
    OpticalField compute;
    compute.setData(compute_data);
    float resid_norm = -1;
    compute.calcSumMagnitude(resid_norm);
    
    // Calculate norm of the estimate
    double est_norm = -1;
    x1.calcL1NormComplex(&est_norm);
    
    //printf("SparseCompressiveHolo::calcObjectiveFunction: resid_norm = %f, est_norm = %f\n", resid_norm, est_norm);
    double objective = 0.5*resid_norm + regularization*est_norm;
    
    printf("  calcObjectiveFunction norms: resid = %0.2f (%0.2f), L1 = %0.2f (%0.2f) ",
        resid_norm, 0.5*resid_norm, est_norm, regularization*est_norm);
    
    if (mode == COMPRESSIVE_MODE_FASTA_FUSED_LASSO)
    {
        double tv_norm = -1;
        x1.calcTVNorm(&tv_norm);
        objective = objective + regularization_TV*tv_norm;
        printf("TV = %0.2f (%0.2f)", tv_norm, regularization_TV*tv_norm);
    }
    else if (mode == COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D)
    {
        double tv_norm = -1;
        x1.calcTVNorm2d(&tv_norm);
        objective = objective + regularization_TV*tv_norm;
        printf("TV = %0.2f (%0.2f)", tv_norm, regularization_TV*tv_norm);
    }
    
    printf("total = %0.2f\n", objective);
    
    SAVE_TIMING(SP_INV_CALC_OBJ);
    CHECK_FOR_ERROR("end SparseCompressiveHolo:calcObjectiveFunction");
    return objective;
}

double SparseCompressiveHolo::calcInitialObjectiveFunction(CompressiveHoloMode mode)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::calcObjectiveFunction");
    
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
        break;
    default:
    {
        std::cout << "SparseCompressiveHolo::calcInitialObjectiveFunction error" << std::endl;
        std::cout << "Mode must be COMPRESSIVE_MODE_FASTA_L1" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO" << std::endl;
        std::cout << "    or COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
    
    Hologram residual(params);
    Hologram estimate(params);
    residual.setSize(width, height);
    estimate.setSize(width, height);
    gradient.reconstruct(measured, RECON_MODE_COMPLEX_CONSTANT_SUM);
    this->calcInitialResidual(&residual, &estimate, RM_EST_TRUE);
    
    double minmax_val = 0.25;
    char temp_filename[FILENAME_MAX];
    // Hologram est_enh(params);
    // estimate.copyTo(&est_enh);
    // CuMat est_enh_data = est_enh.getData();
    // CuMat est_data = estimate.getData();
    // CuMat bg_data = bg_holo.getData();
    // subtract(est_data, bg_data, est_enh_data);
    // est_enh.setData(est_enh_data);
    sprintf(temp_filename, "%s/initial_residual.tif", params.output_path);
    residual.write(temp_filename, -minmax_val, minmax_val);
    sprintf(temp_filename, "%s/estimate_initial.tif", params.output_path);
    estimate.write(temp_filename, -minmax_val, minmax_val);
    
    // CuMat measured_data = measured.getData();
    // subtract(measured_data, bg_data, est_enh_data);
    // est_enh.setData(est_enh_data);
    sprintf(temp_filename, "%s/initial_measured.tif", params.output_path);
    measured.write(temp_filename, -minmax_val, minmax_val);
    
    // Compute the squared L2-norm of the residual
    void* resid_d = residual.getData().getCuData();
    CuMat compute_data;
    compute_data.setCuData(resid_d, height, width, 1, CV_32FC2);
    OpticalField compute;
    compute.setData(compute_data);
    float resid_norm = -1;
    compute.calcSumMagnitude(resid_norm);
    printf("\nInitial residual norm = %f\n", resid_norm);
    
    void* measured_d = measured.getData().getCuData();
    compute_data.setCuData(measured_d, height, width, 1, CV_32FC2);
    compute.setData(compute_data);
    float measured_norm = -1;
    compute.calcSumMagnitude(measured_norm);
    printf("Norm of raw hologram = %f\n\n", measured_norm);
    
    return resid_norm;
}

//void SparseCompressiveHolo::denoise(SparseVolume x1, SparseVolume x0, SparseGradient grad, CompressiveHoloMode mode)
void SparseCompressiveHolo::denoise(CompressiveHoloMode mode)
{
    switch (mode)
    {
    case COMPRESSIVE_MODE_FASTA_L1:
    {
        this->proximalL1();
        break;
    }
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO:
    {
        this->proximalFusedLasso();
        break;
    }
    case COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D:
    {
        this->proximalFusedLasso2d();
        //this->proximalAlternatingFusedLasso2d();
        break;
    }
    default:
    {
        std::cout << "SparseCompressiveHolo::denoise: Unsupported mode: " << mode << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    }
}

__global__ void sparse_soft_threshold_FASTA_kernel
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
        
        // Scale used for elastic net
        float scale = 1.0;//1/(1 + stepsize*regularization * 0.1);
        out_x[idx].x = scale * y.x * in.x;
        out_x[idx].y = scale * y.y * in.y;
        // out_x[idx].x = (absx > T)? in.x : 0.0;
        // out_x[idx].y = (absy > T)? in.y : 0.0;
    }
}

__global__ void simple_soft_threshold_kernel
    (float2* out_x, double stepsize, double regularization, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = out_x[idx].x;
        in.y = out_x[idx].y;
        
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
        
        // absolute threshold
        // out_x[idx].x = (in.x > T)? in.x : 0.0;
        // out_x[idx].y = (in.y > T)? in.y : 0.0;
        // out_x[idx].x = (absx > T)? in.x : 0.0;
        // out_x[idx].y = (absy > T)? in.y : 0.0;
        // out_x[idx].x = (absx > T)? absx : 0.0;
        // out_x[idx].y = (absy > T)? absy : 0.0;
        
        // Reverse threshold
        // out_x[idx].x = (absx > T)? 0.0 : in.x;
        // out_x[idx].y = (absy > T)? 0.0 : in.y;
    }
}

__global__ void block_soft_threshold_kernel
    (float2* out_x, double stepsize, double regularization, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = out_x[idx].x;
        in.y = out_x[idx].y;
        
        float T = stepsize * regularization;
        
        float absx = abs(in.x);
        float absy = abs(in.y);
        float p_absx = absx;//*absx;
        float p_absy = absy;//*absy;
         float2 y;
        y.x = max(p_absx - T, 0.0);
        y.y = max(p_absy - T, 0.0);
        y.x = y.x / (y.x + T);
        y.y = y.y / (y.y + T);
        
        out_x[idx].x = y.x * in.x;
        out_x[idx].y = y.y * in.y;
        
        //float mag = sqrt(in.x*in.x + in.y*in.y);
        //float u = in.x - T*in.x/mag;
        //float v = in.y - T*in.y/mag;
        
        // out_x[idx].x = (mag > T)? u : 0.0;
        // out_x[idx].y = (mag > T)? v : 0.0;
        //float2 y;
        //y.x = (mag > T)? u : 0.0;
        //y.y = (mag > T)? v : 0.0;
        
        // Enforce positivity (phase between 0 and 90 deg)
        // out_x[idx].x = (out_x[idx].x > 0)? out_x[idx].x : 0.0;
        // out_x[idx].y = (out_x[idx].y > 0)? out_x[idx].y : 0.0;
        // out_x[idx].x = max(y.x, 0.0);
        // out_x[idx].y = max(y.y, 0.0);
        //out_x[idx].x = min(y.x, 0.0);
        //out_x[idx].y = min(y.y, 0.0);
        // out_x[idx].x = y.x;
        // out_x[idx].y = y.y;
    }
}

__global__ void block_soft_threshold_FASTA_kernel
    (float2* out_x, float2* xm1, float2* grad, double stepsize, double regularization, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = xm1[idx].x - stepsize*grad[idx].x;
        in.y = xm1[idx].y - stepsize*grad[idx].y;
        
        float T = stepsize * regularization;
        
        float absx = abs(in.x);
        float absy = abs(in.y);
        float p_absx = absx;//*absx;
        float p_absy = absy;//*absy;
        float2 y;
        y.x = max(p_absx - T, 0.0);
        y.y = max(p_absy - T, 0.0);
        y.x = y.x / (y.x + T);
        y.y = y.y / (y.y + T);
        
        out_x[idx].x = y.x * in.x;
        out_x[idx].y = y.y * in.y;
        
        //float mag = sqrt(in.x*in.x + in.y*in.y);
        //float u = in.x - T*in.x/mag;
        //float v = in.y - T*in.y/mag;
        
        // out_x[idx].x = (mag > T)? u : 0.0;
        // out_x[idx].y = (mag > T)? v : 0.0;
        //float2 y;
        //y.x = (mag > T)? u : 0.0;
        //y.y = (mag > T)? v : 0.0;
        
        // Enforce positivity (phase between 0 and 90 deg)
        // out_x[idx].x = (out_x[idx].x > 0)? out_x[idx].x : 0.0;
        // out_x[idx].y = (out_x[idx].y > 0)? out_x[idx].y : 0.0;
        // out_x[idx].x = max(y.x, 0.0);
        // out_x[idx].y = max(y.y, 0.0);
        // out_x[idx].x = min(y.x, 0.0);
        // out_x[idx].y = min(y.y, 0.0);
        //out_x[idx].x = y.x;
        //out_x[idx].y = y.y;
    }
}

__global__ void absorption_soft_threshold_FASTA_kernel
    (float2* out_x, float2* xm1, float2* grad, float bg_abs, float2 bg,
     double stepsize, double regularization, double original_stepsize, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = xm1[idx].x - stepsize*grad[idx].x;
        in.y = xm1[idx].y - stepsize*grad[idx].y;
        
        float2 total;
        // total.x = in.x*bg_abs/original_stepsize + bg.x;
        // total.y = in.y*bg_abs/original_stepsize + bg.y;
        total.x = in.x*bg_abs/original_stepsize + bg_abs*bg_abs;
        total.y = in.y*bg_abs/original_stepsize + bg_abs*bg_abs;
        
        float total_mag = total.x*total.x + total.y*total.y;
        // float bg_mag = bg.x*bg.x + bg.y*bg.y;
        float bg_mag = bg_abs*bg_abs;
        float absorption = -(total_mag - bg_mag);
        
        float T = stepsize * regularization;
        
        // float absx = abs(in.x);
        // float absy = abs(in.y);
        // float p_absx = absx;//*absx;
        // float p_absy = absy;//*absy;
        // float2 y;
        // y.x = max(p_absx - T, 0.0);
        // y.y = max(p_absy - T, 0.0);
        // y.x = y.x / (y.x + T);
        // y.y = y.y / (y.y + T);
        
        // out_x[idx].x = y.x * in.x;
        // out_x[idx].y = y.y * in.y;
        
        float mag = sqrt(in.x*in.x + in.y*in.y);
        float u = in.x - T*in.x/mag;
        float v = in.y - T*in.y/mag;
        
        // out_x[idx].x = (mag > T)? u : 0.0;
        // out_x[idx].y = (mag > T)? v : 0.0;
        float2 y;
        y.x = (mag > T)? u : 0.0;
        y.y = (mag > T)? v : 0.0;
        
        // Enforce positive absorption (non-emmiting)
        // y.x = in.x;
        // y.y = in.y;
        // y.x = (absorption > 0)? y.x : 0.0;
        // y.y = (absorption > 0)? y.y : 0.0;
        
        // Enforce positivity (phase between 0 and 90 deg)
        // out_x[idx].x = (out_x[idx].x > 0)? out_x[idx].x : 0.0;
        // out_x[idx].y = (out_x[idx].y > 0)? out_x[idx].y : 0.0;
        // out_x[idx].x = max(y.x, 0.0);
        // out_x[idx].y = max(y.y, 0.0);
        // out_x[idx].x = min(y.x, 0.0);
        // out_x[idx].y = min(y.y, 0.0);
        
        out_x[idx].x = y.x;
        out_x[idx].y = y.y;
        
        // out_x[idx].x = absorption;
        // out_x[idx].x = in.x;
        // out_x[idx].x = (absorption > 0)? absorption : 0.0;
    }
}

__global__ void absorption_soft_threshold_FASTA_kernel
    (float2* out_x, float2* xm1, float2* grad, float2* bg,
     double stepsize, double regularization, double original_stepsize, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float2 in;
        in.x = xm1[idx].x - stepsize*grad[idx].x;
        in.y = xm1[idx].y - stepsize*grad[idx].y;
        
        float2 total;
        total.x = in.x + bg[idx].x*original_stepsize;
        total.y = in.y + bg[idx].y*original_stepsize;
        
        float total_mag = total.x*total.x + total.y*total.y;
        float bg_mag = bg[idx].x*bg[idx].x + bg[idx].y*bg[idx].y;
        bg_mag *= original_stepsize*original_stepsize;
        float absorption = -(total_mag - bg_mag);
        
        float T = stepsize * regularization;
        
        float mag = sqrt(in.x*in.x + in.y*in.y);
        float u = in.x - T*in.x/mag;
        float v = in.y - T*in.y/mag;
        
        float2 y;
        y.x = (mag > T)? u : 0.0;
        y.y = (mag > T)? v : 0.0;
        
        // Enforce positive absorption (non-emmiting)
        // y.x = in.x;
        // y.y = in.y;
        y.x = (absorption > 0)? y.x : 0.0;
        y.y = (absorption > 0)? y.y : 0.0;
        
        out_x[idx].x = y.x;
        out_x[idx].y = y.y;
        
        // out_x[idx].x = absorption;
        // out_x[idx].x = in.x;
        // out_x[idx].x = (absorption > 0)? absorption : 0.0;
    }
}

void SparseCompressiveHolo::proximalL1()
{
    DECLARE_TIMING(SP_INV_DENOISE);
    START_TIMING(SP_INV_DENOISE);
    //printf("SparseCompressiveHolo::denoise:\n");
    printf("tau = %f, stepsize = %f, regularization = %f\n", stepsize*regularization, stepsize, regularization);
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    
    // float bg_abs = this->measured.getSqrtBackgroundMean();
    // double original_stepsize = 1/this->lipschitz;
    // double original_stepsize = 1.0/(double)depth;
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        //printf("denoise plane %d\n", zid);
        DECLARE_TIMING(DENOISE_GETTERS_A1);
        START_TIMING(DENOISE_GETTERS_A1);
        x1.getPlane(&x1_plane, zid);
        SAVE_TIMING(DENOISE_GETTERS_A1);
        DECLARE_TIMING(DENOISE_GETTERS_A2);
        START_TIMING(DENOISE_GETTERS_A2);
        x0.getPlane(&x0_plane, zid);
        SAVE_TIMING(DENOISE_GETTERS_A2);
        DECLARE_TIMING(DENOISE_GETTERS_A3);
        START_TIMING(DENOISE_GETTERS_A3);
        gradient.getPlane(&grad_plane, zid);
        SAVE_TIMING(DENOISE_GETTERS_A3);
        //printf("initially x1.hasNan = %d\n", x1_plane.hasNan());
        //if (x1_plane.hasNan()) printf("denoise plane %d x1 initially has nan!\n", zid);
        //if (x0_plane.hasNan()) printf("denoise plane %d x0 initially has nan!\n", zid);
        //if (grad_plane.hasNan()) printf("denoise plane %d grad initially has nan!\n", zid);
        
        //std::cout << "SparseCompressiveHolo::denoise plane " << zid << std::endl;
        //std::cout << "grad_plane: " << std::endl;
        //std::cout << grad_plane.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
        
        DECLARE_TIMING(DENOISE_GETTERS_B);
        START_TIMING(DENOISE_GETTERS_B);
        float2* x_d = (float2*)x1_plane.getCuData();
        float2* xm1_d = (float2*)x0_plane.getCuData();
        float2* grad_d = (float2*)grad_plane.getCuData();
        SAVE_TIMING(DENOISE_GETTERS_B);
        
        // cv::Rect print_roi(0,0,5,5);
        // std::cout << "Before denoising:" << std::endl;
        // std::cout << "  roi = " << print_roi << std::endl;
        // std::cout << "grad = " << std::endl;
        // std::cout << grad_plane.getReal().getMatData()(print_roi) << std::endl;
        
        // float z = params.plane_list[zid];
        // double phase = 2*M_PI*z/params.wavelength;
        // double p_x = cos(phase);
        // double p_y = sin(phase);
        // float2 bg_phase;
        // bg_phase.x = bg_abs*bg_abs*p_x;
        // bg_phase.y = bg_abs*bg_abs*p_y;
        // printf("plane %d:\n bg_abs = %f, bg_phase = {%f, %f}\n", bg_abs, bg_phase.x, bg_phase.y);
        
        // bg_rec.getPlane(&bg_plane, zid);
        // float2* bg_d = (float2*)bg_plane.getCuData();
        
        // sparse_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
        //     (x_d, xm1_d, grad_d, stepsize, regularization, width*height);
        block_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
            (x_d, xm1_d, grad_d, stepsize, regularization, width*height);
        // absorption_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
        //     (x_d, xm1_d, grad_d, bg_abs, bg_phase, stepsize, regularization, original_stepsize, width*height);
        // absorption_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
        //     (x_d, xm1_d, grad_d, bg_d, stepsize, regularization, original_stepsize, width*height);
        
        DECLARE_TIMING(DENOISE_SETTERS_A);
        START_TIMING(DENOISE_SETTERS_A);
        x1_plane.setCuData(x_d);
        SAVE_TIMING(DENOISE_SETTERS_A);
        //if (!(zid%10)) printf("denoise plane %d: after threshold x1.hasNan = %d\n", zid, x1_plane.hasNan());
        //if (x1_plane.hasNan()) printf("denoise plane %d has nan!\n", zid);
        DECLARE_TIMING(DENOISE_SETTERS_B);
        START_TIMING(DENOISE_SETTERS_B);
        x1.setPlane(x1_plane, zid);
        SAVE_TIMING(DENOISE_SETTERS_B);
        
        // std::cout << "After denoising:" << std::endl;
        // std::cout << "  roi = " << print_roi << std::endl;
        // std::cout << "x1 = " << std::endl;
        // std::cout << x1_plane.getReal().getMatData()(print_roi) << std::endl;
        
        // cv::namedWindow("absorption", cv::WINDOW_AUTOSIZE);
        // cv::Mat absorption = x1_plane.getReal().getMatData();
        // double min_val, max_val;
        // cv::minMaxLoc(absorption, &min_val, &max_val);
        // printf("min_val = %f, max_val = %f\n", min_val, max_val);
        // // min_val = -0.5;
        // // max_val = 0.5;
        // absorption = (absorption - min_val) / (max_val - min_val);
        // cv::imshow("absorption", absorption);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
        // return;
        
        x0.unGetPlane(&x0_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
        
        //std::cout << "post x1_plane: " << std::endl;
        //std::cout << x1_plane.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    }
    
    /*x1.getPlane(&x1_plane, 0);
    std::cout << "Recovered first x1_plane: " << std::endl;
    std::cout << x1_plane.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    printf("Test plane 0 for nan: %d\n", x1_plane.hasNan());
    x1.getPlane(&x1_plane, 255);
    std::cout << "Recovered last x1_plane: " << std::endl;
    std::cout << x1_plane.getMatData()(cv::Rect(0,0,4,4)) << std::endl;
    printf("Test plane 255 for nan: %d\n", x1_plane.hasNan());*/
    
    SAVE_TIMING(SP_INV_DENOISE);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::proximalL1");
}

void SparseCompressiveHolo::enforceSparsity()
{
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    printf("**enforcing sparsity with threshold = %f\n", stepsize*regularization);
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1.getPlane(&x1_plane, zid);
        float2* x_d = (float2*)x1_plane.getCuData();
        simple_soft_threshold_kernel<<<grid_dim, block_dim>>>
            (x_d, stepsize, regularization, width*height);
        x1_plane.setCuData(x_d);
        x1.setPlane(x1_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
        
        y0.getPlane(&y0_plane, zid);
        float2* y_d = (float2*)y0_plane.getCuData();
        simple_soft_threshold_kernel<<<grid_dim, block_dim>>>
            (y_d, stepsize, regularization, width*height);
        y0_plane.setCuData(y_d);
        y0.setPlane(y0_plane, zid);
        y0.unGetPlane(&y0_plane, zid);
    }
    
    CHECK_FOR_ERROR("end SparseCompressiveHolo::enforceSparsity");
}

void SparseCompressiveHolo::proximalFusedLasso()
{
    DECLARE_TIMING(SP_INV_PROX_FL);
    START_TIMING(SP_INV_PROX_FL);
    
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    
    cv::Rect roi(100,150,10,10);
    
    // First, apply the soft threshold with half the threshold to ensure a
    // sparse input to the total variation proximal calculation
    printf("Soft threshold regularization = %f, stepsize = %f\n", regularization, stepsize);
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1.getPlane(&x1_plane, zid);
        x0.getPlane(&x0_plane, zid);
        gradient.getPlane(&grad_plane, zid);
        
        float2* x_d = (float2*)x1_plane.getCuData();
        float2* xm1_d = (float2*)x0_plane.getCuData();
        float2* grad_d = (float2*)grad_plane.getCuData();
        
        // sparse_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
        //     (x_d, xm1_d, grad_d, stepsize, regularization/2.0, width*height);
        block_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
            (x_d, xm1_d, grad_d, stepsize, regularization/2.0, width*height);
        
        x1_plane.setCuData(x_d);
        x1.setPlane(x1_plane, zid);
        x0.unGetPlane(&x0_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
    }
    
    // Here is where the real work is done
    this->totalVariationProjection();
    
    // Apply another half of the threshold to get more sparse output
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1.getPlane(&x1_plane, zid);
        float2* x_d = (float2*)x1_plane.getCuData();
        // simple_soft_threshold_kernel<<<grid_dim, block_dim>>>
        //     (x_d, stepsize, regularization/2.0, width*height);
        block_soft_threshold_kernel<<<grid_dim, block_dim>>>
            (x_d, stepsize, regularization/2.0, width*height);
        x1_plane.setCuData(x_d);
        x1.setPlane(x1_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
    }
    
    SAVE_TIMING(SP_INV_PROX_FL);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::proximalFusedLasso");
}

// out_x = xm1 - stepsize*grad
__global__ void sparse_get_xhat_kernel
    (float2* out_x, float2* xm1, float2* grad, double stepsize, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out_x[idx].x = xm1[idx].x - stepsize*grad[idx].x;
        out_x[idx].y = xm1[idx].y - stepsize*grad[idx].y;
        // out_x[idx].x = xm1[idx].x + stepsize*grad[idx].x;
        // out_x[idx].y = xm1[idx].y + stepsize*grad[idx].y;
    }
}

__global__ void tv_denoised_estimate_2d_kernel
    (float2* out, float2* im, float2* gx, float2* gy, double mu, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xp = (x<Nx-1)? x+1 : 0;
        int yp = (y<Ny-1)? y+1 : 0;
        int idx_xp = y*Nx + xp;
        int idx_yp = yp*Nx + x;
        
        float2 div;
        div.x = gx[idx_xp].x - gx[idx].x +
                gy[idx_yp].x - gy[idx].x;
        div.y = gx[idx_xp].y - gx[idx].y +
                gy[idx_yp].y - gy[idx].y;
        
        out[idx].x = im[idx].x - mu*div.x;
        out[idx].y = im[idx].y - mu*div.y;
    }
}

__global__ void tv_denoised_estimate_2d_alt_kernel
    (float2* out, float2* im, float2* sparse, float2* gx, float2* gy, double mu, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xp = (x<Nx-1)? x+1 : 0;
        int yp = (y<Ny-1)? y+1 : 0;
        int idx_xp = y*Nx + xp;
        int idx_yp = yp*Nx + x;
        
        float2 div;
        div.x = gx[idx_xp].x - gx[idx].x +
                gy[idx_yp].x - gy[idx].x;
        div.y = gx[idx_xp].y - gx[idx].y +
                gy[idx_yp].y - gy[idx].y;
        
        out[idx].x = (sparse[idx].x==0.0)? 0.0 : div.x - im[idx].x/mu;
        out[idx].x = (sparse[idx].y==0.0)? 0.0 : div.y - im[idx].y/mu;
    }
}

__global__ void tv_gradient_projection_2d_kernel
    (float2* gx, float2* gy, float2* y1,
     double scale, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xm = (x>0)? x-1 : Nx-1;
        int ym = (y>0)? y-1 : Ny-1;
        int idx_xm = y*Nx + xm;
        int idx_ym = ym*Nx + x;
        
        // grad is the result of grad(y1)
        float2 grad_x, grad_y;
        grad_x.x = y1[idx_xm].x - y1[idx].x;
        grad_y.x = y1[idx_ym].x - y1[idx].x;
        grad_x.y = y1[idx_xm].y - y1[idx].y;
        grad_y.y = y1[idx_ym].y - y1[idx].y;
        
        // // Determine the scaling for the projection
        // // Norm method A
        // float2 norm_value;
        // norm_value.x = sqrt(grad_x.x*grad_x.x + grad_y.x*grad_y.x);
        // norm_value.y = sqrt(grad_x.y*grad_x.y + grad_y.y*grad_y.y);
        // float2 normalizer;
        // normalizer.x = 1 + scale*norm_value.x;
        // normalizer.y = 1 + scale*norm_value.y;
        
        // After this grad is g + scale*grad(y1)
        grad_x.x = gx[idx].x + scale*grad_x.x;
        grad_y.x = gy[idx].x + scale*grad_y.x;
        grad_x.y = gx[idx].y + scale*grad_x.y;
        grad_y.y = gy[idx].y + scale*grad_y.y;
        
        // Now do the projection (scaling)
        float normalizer = grad_x.x*grad_x.x + grad_x.y*grad_x.y +
                           grad_y.x*grad_y.x + grad_y.y*grad_y.y;
        //normalizer = sqrt(normalizer); // Norm B
        normalizer = max(sqrt(normalizer), 1.0); // Norm C
        // normalizer = 1.0; // Norm D
        
        // // Norm method E
        // float2 normalizer;
        // normalizer.x = max(sqrt(grad_x.x*grad_x.x + grad_y.x*grad_y.x), 1.0);
        // normalizer.y = max(sqrt(grad_x.y*grad_x.y + grad_y.y*grad_y.y), 1.0);
        
        gx[idx].x = grad_x.x / normalizer;
        gx[idx].y = grad_x.y / normalizer;
        gy[idx].x = grad_y.x / normalizer;
        gy[idx].y = grad_y.y / normalizer;
        
        // gx[idx].x = grad_x.x / normalizer.x;
        // gx[idx].y = grad_x.y / normalizer.y;
        // gy[idx].x = grad_y.x / normalizer.x;
        // gy[idx].y = grad_y.y / normalizer.y;
    }
}

__global__ void zero_count_kernel
    (int* count, float2* gx, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        if (gx[idx].x == 0 && gx[idx].y == 0)
        {
            atomicAdd(count, 1);
        }
    }
}

float2 checkGradientSparsity(float2* gx_d, float2* gy_d, size_t width, size_t height)
{
    CHECK_FOR_ERROR("begin checkGradientSparsity");
    float2 sparsity = {0,0};
    
    int* zero_count_x_d;
    int* zero_count_y_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&zero_count_x_d, 1*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&zero_count_y_d, 1*sizeof(int)));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    zero_count_kernel<<<grid_dim, block_dim>>>(zero_count_x_d, gx_d, width, height);
    zero_count_kernel<<<grid_dim, block_dim>>>(zero_count_y_d, gy_d, width, height);
    
    int zero_count_x_h = 0;
    int zero_count_y_h = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&zero_count_x_h, zero_count_x_d, 1*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&zero_count_y_h, zero_count_y_d, 1*sizeof(int), cudaMemcpyDeviceToHost));
    
    int numel = width*height;
    int nonzero_count_x = numel - zero_count_x_h;
    int nonzero_count_y = numel - zero_count_y_h;
    
    sparsity.x = (float)nonzero_count_x / (float)numel;
    sparsity.y = (float)nonzero_count_y / (float)numel;
    
    CuMat gx_data;
    gx_data.setCuData((void*)gx_d, height, width, CV_32FC2);
    cv::Mat mat_data = gx_data.getReal().getMatData();
    cv::Rect roi(385,675,8,8);
    std::cout << "real(gx): roi = " << roi << std::endl << mat_data(roi) << std::endl;
    
    CHECK_FOR_ERROR("end checkGradientSparsity");
    return sparsity;
}

void SparseCompressiveHolo::proximalFusedLasso2d()
{
    DECLARE_TIMING(SP_INV_PROX_FL2D);
    START_TIMING(SP_INV_PROX_FL2D);
    
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    dim3 block_dim2d(16, 16);
    dim3 grid_dim2d(width / block_dim2d.x, height / block_dim2d.y);
    
    // Scaled by 1/4*numdims (3D volume) comes from math
    // double scale = 1/(4*3*regularization_TV);
    double scale = 1/(4*3*stepsize*regularization_TV);
    //double scale = 1/(4*2*regularization_TV);
    //double scale = 1/(stepsize*regularization_TV);
    //scale = 0.05; // Hardcode only for testing with dandelion data
    //double mu = stepsize * regularization_TV / 2;
    
    //printf("    proximalFusedLasso2d scale = %f (regTV = %f, step = %f)\n",
    //    scale, regularization_TV, stepsize);
    
    float2* gx_d = (float2*)gx_plane.getCuData();
    float2* gy_d = (float2*)gy_plane.getCuData();
    float2* temp_d = (float2*)y1_plane.getCuData();
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1.getPlane(&x1_plane, zid);
        x0.getPlane(&x0_plane, zid);
        gradient.getPlane(&grad_plane, zid);
        float2* x_d = (float2*)x1_plane.getCuData();
        float2* xm1_d = (float2*)x0_plane.getCuData();
        float2* grad_d = (float2*)grad_plane.getCuData();
        
        // Need to set x = x0 - stepsize*grad
        sparse_get_xhat_kernel<<<grid_dim, block_dim>>>
            (x_d, xm1_d, grad_d, stepsize, width*height);
        
        // Total variation projection
        CUDA_SAFE_CALL(cudaMemset(gx_d, 0, width*height*sizeof(float2)));
        CUDA_SAFE_CALL(cudaMemset(gy_d, 0, width*height*sizeof(float2)));
        for (int i = 0; i < num_tv_iterations; ++i)
        {
            // y1 = x1 - mu*div(g)
            tv_denoised_estimate_2d_kernel<<<grid_dim2d, block_dim2d>>>
                (temp_d, x_d, gx_d, gy_d, stepsize*regularization_TV, width, height);
            
            // g = projectIsotropic(g + scale*grad(y1))
            tv_gradient_projection_2d_kernel<<<grid_dim2d, block_dim2d>>>
                (gx_d, gy_d, temp_d, scale, width, height);
        }
        tv_denoised_estimate_2d_kernel<<<grid_dim2d, block_dim2d>>>
            (temp_d, x_d, gx_d, gy_d, stepsize*regularization_TV, width, height);
        
        CUDA_SAFE_CALL(cudaMemcpy(x_d, temp_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice));
        
        /*if (zid == 7)
        {
            float2 grad_sparsity = checkGradientSparsity(gx_d, gy_d, width, height);
            printf("    plane %d sparsity = (%f, %f)\n", zid, grad_sparsity.x, grad_sparsity.y);
        }*/
        
        // Apply L1 threshold to get sparse output
        if (regularization > 0.0)
        {
            // simple_soft_threshold_kernel<<<grid_dim, block_dim>>>
            //     (x_d, stepsize, regularization, width*height);
            block_soft_threshold_kernel<<<grid_dim, block_dim>>>
                (x_d, stepsize, regularization, width*height);
        }
        
        cudaDeviceSynchronize();
        x1_plane.setCuData(x_d);
        x1.setPlane(x1_plane, zid);
        x0.unGetPlane(&x0_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
    }
    
    SAVE_TIMING(SP_INV_PROX_FL2D);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::proximalFusedLasso2d");
}

__global__ void duplicate_sparsity_kernel
        (float2* data, float2* sparse, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        data[idx].x = (sparse[idx].x != 0.0)? data[idx].x : 0.0;
        data[idx].y = (sparse[idx].y != 0.0)? data[idx].y : 0.0;
    }
}

void SparseCompressiveHolo::proximalAlternatingFusedLasso2d()
{
    DECLARE_TIMING(SP_INV_PROX_FL2D);
    START_TIMING(SP_INV_PROX_FL2D);
    
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    dim3 block_dim2d(16, 16);
    dim3 grid_dim2d(width / block_dim2d.x, height / block_dim2d.y);
    
    // Scaled by 1/4*numdims (3D volume) comes from math
    double scale = 1/(4*3*regularization_TV);
    
    float2* gx_d = (float2*)gx_plane.getCuData();
    float2* gy_d = (float2*)gy_plane.getCuData();
    float2* temp_d = (float2*)y1_plane.getCuData();
    
    bool is_lasso_iteration = false;
    bool is_tv_iteration = false;
    
    int cutoff_iteration = params.num_inverse_iterations/10;
    if (this->iteration < cutoff_iteration)// || (iteration%10 <= 2))
    {
        is_lasso_iteration = true;
        printf("is a lasso iteration\n");
    }
    else
    {
        is_tv_iteration = true;
    }
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1.getPlane(&x1_plane, zid);
        x0.getPlane(&x0_plane, zid);
        gradient.getPlane(&grad_plane, zid);
        float2* x_d = (float2*)x1_plane.getCuData();
        float2* xm1_d = (float2*)x0_plane.getCuData();
        float2* grad_d = (float2*)grad_plane.getCuData();
        
        if (is_lasso_iteration)
        {
            sparse_soft_threshold_FASTA_kernel<<<grid_dim, block_dim>>>
                (x_d, xm1_d, grad_d, stepsize, regularization, width*height);
        }
        
        if (is_tv_iteration)
        {
            // Need to set x = x0 - stepsize*grad
            sparse_get_xhat_kernel<<<grid_dim, block_dim>>>
                (x_d, xm1_d, grad_d, stepsize, width*height);
            
            // Total variation projection
            CUDA_SAFE_CALL(cudaMemset(gx_d, 0, width*height*sizeof(float2)));
            CUDA_SAFE_CALL(cudaMemset(gy_d, 0, width*height*sizeof(float2)));
            for (int i = 0; i < num_tv_iterations; ++i)
            {
                // y1 = x1 - mu*div(g)
                tv_denoised_estimate_2d_kernel<<<grid_dim2d, block_dim2d>>>
                    (temp_d, x_d, gx_d, gy_d, regularization_TV, width, height);
                
                // g = projectIsotropic(g + scale*grad(y1))
                tv_gradient_projection_2d_kernel<<<grid_dim2d, block_dim2d>>>
                    (gx_d, gy_d, temp_d, scale, width, height);
            }
            tv_denoised_estimate_2d_kernel<<<grid_dim2d, block_dim2d>>>
                (temp_d, x_d, gx_d, gy_d, regularization_TV, width, height);
            
            CUDA_SAFE_CALL(cudaMemcpy(x_d, temp_d, width*height*sizeof(float2), cudaMemcpyDeviceToDevice));
            
            duplicate_sparsity_kernel<<<grid_dim2d, block_dim2d>>>
                (x_d, xm1_d, width, height);
        }
        
        cudaDeviceSynchronize();
        x1_plane.setCuData(x_d);
        x1.setPlane(x1_plane, zid);
        x0.unGetPlane(&x0_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
    }
    
    SAVE_TIMING(SP_INV_PROX_FL2D);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::proximalFusedLasso2d");
}

void SparseCompressiveHolo::totalVariationProjection()
{
    int block_dim = 256;
    int grid_dim = ceil(width*height / (double)block_dim);
    cv::Rect roi(100,150,10,10);
    
    gx.erase();
    gy.erase();
    gz.erase();
    
    for (int i = 0; i < num_tv_iterations; ++i)
    {
        // y1 is used as a temporary storage buffer
        // y1 = x1 - mu*div(g)
        totalVariationDenoisedEstimate();
        
        // g = projectIsotropic(g + scale*grad(y1))
        totalVariationGradientOperator();
        CHECK_MEMORY("end SparseCompressiveHolo::totalVariationProjection iteration");
    }
    
    totalVariationDenoisedEstimate();
    
    x1 = y1;
}

__global__ void tv_denoised_estimate_kernel
    (float2* out, float2* im, float2* gx, float2* gy, float2* gz, float2* gzp, double mu, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xp = (x<Nx-1)? x+1 : 0;
        int yp = (y<Ny-1)? y+1 : 0;
        int idx_xp = y*Nx + xp;
        int idx_yp = yp*Nx + x;
        
        float2 div;
        div.x = gx[idx_xp].x - gx[idx].x + 
                gy[idx_yp].x - gy[idx].x + 
                gzp[idx].x - gz[idx].x;
        div.y = gx[idx_xp].y - gx[idx].y + 
                gy[idx_yp].y - gy[idx].y + 
                gzp[idx].y - gz[idx].y;
        
        out[idx].x = im[idx].x - mu*div.x;
        out[idx].y = im[idx].y - mu*div.y;
    }
}

void SparseCompressiveHolo::totalVariationDenoisedEstimate()
{
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    CuMat next_gz_plane;
    gz.getPlane(&gz_plane, 0);
    
    // y1 can be used as a temporary buffer, value will be overwritten
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        y1.getPlane(&y1_plane, zid);
        x1.getPlane(&x1_plane, zid);
        gx.getPlane(&gx_plane, zid);
        gy.getPlane(&gy_plane, zid);
        if (zid < depth-1)
            gz.getPlane(&next_gz_plane, zid+1);
        else
            gz.getPlane(&next_gz_plane, 0);
        
        float2* y_d = (float2*)y1_plane.getCuData();
        float2* x_d = (float2*)x1_plane.getCuData();
        float2* gx_d = (float2*)gx_plane.getCuData();
        float2* gy_d = (float2*)gy_plane.getCuData();
        float2* gz_d = (float2*)gz_plane.getCuData();
        float2* gzp_d = (float2*)next_gz_plane.getCuData();
        
        tv_denoised_estimate_kernel<<<grid_dim, block_dim>>>
            (y_d, x_d, gx_d, gy_d, gz_d, gzp_d, stepsize*regularization_TV, width, height);
        
        y1_plane.setCuData(y_d);
        y1.setPlane(y1_plane, zid);
        x1.unGetPlane(&x1_plane, zid);
        gx.unGetPlane(&gx_plane, zid);
        gy.unGetPlane(&gy_plane, zid);
        y1.unGetPlane(&y1_plane, zid);
        
        gz_plane = next_gz_plane;
    }
    
    next_gz_plane.destroy();
    CHECK_FOR_ERROR("end SparseCompressiveHolo::totalVariationDenoisedEstimate");
}

__global__ void tv_gradient_projection_kernel
    (float2* gx, float2* gy, float2* gz, float2* y1, float2* y1m,
     double scale, size_t Nx, size_t Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (idx < Nx*Ny)
    {
        int xm = (x>0)? x-1 : Nx-1;
        int ym = (y>0)? y-1 : Ny-1;
        int idx_xm = y*Nx + xm;
        int idx_ym = ym*Nx + x;
        
        // grad is the result of grad(y1)
        float2 grad_x, grad_y, grad_z;
        grad_x.x = y1[idx_xm].x - y1[idx].x;
        grad_y.x = y1[idx_ym].x - y1[idx].x;
        grad_z.x = y1m[idx].x - y1[idx].x;
        grad_x.y = y1[idx_xm].y - y1[idx].y;
        grad_y.y = y1[idx_ym].y - y1[idx].y;
        grad_z.y = y1m[idx].y - y1[idx].y;
        
        // After this grad is g + scale*grad(y1)
        grad_x.x = gx[idx].x + scale*grad_x.x;
        grad_y.x = gy[idx].x + scale*grad_y.x;
        grad_z.x = gz[idx].x + scale*grad_z.x;
        grad_x.y = gx[idx].y + scale*grad_x.y;
        grad_y.y = gy[idx].y + scale*grad_y.y;
        grad_z.y = gz[idx].y + scale*grad_z.y;
        
        // Now do the projection (scaling)
        float normalizer = grad_x.x*grad_x.x + grad_x.y*grad_x.y +
                           grad_y.x*grad_y.x + grad_y.y*grad_y.y +
                           grad_z.x*grad_z.x + grad_z.y*grad_z.y;
        normalizer = max(sqrt(normalizer), 1.0);
        normalizer = 1.0;
        
        gx[idx].x = grad_x.x / normalizer;
        gx[idx].y = grad_x.y / normalizer;
        gy[idx].x = grad_y.x / normalizer;
        gy[idx].y = grad_y.y / normalizer;
        gz[idx].x = grad_z.x / normalizer;
        gz[idx].y = grad_z.y / normalizer;
    }
}

// g = projectIsotropic(g + scale*grad(y1))
void SparseCompressiveHolo::totalVariationGradientOperator()
{
    dim3 block_dim(16, 16);
    dim3 grid_dim(width / block_dim.x, height / block_dim.y);
    
    // Scaled by 1/4*numdims (3D volume) comes from math
    double scale = 1/(4*3*stepsize*regularization_TV);
    
    CuMat prev_y_plane;
    y1.getPlane(&prev_y_plane, depth-1);
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        y1.getPlane(&y1_plane, zid);
        gx.getPlane(&gx_plane, zid);
        gy.getPlane(&gy_plane, zid);
        gz.getPlane(&gz_plane, zid);
        
        float2* y_d = (float2*)y1_plane.getCuData();
        float2* ym_d = (float2*)prev_y_plane.getCuData();
        float2* gx_d = (float2*)gx_plane.getCuData();
        float2* gy_d = (float2*)gy_plane.getCuData();
        float2* gz_d = (float2*)gz_plane.getCuData();
        
        tv_gradient_projection_kernel<<<grid_dim, block_dim>>>
            (gx_d, gy_d, gz_d, y_d, ym_d, scale, width, height);
        
        gx_plane.setCuData(gx_d);
        gy_plane.setCuData(gy_d);
        gz_plane.setCuData(gz_d);
        gx.setPlane(gx_plane, zid);
        gy.setPlane(gy_plane, zid);
        gz.setPlane(gz_plane, zid);
        gx.unGetPlane(&gx_plane, zid);
        gy.unGetPlane(&gy_plane, zid);
        gz.unGetPlane(&gz_plane, zid);
        
        prev_y_plane = y1_plane;
        y1.unGetPlane(&y1_plane, zid);
    }
    
    prev_y_plane.destroy();
    CHECK_FOR_ERROR("end SparseCompressiveHolo::totalVariationGradientOperator");
}

__global__ void tv_denoised_estimate_1dz_kernel
    (float2* out, float2* in, float2* gz, double mu, size_t Nx, size_t Ny, size_t Nz)
{
    // x and z are swapped to fit in max block dims of (1024, 1024, 64)
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z*Nx*Ny + y*Nx + x;
    if (idx < Nx*Ny*Nz)
    {
        int zp = (z<Nz-1)? z+1 : 0;
        int idx_zp = zp*Nx*Ny + y*Nx + x;
        
        float2 div;
        div.x = gz[idx_zp].x - gz[idx].x;
        div.y = gz[idx_zp].y - gz[idx].y;
        
        out[idx].x = in[idx].x - mu*div.x;
        out[idx].y = in[idx].y - mu*div.y;
    }
}

__global__ void tv_gradient_projection_1dz_kernel
    (float2* gz, float2* y1,
     double scale, size_t Nx, size_t Ny, size_t Nz)
{
    // x and z are swapped to fit in max block dims of (1024, 1024, 64)
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z*Nx*Ny + y*Nx + x;
    if (idx < Nx*Ny*Nz)
    {
        int zm = (z>0)? z-1 : Nz-1;
        int idx_zm = zm*Nx*Ny + y*Nx + x;
        
        // grad is the result of grad(y1)
        float2 grad_z;
        grad_z.x = y1[idx_zm].x - y1[idx].x;
        grad_z.y = y1[idx_zm].y - y1[idx].y;
        
        // After this grad is g + scale*grad(y1)
        grad_z.x = gz[idx].x + scale*grad_z.x;
        grad_z.y = gz[idx].y + scale*grad_z.y;
        
        // Now do the projection (scaling)
        float normalizer = grad_z.x*grad_z.x + grad_z.y*grad_z.y;
        normalizer = max(sqrt(normalizer), 1.0);
        
        gz[idx].x = grad_z.x / normalizer;
        gz[idx].y = grad_z.y / normalizer;
    }
}

void SparseCompressiveHolo::proximalFusedLasso1dZ()
{
    printf("begin proximalFusedLasso1dZ\n");
    DECLARE_TIMING(SP_INV_PROX_FL1DZ);
    START_TIMING(SP_INV_PROX_FL1DZ);
    
    // Scaled by 1/4*numdims (3D volume) comes from math
    double scale = 1/(4*3*stepsize*regularization_TV);
    // double scale = 1/(4*stepsize*regularization_TV);
    
    // float2* gx_d = (float2*)gx_plane.getCuData();
    // float2* gy_d = (float2*)gy_plane.getCuData();
    // float2* temp_d = (float2*)y1_plane.getCuData();
    
    // Determine max size of buffer
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t bytes_per_row = width*depth*sizeof(float2) * 3; // Need 3 volumes for processing
    int buffer_rows = free_bytes / (2*bytes_per_row); // Safety factor of 2
    buffer_rows = std::min(buffer_rows, (int)height);
    buffer_rows = 150;
    printf("buffer_rows = %d\n", buffer_rows);
    printf("height = %d\n", height);
    
    int block_dim = 256;
    int grid_dim = ceil(width*buffer_rows / (double)block_dim);
    printf("grid_dim = %d, block_dim = %d\n", grid_dim, block_dim);
    
    int block_dim3d = 256;
    int grid_dim3d = ceil(width*buffer_rows*depth / (double)block_dim);
    printf("grid_dim = %d, block_dim = %d\n", grid_dim, block_dim);
    
    // x and z are swapped to fit in max block dims of (1024, 1024, 64)
    dim3 block_dim1d(256, 1, 1);
    dim3 grid_dim1d(ceil((double)depth/(double)block_dim1d.x), buffer_rows, width);
    
    CuMat buffer;
    CuMat buffer2;
    CuMat gz;
    buffer.allocateCuData(width, buffer_rows, depth, sizeof(float2));
    buffer2.allocateCuData(width, buffer_rows, depth, sizeof(float2));
    gz.allocateCuData(width, buffer_rows, depth, sizeof(float2));
    float2* buffer_x_d = (float2*)buffer.getCuData();
    float2* temp_d = (float2*)buffer2.getCuData();
    float2* gz_d = (float2*)gz.getCuData();
    
    printf("  before iterations\n");
    bool continue_block_iterations = true;
    int buffer_start = 0;
    // for (int buffer_start = 0; buffer_start+buffer_rows < height; buffer_start+=buffer_rows)
    while (continue_block_iterations)
    {
        int buffer_end = buffer_start + buffer_rows;
        if (buffer_end == height) {
            // this is the last iteration
            continue_block_iterations = false;
        }
        else if (buffer_end > height)
        {
            // Adjust start position and then run this as last iteration
            buffer_start = height - buffer_rows;
            buffer_end = buffer_start + buffer_rows;
            continue_block_iterations = false;
        }
        printf("  buffer start: %d, end: %d\n", buffer_start, buffer_end);
        // Fill the buffer volume
        for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
        {
            CHECK_FOR_ERROR("before getCuData calls");
            size_t buffer_offset = zid*width*buffer_rows;
            size_t offset = buffer_start*width;
            x0.getPlane(&x0_plane, zid);
            gradient.getPlane(&grad_plane, zid);
            float2* x_d = buffer_x_d + buffer_offset;
            float2* xm1_d = (float2*)x0_plane.getCuData() + offset;
            float2* grad_d = (float2*)grad_plane.getCuData() + offset;
            
            // Need to set x = x0 - stepsize*grad
            CHECK_FOR_ERROR("before sparse_get_xhat_kernel");
            sparse_get_xhat_kernel<<<grid_dim, block_dim>>>
                (x_d, xm1_d, grad_d, stepsize, width*height);
            CHECK_FOR_ERROR("after sparse_get_xhat_kernel");
            
            x0.unGetPlane(&x0_plane, zid);
        }
        
        // Total variation projection
        // printf("    begin TV projection\n");
        // printf("  grid_dim1d = (%d, %d, %d), block_dim1d = (%d, %d, %d)\n",
        //     grid_dim1d.x, grid_dim1d.y, grid_dim1d.z,
        //     block_dim1d.x, block_dim1d.y, block_dim1d.z);
        for (int i = 0; i < num_tv_iterations; ++i)
        {
            // printf("      iteration %d\n", i);
            tv_denoised_estimate_1dz_kernel<<<grid_dim1d, block_dim1d>>>
                (temp_d, buffer_x_d, gz_d, stepsize*regularization_TV, width, buffer_rows, depth);
            
            // g = projectIsotropic(g + scale*grad(y1))
            tv_gradient_projection_1dz_kernel<<<grid_dim1d, block_dim1d>>>
                (gz_d, temp_d, scale, width, buffer_rows, depth);
        }
        tv_denoised_estimate_1dz_kernel<<<grid_dim1d, block_dim1d>>>
            (temp_d, buffer_x_d, gz_d, stepsize*regularization_TV, width, buffer_rows, depth);
        
        CUDA_SAFE_CALL(cudaMemcpy((void*)buffer_x_d, (void*)temp_d, width*buffer_rows*depth*sizeof(float2), cudaMemcpyDeviceToDevice));
        
        // Apply L1 threshold to get sparse output
        if (regularization > 0.0)
        {
            printf("    begin soft threshold, step = %f, reg = %f\n", stepsize, regularization);
            block_soft_threshold_kernel<<<grid_dim3d, block_dim3d>>>
                (buffer_x_d, stepsize, regularization, width*buffer_rows*depth);
        }
        
        // printf("    store outputs\n");
        for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
        {
            size_t buffer_offset = zid*width*buffer_rows;
            size_t offset = buffer_start*width;
            // offset = 1;
            // buffer_offset = width*height*12343;
            // std::cout << "zid = " << zid
            //           << ", buffer_offset = " << buffer_offset
            //           << ", offset = " << offset << std::endl;
            // printf("  offset = %d * %d = %d\n", buffer_start, width, offset);
            x1.getPlane(&x1_plane, zid);
            float2* x_d = (float2*)x1_plane.getCuData();
            
            CUDA_SAFE_CALL(cudaMemcpy(x_d+offset, buffer_x_d+buffer_offset, width*buffer_rows*sizeof(float2), cudaMemcpyDeviceToDevice));
            // CUDA_SAFE_CALL(cudaMemset(x_d + offset, 0, width*buffer_rows*sizeof(float2)));
            // CUDA_SAFE_CALL(cudaMemset(buffer_x_d + buffer_offset, 0, width*buffer_rows*sizeof(float2)));
            
            CHECK_FOR_ERROR("before final sets");
            cudaDeviceSynchronize();
            x1_plane.setCuData(x_d);
            x1.setPlane(x1_plane, zid);
            x1.unGetPlane(&x1_plane, zid);
        }
        
        buffer_start = buffer_start + buffer_rows;
    }
    printf("  after iterations\n");
    
    buffer.destroy();
    buffer2.destroy();
    gz.destroy();
    
    SAVE_TIMING(SP_INV_PROX_FL1DZ);
    CHECK_FOR_ERROR("end SparseCompressiveHolo::proximalFusedLasso1dZ");
}

template <unsigned int blockSize>
__global__ void sparse_linesearch_1_kernel(float2* x1, float2* x0, float2* grad, float* buffer, int size)
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
__global__ void sparse_linesearch_2_kernel(float2* x1, float2* x0, float* buffer, int size)
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

double SparseCompressiveHolo::calcLineSearchLimit(SparseVolume* x1, SparseVolume* x0, SparseGradient* grad)
{
    CHECK_FOR_ERROR("begin CompressiveHolo::calcLineSearchLimit");
    DECLARE_TIMING(SP_INV_CALC_LINESEARCH);
    START_TIMING(SP_INV_CALC_LINESEARCH);
    
    int size = width*height;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    
    float sum1 = 0;
    float sum2 = 0;
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        CHECK_FOR_ERROR("begining of iteration");
        DECLARE_TIMING(LINESEARCH_GETTERS);
        START_TIMING(LINESEARCH_GETTERS);
        x1->getPlane(&x1_plane, zid);
        x0->getPlane(&x0_plane, zid);
        grad->getPlane(&grad_plane, zid);
        
        float2* x1_d = (float2*)x1_plane.getCuData();
        float2* x0_d = (float2*)x0_plane.getCuData();
        float2* grad_d = (float2*)grad_plane.getCuData();
        SAVE_TIMING(LINESEARCH_GETTERS);
        
        DECLARE_TIMING(LINESEARCH_KERNEL_1);
        START_TIMING(LINESEARCH_KERNEL_1);
        sparse_linesearch_1_kernel<512><<<dimGrid, dimBlock, smemSize>>>
            ((float2*)x1_d, (float2*)x0_d, (float2*)grad_d, buffer_d, size);
        CHECK_FOR_ERROR("after linesearch_1_kernel");
        SAVE_TIMING(LINESEARCH_KERNEL_1);
        
        DECLARE_TIMING(LINESEARCH_COPY_SUM);
        START_TIMING(LINESEARCH_COPY_SUM);
        cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dimGrid.x; ++i)
        {
            sum1 += buffer_h[i];
        }
        SAVE_TIMING(LINESEARCH_COPY_SUM);
        
        DECLARE_TIMING(LINESEARCH_KERNEL_2);
        START_TIMING(LINESEARCH_KERNEL_2);
        sparse_linesearch_2_kernel<512><<<dimGrid, dimBlock, smemSize>>>
            ((float2*)x1_d, (float2*)x0_d, buffer_d, size);
        CHECK_FOR_ERROR("after linesearch_2_kernel");
        SAVE_TIMING(LINESEARCH_KERNEL_2);
        
        START_TIMING(LINESEARCH_COPY_SUM);
        cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dimGrid.x; ++i)
        {
            sum2 += buffer_h[i];
        }
        SAVE_TIMING(LINESEARCH_COPY_SUM);
        
        x1->unGetPlane(&x1_plane, zid);
        x0->unGetPlane(&x0_plane, zid);
        CHECK_FOR_ERROR("end of iteration");
    }
    
    free(buffer_h);
    cudaFree(buffer_d);
    
    SAVE_TIMING(SP_INV_CALC_LINESEARCH);
    CHECK_FOR_ERROR("end CompressiveHolo::calcLineSearchLimit");
    
    double linesearch_value = sum1 + sum2/(2*stepsize);
    printf("    linesearch_value = %f, part1 = %f, part2 = %f\n", linesearch_value, sum1, sum2/(2*stepsize));
    // double linesearch_value = sum1 + 2*stepsize*sum2;
    //printf("calcLineSearchLimit result = %f\n", linesearch_value);
    return linesearch_value;
    // double false_value = 100.0;
    // printf("WARNING: calcLineSearchLimit forced return value or %f instead of %f\n",
    //     false_value, sum1 + sum2/(2*stepsize));
    // return false_value;
}

template <unsigned int blockSize>
__global__ void sparse_fistaRestart_kernel(float2* x0, float2* x1, float2* x_accel, float* buffer, int size)
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

bool SparseCompressiveHolo::fistaRestart(SparseVolume* x0, SparseVolume* x1, SparseVolume* y0)
{
    DECLARE_TIMING(SP_INV_FISTARESTART);
    START_TIMING(SP_INV_FISTARESTART);
    // Evaluates (x0(:)-x1(:))'*(x1(:)-x_accel0(:))>0

    int size = width*height;
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(size / (512.0 * 2.0)), 1, 1);
    int smemSize = 512 * sizeof(float);
    
    float* buffer_d;
    cudaMalloc((void**)&buffer_d, dimGrid.x * sizeof(float));
    float* buffer_h = (float*)malloc(dimGrid.x * sizeof(float));
    
    float sum1 = 0;
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1->getPlane(&x1_plane, zid);
        x0->getPlane(&x0_plane, zid);
        y0->getPlane(&y0_plane, zid);
        
        float2* x1_d = (float2*)x1_plane.getCuData();
        float2* x0_d = (float2*)x0_plane.getCuData();
        float2* a_accel_d = (float2*)y0_plane.getCuData();
        
        sparse_fistaRestart_kernel<512><<<dimGrid, dimBlock, smemSize>>>
            ((float2*)x0_d, (float2*)x1_d, (float2*)a_accel_d, buffer_d, size);
        
        cudaMemcpy(buffer_h, buffer_d, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < dimGrid.x; ++i)
        {
            sum1 += buffer_h[i];
        }
        
        x1->unGetPlane(&x1_plane, zid);
        x0->unGetPlane(&x0_plane, zid);
        y0->unGetPlane(&y0_plane, zid);
    }
    
    free(buffer_h);
    cudaFree(buffer_d);
    
    SAVE_TIMING(SP_INV_FISTARESTART);
    CHECK_FOR_ERROR("after CompressiveHolo::fistaRestart");
    return sum1 > 0;
}

__global__ void sparse_fistaUpdate_kernel(float2* x1, float2* y0, float2* y1, double step, size_t size)
{
    // x1 = y1 + step*(y1-y0);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        x1[idx].x = y1[idx].x + step*(y1[idx].x - y0[idx].x);
        x1[idx].y = y1[idx].y + step*(y1[idx].y - y0[idx].y);
    }
}

//void SparseCompressiveHolo::fistaUpdate(SparseVolume x1, SparseVolume y0, SparseVolume y1, double step)
void SparseCompressiveHolo::fistaUpdate(SparseVolume* x1, SparseVolume* y0, SparseVolume* y1, double step)
{
    CHECK_FOR_ERROR("before SparseCompressiveHolo::fistaUpdate SparseVolume");
    DECLARE_TIMING(SP_INV_FISTAUPDATE_SPVOL);
    START_TIMING(SP_INV_FISTAUPDATE_SPVOL);
    size_t size = width*height;
    int block_dim = 256;
    int grid_dim = ceil(size / (double)block_dim);
    
    for (size_t zid = 0; zid < depth; zid+=plane_subsampling)
    {
        x1->getPlane(&x1_plane, zid);
        y1->getPlane(&y1_plane, zid);
        y0->getPlane(&y0_plane, zid);
        
        //cv::Rect roi(25,20,9,9);
        //if (zid == 2) 
        //    std::cout << "x1_plane 2 before update at " << roi << std::endl 
        //    << x1_plane.getReal().getMatData()(roi) << std::endl;
        
        float2* x1_d = (float2*)x1_plane.getCuData();
        float2* y1_d = (float2*)y1_plane.getCuData();
        float2* y0_d = (float2*)y0_plane.getCuData();
        
        /*if (zid == 2)
        {
            float2* x1_h = (float2*)malloc(size*sizeof(float2));
            float2* y1_h = (float2*)malloc(size*sizeof(float2));
            float2* y0_h = (float2*)malloc(size*sizeof(float2));
            cudaMemcpy(x1_h, x1_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            cudaMemcpy(y1_h, y1_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            cudaMemcpy(y0_h, y0_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            
            printf("Prior to update:\n");
            printf("x1\ty1\ty0\n");
            size_t p = width*20 + 25;
            for (int i = 0; i < 5; ++i) printf("%f, %f, %f\n", x1_h[i+p].x, y1_h[i+p].x, y0_h[i+p].x);
            printf("\n");
            
            free(x1_h);
            free(y1_h);
            free(y0_h);
        }*/
        
        sparse_fistaUpdate_kernel<<<grid_dim, block_dim>>>
            ((float2*)x1_d, (float2*)y0_d, (float2*)y1_d, step, size);
        
        /*if (zid == 2)
        {
            float2* x1_h = (float2*)malloc(size*sizeof(float2));
            float2* y1_h = (float2*)malloc(size*sizeof(float2));
            float2* y0_h = (float2*)malloc(size*sizeof(float2));
            cudaMemcpy(x1_h, x1_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            cudaMemcpy(y1_h, y1_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            cudaMemcpy(y0_h, y0_d, size*sizeof(float2), cudaMemcpyDeviceToHost);
            
            printf("Data from device:\n");
            printf("step = %f\n", step);
            printf("x1\ty1\ty0\n");
            size_t p = width*20 + 25;
            for (int i = 0; i < 5; ++i) printf("%f, %f, %f\n", x1_h[i+p].x, y1_h[i+p].x, y0_h[i+p].x);
            printf("\n");
            
            free(x1_h);
            free(y1_h);
            free(y0_h);
        }*/
        
        x1_plane.setCuData(x1_d);
        x1->setPlane(x1_plane, zid);
        //if (zid == 2) 
        //    std::cout << "x1_plane 2 after update at " << roi << std::endl 
        //    << x1_plane.getReal().getMatData()(roi) << std::endl;
        x1->unGetPlane(&x1_plane, zid);
        y1->unGetPlane(&y1_plane, zid);
        y0->unGetPlane(&y0_plane, zid);
    }
    
    SAVE_TIMING(SP_INV_FISTAUPDATE_SPVOL);
    CHECK_FOR_ERROR("after SparseCompressiveHolo::fistaUpdate SparseVolume");
}

void quick_print_data2(void* data_d, char* str)
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


void SparseCompressiveHolo::fistaUpdate(Hologram* est, Hologram* prev, Hologram* current, double step)
{
    DECLARE_TIMING(SP_INV_FISTAUPDATE_HOLO);
    START_TIMING(SP_INV_FISTAUPDATE_HOLO);
    CHECK_FOR_ERROR("before SparseCompressiveHolo::fistaUpdate Hologram");
    size_t size = width*height;
    int block_dim = 256;
    int grid_dim = ceil(size / (double)block_dim);
    
    CuMat est_data = est->getData();
    CuMat prev_data = prev->getData();
    CuMat current_data = current->getData();
    
    size_t expected_size = width*height*sizeof(float2);
    bool size_check = true;
    if (!est_data.isAllocated()) est_data.allocateCuData(width, height, 1, sizeof(float2));
    if (!prev_data.isAllocated()) prev_data.allocateCuData(width, height, 1, sizeof(float2));
    if (!current_data.isAllocated()) current_data.allocateCuData(width, height, 1, sizeof(float2));
    if (est_data.getDataSize() != expected_size) size_check = false;
    if (prev_data.getDataSize() != expected_size) size_check = false;
    if (current_data.getDataSize() != expected_size) size_check = false;
    if (!size_check)
    {
        std::cout << "SparseCompressiveHolo::fistaUpdate: input data size not correct" << std::endl;
        throw HOLO_ERROR_INVALID_DATA;
    }
    
    float2* est_d = (float2*)est_data.getCuData();
    float2* prev_d = (float2*)prev_data.getCuData();
    float2* current_d = (float2*)current_data.getCuData();
    
    sparse_fistaUpdate_kernel<<<grid_dim, block_dim>>>
        (est_d, prev_d, current_d, step, size);
    
    est_data.setCuData(est_d);
    prev_data.setCuData(prev_d);
    current_data.setCuData(current_d);
    est->setData(est_data);
    prev->setData(prev_data);
    current->setData(current_data);
    
    SAVE_TIMING(SP_INV_FISTAUPDATE_HOLO);
    CHECK_FOR_ERROR("after SparseCompressiveHolo::fistaUpdate Hologram");
}

void SparseCompressiveHolo::calcResidualFrom(Hologram* residual, Hologram* estimate, ResidualMode mode)
{
    CHECK_FOR_ERROR("begin SparseCompressiveHolo::calcResidual");
    DECLARE_TIMING(SP_INV_CALC_RESIDUAL_FROM);
    START_TIMING(SP_INV_CALC_RESIDUAL_FROM);
    
    CuMat mdata = measured.getData();
    CuMat edata = estimate->getData();
    CuMat rdata = residual->getData();
    
    cv::Rect roi(20,120,10,10);
    //std::cout << "roi: " << roi << std::endl;
    //std::cout << "SparseCompressiveHolo::calcResidualFrom: measured = " << std::endl;
    //std::cout << mdata.getMatData()(roi) << std::endl;
    //std::cout << "SparseCompressiveHolo::calcResidualFrom: estimate = " << std::endl;
    //std::cout << edata.getMatData()(roi) << std::endl;
    
    size_t width = mdata.getWidth();
    size_t height = mdata.getHeight();
    rdata.allocateCuData(width, height, 1, sizeof(float2));
    edata.allocateCuData(width, height, 1, sizeof(float2));
    //mdata.allocateCuData(width, height, 1, sizeof(float2));
    
    float2* mdata_d = (float2*)mdata.getCuData();
    float2* edata_d = (float2*)edata.getCuData();
    float2* rdata_d = (float2*)rdata.getCuData();
    
    size_t numel = width*height;
    size_t dim_block = 256;
    size_t dim_grid = ceil((float)numel / (float)dim_block);
    
    CHECK_FOR_ERROR("before calc_residual_kernel");
    if (mode == RM_TRUE_EST)
    {
        sparse_calc_residual_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else if (mode == RM_EST_TRUE)
    {
        sparse_calc_residual_inverse_kernel<<<dim_grid, dim_block>>>
            (rdata_d, mdata_d, edata_d, numel);
    }
    else
    {
        std::cout << "SparseCompressiveHolo::calcResidual error unknown mode" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    CHECK_FOR_ERROR("after calc_residual_kernel");
    
    rdata.setCuData((void*)rdata_d, height, width, CV_32FC2);
    edata.setCuData((void*)edata_d, height, width, CV_32FC2);
    //mdata.setCuData((void*)mdata_d, height, width, CV_32FC2);
    CHECK_FOR_ERROR("after setCuData");
    
    //std::cout << "SparseCompressiveHolo::calcResidualFrom: residual = " << std::endl;
    //std::cout << rdata.getMatData()(roi) << std::endl;
    
    /*float2* temp_mdata_h = (float2*)malloc(width*height*sizeof(float2));
    float2* temp_edata_h = (float2*)malloc(width*height*sizeof(float2));
    float2* temp_rdata_h = (float2*)malloc(width*height*sizeof(float2));
    cudaMemcpy(temp_mdata_h, mdata_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_edata_h, edata_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_rdata_h, rdata_d, width*height*sizeof(float2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    bool bad_mdata = false;
    bool bad_edata = false;
    bool bad_rdata = false;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            size_t idx = i*width + j;
            if (isnan(temp_mdata_h[idx].x) || isnan(temp_mdata_h[idx].y))
            {
                bad_mdata = true;
                printf("bad mdata at row %d, col %d: %f, %f\n",
                    i, j, temp_mdata_h[idx].x, temp_mdata_h[idx].y);
            }
            if (isnan(temp_edata_h[idx].x) || isnan(temp_edata_h[idx].y)) bad_edata = true;
            if (isnan(temp_rdata_h[idx].x) || isnan(temp_rdata_h[idx].y)) bad_rdata = true;
        }
    }
    printf("Is mdata bad: %d\n", bad_mdata);
    printf("Is edata bad: %d\n", bad_edata);
    printf("Is rdata bad: %d\n", bad_rdata);
    free(temp_mdata_h);
    free(temp_edata_h);
    free(temp_rdata_h);*/
    
    residual->setData(rdata);
    estimate->setData(edata);
    residual->setState(HOLOGRAM_STATE_RESIDUAL);
    CHECK_FOR_ERROR("after setDatas");
    
    mdata.destroy();
    
    CHECK_FOR_ERROR("end SparseCompressiveHolo::calcResidual");
    SAVE_TIMING(SP_INV_CALC_RESIDUAL_FROM);
}
