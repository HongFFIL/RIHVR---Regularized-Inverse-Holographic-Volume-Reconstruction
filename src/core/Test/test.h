#ifndef HOLO_TEST_H
#define HOLO_TEST_H

#include <stdio.h>
//#include <tchar.h>
#include <iostream>
#include <string>
#include <stdint.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../cumat.h"
#include "../holo_ui.h"

int test_main(int argc, char* argv[]);

// Utility functions
double getSimilarity(const cv::Mat A, const cv::Mat B);
std::string type2str(int type);
bool compareData(umnholo::CuMat my_data, umnholo::CuMat true_data, double thr = 0.00001, int count_thr = 0, bool verbose = true);
bool compareData(cv::Mat my_data, cv::Mat true_data, double thr = 0.00001, int count_thr = 0, bool verbose = true);
bool compareDataSparse(cv::Mat my_data, cv::Mat true_data, double thr = 0.00001, double count_thr = 0, bool verbose = true);
bool compareFile(umnholo::CuMat my_data, char* filename, double thr = 0.00001, int count_thr = 0, bool verbose = true);
bool compareFileSparse(umnholo::CuMat my_data, char* filename, double thr = 0.00001, double count_thr = 0.0, bool verbose = true);
bool data_exists(umnholo::Parameters params);

// Suite of tests for CuMat class
bool test_cumat();
bool test_cumat_changed_mat();
bool test_cumat_getCuData();
bool test_cumat_unavailable_memory();
bool test_cumat_convertTo();
bool test_cumat_modified_CuData();
bool test_cumat_save_load_binary();
bool test_cumat_save_load_yml();
bool test_cumat_save_load_ascii();
bool test_cumat_load_matlab();
bool test_cumat_loat_matlab_long_col();

// Suite of tests for Hologram class
bool test_hologram();
bool test_hologram_normalize();
bool test_hologram_normalize_mean0();
bool test_hologram_normalize_inverse();
bool test_hologram_convertTo();
bool test_hologram_fft();
bool test_hologram_fft_zeropad();
bool test_hologram_window();
bool test_hologram_lowpass();
bool test_hologram_createGaussianWindow();
bool test_hologram_array_of_holograms();
bool test_hologram_crop();

// Suite of tests for OpticalField class
bool test_optical_field();
bool test_optical_field_recon_plane0();

// Suite of tests for ParticleExtraction class
bool test_particle_extraction();
bool test_particle_extraction_findThreshold2d();
bool test_particle_extraction_average_filter();
bool test_particle_extraction_threshold();
bool test_particle_extraction_threshold_practical();
bool test_particle_extraction_min_max_block_kernel_2d();
bool test_particle_extraction_min_max_block_kernel();
bool test_particle_extraction_normalize_blocks();
bool test_particle_extraction_enhance();
bool test_particle_extraction_binarize();
bool test_particle_extraction_dilate();
bool test_particle_extraction_labelConnectedComponents();

// Suite of tests for Reconstruction class
bool test_reconstruction();
bool test_reconstruction_fsp();
bool test_reconstruction_reconstruct_basic();
bool test_reconstruction_reconstruct_always_positive();
bool test_reconstruction_combinedXY_filled();
bool test_reconstruction_kf_mult_kernel();
bool test_reconstruction_rs_mult_kernel();
bool test_reconstruction_calcFocusMetric();
bool test_reconstruction_focusMetric_kernel();
bool test_reconstruction_bulkFocusMetric_kernel();

// Suite of tests for Deconvolution class
bool test_deconvolution();
bool test_deconvolution_reconstruct();
bool test_deconvolution_multiply_conjugate();
bool test_deconvolution_prescaling();
bool test_deconvolution_deconvolve();
bool test_deconvolution_combinedXY();

// Suite of tests for FocusMetric class
bool test_focus_metric();
bool test_focus_metric_set_raw();

// Suite of tests for PointCloud class
bool test_point_cloud();
bool test_point_cloud_read();
bool test_point_cloud_getCountAtFrame();
bool test_point_cloud_getParticle();

// Suite of tests for HoloSequence class
bool test_holo_sequence();
bool test_holo_sequence_findLargestCount();
bool test_holo_sequence_findLargestCount_roi();
bool test_holo_sequence_init();
bool test_holo_sequence_loadNextChunk();
bool test_holo_sequence_normalize_inverse();
bool test_holo_sequence_window();
bool test_holo_sequence_lowpass();
bool test_holo_sequence_fft();
bool test_holo_sequence_fftshift();

// Suite of tests for ObjectCloud class
bool test_object_cloud();

// Suite of tests for Blob3d class
bool test_blob3d();

// Suite of tests for CompressiveHolo class
bool test_compressive_holo();

// Suite of tests for SparseCompressiveHolo class
bool test_sparse_compressive_holo();

// Optional tests and examples for complete processing codes
// The data for these tests may not be included with the code
int sample_main_fly_processing(int argc, char* argv[]);

// Complete processing test to confirm everything working together
bool test_complete();
bool test_complete_fly_focus();
bool test_complete_deconv_removal();
bool test_complete_deconv_iterations();
bool test_complete_deconv_iterations_1kz();

// Template functions
template<class T>
bool compareVector(std::vector<T> my_vec, std::vector<T> true_vec)
{
    if (my_vec.size() != true_vec.size())
    {
        std::cout << "Failed!" << std::endl;
        std::cout << "Vectors do not have equal numbers of elements" << std::endl;
        std::cout << "my_vec has " << my_vec.size() << " elements" << std::endl;
        std::cout << "true_vec has " << true_vec.size() << " elements" << std::endl;
        return false;
    }
    
    // Test that every element in the true vector is contained in mine
    std::vector<float>::iterator it;
    for (int i = 0; i < true_vec.size(); ++i)
    {
        it = std::find(my_vec.begin(), my_vec.end(), true_vec[i]);
        if (it == my_vec.end()) // Element not found
        {
            std::cout << "Failed!" << std::endl;
            std::cout << "Element " << i << " (" << true_vec[i] << ") missing from mine" << std::endl;
            
            int print_count = (true_vec.size() < 20)? true_vec.size() : 20;
            for (int n = 0; n < print_count; ++n)
            {
                std::cout << n << ": true = " << true_vec[n] << ", mine = " << my_vec[n] << std::endl;
            }
            
            return false;
        }
    }
    
    return true;
}

#endif //HOLO_TEST_H
