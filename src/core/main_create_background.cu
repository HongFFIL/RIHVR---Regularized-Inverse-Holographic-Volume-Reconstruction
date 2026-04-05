#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#ifdef LINUX_BUILD
#include <getopt.h>
#else
#include "XGetopt.h"
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "umnholo.h"
#include "cumat.h"
#include "holo_ui.h"
#include "hologram.h"
#include "holo_sequence.h"

using namespace umnholo;

// Uses a sorting network to sort 5 elements using 9 compare-and-swap operations
// Source: http://pages.ripco.net/~jgamble/nw.html with Bose-Nelson algorithm
__global__ void median5plane_kernel(float2* med, float2* image_stack, size_t size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        // Load the five planes into local memory
        // Assume all data is real
        float a0 = image_stack[idx + 0*size].x;
        float a1 = image_stack[idx + 1*size].x;
        float a2 = image_stack[idx + 2*size].x;
        float a3 = image_stack[idx + 3*size].x;
        float a4 = image_stack[idx + 4*size].x;
        
        // The comparisons to be performed:
        // [0,1],[3,4],[2,4],[2,3],[1,4],[0,3],[0,2],[1,3],[1,2]
        
        // [0,1]
        float mx = (a0 > a1)? a0 : a1;
        float mn = (a0 < a1)? a0 : a1;
        a0 = mn;
        a1 = mx;
        
        // [3,4]
        mx = (a3 > a4)? a3 : a4;
        mn = (a3 < a4)? a3 : a4;
        a3 = mn;
        a4 = mx;
        
        // [2,4]
        mx = (a2 > a4)? a2 : a4;
        mn = (a2 < a4)? a2 : a4;
        a2 = mn;
        a4 = mx;
        
        // [2,3]
        mx = (a2 > a3)? a2 : a3;
        mn = (a2 < a3)? a2 : a3;
        a2 = mn;
        a3 = mx;
        
        // [1,4]
        mx = (a1 > a4)? a1 : a4;
        mn = (a1 < a4)? a1 : a4;
        a1 = mn;
        a4 = mx;
        
        // [0,3]
        mx = (a0 > a3)? a0 : a3;
        mn = (a0 < a3)? a0 : a3;
        a0 = mn;
        a3 = mx;
        
        // [0,2]
        mx = (a0 > a2)? a0 : a2;
        mn = (a0 < a2)? a0 : a2;
        a0 = mn;
        a2 = mx;
        
        // [1,3]
        mx = (a1 > a3)? a1 : a3;
        mn = (a1 < a3)? a1 : a3;
        a1 = mn;
        a3 = mx;
        
        // [1,2]
        mx = (a1 > a2)? a1 : a2;
        mn = (a1 < a2)? a1 : a2;
        a1 = mn;
        a2 = mx;
        
        // Now that they are sorted, a2 is the median
        med[idx].x = a2;
        med[idx].y = 0;
    }
}

// Median background calculation is a bit of a quick hack
// Only functions with 5 frame window size
void median_background_calc(Parameters params)
{
    // Load the first hologram
    Hologram holo(params);
    char read_filename[FILENAME_MAX];
    sprintf(read_filename, params.image_filename, params.start_image);
    holo.read(read_filename);
    
    // Create space for stack of 5 images
    size_t width = holo.getWidth();
    size_t height = holo.getHeight();
    float2* image_stack = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&image_stack, width*height*5*sizeof(float2)));
    
    size_t num_background_frames = params.background_end - params.background_start;
    if (num_background_frames < 5)
    {
        std::cout << "Median background calculation requires at least 5 frames" << std::endl;
        throw HOLO_ERROR_CRITICAL_ASSUMPTION;
    }
    
    // Load the first 5 images
    for (int i = 0; i < 5; ++i)
    {
        int id = params.background_start + i;
        sprintf(read_filename, params.image_filename, id);
        holo.read(read_filename);
        
        float2* data = (float2*)holo.getData().getCuData();
        float2* plane = image_stack + width*height*i;
        CUDA_SAFE_CALL(cudaMemcpy(plane, data, width*height*sizeof(float2), cudaMemcpyDeviceToDevice));
    }
    
    // Compute median for the first 5 images
    Hologram background(params);
    background.read(read_filename); // Only for easy allocation
    CuMat bg_data = background.getData();
    float2* bg_data_d = (float2*)bg_data.getCuData();
    int block_dim = 256;
    int grid_dim = ceil((double)(width*height) / (double)block_dim);
    median5plane_kernel<<<grid_dim, block_dim>>>(bg_data_d, image_stack, width*height);
    CHECK_FOR_ERROR("after first median5plane_kernel");
    
    // Save that image as the first 3 backgrounds
    bg_data.setCuData(bg_data_d);
    background.setData(bg_data);
    for (int i = 0; i < 3; ++i)
    {
        char write_filename[FILENAME_MAX];
        sprintf(write_filename, "%s/background_%04d.png", params.output_path, params.background_start + i);
        background.write(write_filename);
    }
    
    printf("Computing background for image %04d of %04d\n", 3, num_background_frames);
    for (int i = 3; i < num_background_frames - 2; ++i)
    {
        printf("Computing background for image %04d of %04d\n", i, num_background_frames);
        
        // Read new hologram
        int id = params.background_start + i;
        sprintf(read_filename, params.image_filename, id);
        holo.read(read_filename);
        
        // Overwrite the oldest plane with the newest
        // Will cycle through planes 0-4 in order
        // Stack will not be in order but all planes will be there
        int plane_id = (i-3) % 5;
        float2* data = (float2*)holo.getData().getCuData();
        float2* plane = image_stack + width*height*plane_id;
        CUDA_SAFE_CALL(cudaMemcpy(plane, data, width*height*sizeof(float2), cudaMemcpyDeviceToDevice));
        cudaFree(data);
        
        // Calculate median
        median5plane_kernel<<<grid_dim, block_dim>>>(bg_data_d, image_stack, width*height);
        CHECK_FOR_ERROR("after loop median5plane_kernel");
        
        // Write result
        bg_data.setCuData(bg_data_d);
        background.setData(bg_data);
        char write_filename[FILENAME_MAX];
        sprintf(write_filename, "%s/background_%04d.png", params.output_path, params.background_start + i);
        background.write(write_filename);
    }
    
    background.destroy();
    holo.destroy();
    CHECK_FOR_ERROR("end median_background_calc");
    return;
}

int main(int argc, char* argv[])
{
    printf("Begining background calculation\n");
    CHECK_FOR_ERROR("start");

    char params_filename[FILENAME_MAX];
    bool params_set = false;

	std::cout << "Parsing command line inputs" << std::endl;
    int c;
    while ((c = getopt(argc, argv, "F:")) != -1)
    {
        switch(c)
        {
            case 'F':
            {
                const char* temp_params_file = NULL;
                temp_params_file = optarg;
                strcpy(params_filename, temp_params_file);
                params_set = true;
                break;
            }
            default:
            {
                std::cout << "Unknown input argument" << std::endl;
                return -1;
            }
        }
    }

    if (!params_set)
    {
        std::cout << "User must give input parameter file using -F flag" << std::endl;
        return -1;
    }
    std::cout << "Parameters have been set, begining processing" << std::endl;
    
    HoloUi ui;
    ui.readFile(params_filename);
    Parameters params = ui.getParameters();
    
    if (params.use_median_background)
    {
        median_background_calc(params);
        return 0;
    }
    
    HoloSequence holo(params);
    int max_count = holo.findLargestCount(2);
    holo.init();

    holo.useBackgroundSubtraction();

    char background_filename[FILENAME_MAX];
    sprintf(background_filename, "%s/background.png", params.output_path);
    holo.getBackground().write(background_filename);

    // Also write parameters to text file for later use
    double alpha, beta, minimum, average;
    holo.getBackgroundStats(&alpha, &beta, &minimum, &average);
    char stats_filename[FILENAME_MAX];
    sprintf(stats_filename, "%s/background_stats.txt", params.output_path);
    FILE* stats_file = fopen(stats_filename, "w");
    if (stats_file == NULL)
    {
        std::cout << "Unable to open output file: " << stats_filename << std::endl;
        return -1;
    }

    fprintf(stats_file, "%f\n", alpha);
    fprintf(stats_file, "%f\n", beta);
    fprintf(stats_file, "%f\n", minimum);
    fprintf(stats_file, "%f\n", average);

    fclose(stats_file);

    holo.destroy();

    return 0;
}
