#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/stat.h>
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
#include "optical_field.h"
#include "reconstruction.h"
#include "deconvolution.h"
#include "particle_extraction.h"
#include "point_cloud.h"
#include "focus_metric.h"
#include "holo_sequence.h"
#include "compressive_holo.h"
#include "Test/test.h"
#include "sparse_compressive_holo.h"
#include "sparse_segmentation.h"

using namespace umnholo;

// Compressive inverse holography for improved reconstruction quality
int main(int argc, char* argv[])
{
    try
    {
    std::cout << "Begining sparse inverse reconstruction" << std::endl;
    CHECK_FOR_ERROR("start");
    OPEN_TIMING_FILE();
    
    clock_t start = clock();

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
    
    CHECK_MEMORY("start");
    
    // Read parameters and load hologram for initializing volume
    
    HoloUi ui;
    ui.readFile(params_filename);
    Parameters params = ui.getParameters();
    
    Hologram holo(params);
    char read_filename[FILENAME_MAX];
    sprintf(read_filename, params.image_filename, params.start_image);
    
    holo.read(read_filename);
    
    holo = holo.crop();
    
    holo = holo.zeroPad(params.zero_padding);
    
    SparseVolume volume;
    volume.initialize(holo);
    
    // Assume common format name
    char filename[FILENAME_MAX];
    sprintf(filename, "%s/inverse_voxels_%04d.csv",
        params.output_path, params.start_image);
    volume.loadData(filename);
    
    SparseSegmentation segment(volume);
    segment.binarize();
    segment.close();
    printf("labeling connected components...\n");
    segment.labelConnectedComponents();
    
    printf("extracting and saving object centroids\n");
    ObjectCloud particles = segment.extractObjects();
    particles.blobSizeFilter(params.segment_min_area);
    
    sprintf(filename, "%s/centroids_%04d.csv",
        params.output_path, params.start_image);
    particles.writeCentroids(filename);
    sprintf(filename, "%s/centroids_weighted_%04d.csv",
        params.output_path, params.start_image);
    particles.writeCentroids(filename, true);
    
    sprintf(filename, "%s/centroids_verbose_%04d.csv",
        params.output_path, params.start_image);
    particles.writeCentroids(filename, false, true);
    
    clock_t stop = clock();
    std::cout << "Done. Total time: "
        << (float)(stop - start) / (CLOCKS_PER_SEC / 1000)
        << " ms" << std::endl;
    
    return 0;
    }
    catch (HoloError err)
    {
        std::cout << "Detected HoloError: " << err << std::endl;
        return -1;
    }
}