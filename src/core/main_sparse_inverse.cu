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

#ifdef _WIN32
  #include <direct.h>  // for _mkdir
#else
  #include <sys/stat.h>  // for mkdir
#endif


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

        // New flag to control saving planes in subfolders by image number
        bool save_planes_in_subfolder = false;

        std::cout << "Parsing command line inputs" << std::endl;
        int c;
        while ((c = getopt(argc, argv, "F:S:")) != -1)  // Added 'S:' to getopt flags
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
                case 'S':
                {
                    // Expect "1" to enable saving planes in subfolders, else disabled
                    save_planes_in_subfolder = (atoi(optarg) != 0);
                    std::cout << "Save planes in subfolder flag set to " << save_planes_in_subfolder << std::endl;
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

        HoloUi ui;
        ui.readFile(params_filename);
        Parameters params = ui.getParameters();

        Hologram holo(params);
        char read_filename[FILENAME_MAX];
        sprintf(read_filename, params.image_filename, params.start_image);
        holo.read(read_filename);

        Hologram bg(params);
        if (params.remove_background)
        {
            std::cout << "Removing background" << std::endl;
            bg.read(params.background_filename);
            holo.rescale(HOLOGRAM_SCALE_0_1);
            bg.rescale(HOLOGRAM_SCALE_0_1);
            holo.removeBackground(bg);
        }
        else
        {
            std::cout << "Rescaling hologram" << std::endl;
            holo.rescale(HOLOGRAM_SCALE_0_1);
            holo.normalize_mean0();
        }

        holo = holo.crop();

        holo = holo.zeroPad(params.zero_padding);

        char enh_filename[FILENAME_MAX];
        sprintf(enh_filename, "%s/enh.tif", params.output_path);
        holo.write(enh_filename);

        SparseCompressiveHolo recon(holo);

        CHECK_MEMORY("after allocations");

        std::cout << "Sizeof(size_t) = " << sizeof(size_t) << std::endl;
        std::cout << "Sizeof(float2) = " << sizeof(float2) << std::endl;

        printf("before recon.inverseReconstruct\n");
        recon.inverseReconstruct(holo, COMPRESSIVE_MODE_FASTA_FUSED_LASSO_2D);
        printf("after recon.inverseReconstruct\n");

        CHECK_MEMORY("after iterative reconstruction");
		
		// Save reconstructed slices to file for visualization
		if (params.output_planes)
		{
			if (save_planes_in_subfolder)
			{
				// Construct subfolder path: output_path/<image_number>/
				char subfolder_path[FILENAME_MAX];
				sprintf(subfolder_path, "%s/%04d/", params.output_path, params.start_image);

				// Create the subfolder if it doesn't exist
				#ifdef _WIN32
					_mkdir(subfolder_path);
				#else
					mkdir(subfolder_path, 0755);
				#endif

				// Construct full prefix including subfolder and prefix string "inverse"
				char prefix_with_subfolder[FILENAME_MAX];
				sprintf(prefix_with_subfolder, "%s%s", subfolder_path, "inverse");

				// Call savePlanes with the full path + prefix
				recon.savePlanes(prefix_with_subfolder);
			}
			else
			{
				// Always prepend output_path to avoid silent mis-save
				char prefix_with_path[FILENAME_MAX];
				sprintf(prefix_with_path, "%s/%s", params.output_path, "inverse");

				recon.savePlanes(prefix_with_path);
			}
		}

        // Save projections and sparse particles as before (unchanged)
        recon.saveProjections("cmb");
        recon.saveSparse("inverse_voxels");

        holo.destroy();
        bg.destroy();
        recon.destroy();

        CHECK_MEMORY("end");

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

