#ifndef HOLO_UI_H
#define HOLO_UI_H

// SYSTEM INCLUDES
#include <stdio.h>
#include <string.h>
#include <iostream>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"

// LOCAL INCLUDES
//#include "parameters.h"
//#include "reconstruction.h"
//#include "hologram.h"
//#include "cumat.h"


namespace umnholo {
    enum HoloParameter {
        HOLO_PARAM_NUM_IMAGES = 1001,
        HOLO_PARAM_START_IMAGE = 1002,
        HOLO_PARAM_BG_START = 1003,
        HOLO_PARAM_BG_END = 1004,

        HOLO_PARAM_INPUT_FN = 2001,
        HOLO_PARAM_INPUT_PATHN = 2002,
        HOLO_PARAM_INPUT_FORMATFN = 2003,
        HOLO_PARAM_OUTPUT_PATHN = 2004,
        HOLO_PARAM_INPUT_BACKGROUND = 2005,
        HOLO_PARAM_EXE_PATHN = 2006,

        HOLO_PARAM_START_PLANE = 3001,
        HOLO_PARAM_PLANE_STEP = 3002,
        HOLO_PARAM_NUM_PLANES = 3003,
        HOLO_PARAM_WAVELENGTH = 3004,
        HOLO_PARAM_RESOLUTION = 3005,
        HOLO_PARAM_ROI_X = 3006,
        HOLO_PARAM_ROI_Y = 3007,
        HOLO_PARAM_ROI_Z = 3008,
        HOLO_PARAM_ROI_SIZE = 3009,
        HOLO_PARAM_ZERO_PADDING = 3010,
        HOLO_PARAM_ROI_RECT_X = 3011,
        HOLO_PARAM_ROI_RECT_Y = 3012,
        HOLO_PARAM_ROI_RECT_W = 3013,
        HOLO_PARAM_ROI_RECT_H = 3014,

        HOLO_PARAM_BETA = 4001,
        HOLO_PARAM_IIPE_ITERATIONS = 4002,

        HOLO_PARAM_PARTICLE_DIAMETER = 5001,
        HOLO_SIM_CHAMBER_RADIUS = 5002,
        HOLO_SIM_PARTICLE_CONCENTRATION = 5003,
        HOLO_SIM_PARTICLE_LENGTH = 5004,
        HOLO_SIM_PARTICLE_RESOLUTION = 5005,
        HOLO_SIM_IMAGE_SIZE = 5006,
        HOLO_SIM_ROTATION_STEP = 5007,
        HOLO_SIM_OVERSAMPLE = 5008,
        
        HOLO_PARAM_SEGMENT_MIN_AREA = 5009,
        HOLO_PARAM_SEGMENT_MIN_INTENSITY = 5010,
        HOLO_PARAM_SEGMENT_CLOSE_SIZE = 5011,
        
        HOLO_PARAM_NUM_INVERSE_ITS = 6001,
        HOLO_PARAM_INVERSE_REG_TAU = 6002,
        HOLO_PARAM_INVERSE_MAX_SVD = 6003,
        HOLO_PARAM_INVERSE_REG_TV  = 6004,
        HOLO_PARAM_NUM_TV_ITS = 6005,
        
        HOLO_FLAG_OUTPUT_PLANES = 7001,
        HOLO_FLAG_OUTPUT_STEPS = 7005,
        HOLO_FLAG_STANDARD_RECON = 7002,
        HOLO_FLAG_INVERSE_RECON = 7003,
        HOLO_FLAG_DECONV_RECON = 7004,
        HOLO_FLAG_MEDIAN_BACKGROUND = 7006
    };

    /**
     * @brief Execution parameters data structure
     */
    struct CV_EXPORTS Parameters
    {
        char image_path[FILENAME_MAX];
        char image_formatted_name[FILENAME_MAX];
        char image_filename[FILENAME_MAX];
        char background_filename[FILENAME_MAX];
        int image_count;
        int start_image;
        int background_start;
        int background_end;

        char output_path[FILENAME_MAX];
        char exe_path[FILENAME_MAX];

        float start_plane;
        float plane_stepsize;
        int num_planes;
        std::vector<float> plane_list;

        float wavelength;
        float resolution;
        
        cv::Point3f center_point;
        int roi_size;
        cv::Rect roi;

        float beta;
        int num_iipe_iterations;

        float particle_diameter;

        int zero_padding;
        float window_r;
        float window_alpha;
        
        int num_inverse_iterations;
        int num_tv_iterations;
        float regularization_param;
        float regularization_TV_param;
        float max_svd;
        
        double sim_chamber_radius;
        double sim_particle_concentration;
        double sim_particle_diameter;
        double sim_particle_length;
        double sim_particle_resolution;
        int sim_hologram_image_size;
        double sim_rotation_step;
        int sim_oversample;
        
        int segment_min_area;
        double segment_min_intensity;
        int segment_close_size;
        
        bool remove_background;
        bool segment_particles;
        bool output_planes;
        bool output_steps;
        bool use_standard_recon;
        bool use_inverse_recon;
        bool use_deconvolution;
        bool use_median_background;
    };

    //std::ostream& operator<<(std::ostream& os, const Parameters& obj);

    /**
     * @brief User interface for setting execution parameters
     */
    class CV_EXPORTS HoloUi
    {
    public:
        // LIFECYCLE

        /** 
         * @brief Default constructor
         */
        HoloUi();

        /**
         * @brief Construtor using command line arguments
         * @param argc Number of command line arguments (standard)
         * @param argv Text array of command line arguments (standard)
         */
        HoloUi(int argc, char* argv[]);

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Perform user query tasks needed to get necessary information
         */
        int run();

        /**
         *@brief Read information from formatted text file
         *@param filename Full path filename for parameter text file
         */
        int readFile(char* filename);

        /**
         * @brief Check properties of GPU
         */
        void checkGpu();
        
        /**
         * @brief Opens a window for the user to visualize the reconstruction
         * @param recon Initialized Reconstruction object
         */
        //void viewReconstruction(Reconstruction& recon, Hologram holo);

        // ACCESS

        /**
         *@brief Processing parameters
         */
        Parameters getParameters();

        // INQUIRY

    protected:
    private:
        char input_file[FILENAME_MAX];
        Parameters params;
        bool file_selected;

        void textBasedUi();
        bool validateParameters();
    };
} // namespace umnholo

#endif // FLY_UI_H
