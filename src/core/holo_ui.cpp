#include "holo_ui.h"

using namespace umnholo;
using namespace std;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

HoloUi::HoloUi()
{
    file_selected = false;

    params.start_plane = 0;
    params.plane_stepsize = 1.0;
    params.num_planes = 256;
    params.wavelength = 632;
    params.resolution = 1000;
    params.center_point.x = -1;
    params.center_point.y = -1;
    params.center_point.z = -1;
    params.roi_size = -1;
    params.roi = {0,0,0,0};
    params.beta = 0.1;
    params.zero_padding = 256;
    params.window_r = 0.5;
    params.window_alpha = 2.0;
    params.background_start = -1;
    params.background_end = -1;
    params.num_iipe_iterations = 7;
    params.num_inverse_iterations = 500;
    params.num_tv_iterations = 5;
    params.regularization_param = 0.01;
    params.regularization_TV_param = 0.0;
    params.max_svd = 1;
    params.sim_chamber_radius = 16000000;
    params.sim_particle_concentration = 5e-10;
    params.sim_particle_diameter = 7000;
    params.sim_particle_length = 70000;
    params.sim_particle_resolution = 1000;
    params.sim_hologram_image_size = 512;
    params.sim_rotation_step = 0.001;
    params.sim_oversample = 4;
    params.segment_min_area = 2;
    params.segment_min_intensity = 1;
    params.segment_close_size = 3;
    params.remove_background = false;
    params.segment_particles = true;
    params.output_planes = false;
    params.output_steps = false;
    params.use_standard_recon = false;
    params.use_inverse_recon = true;
    params.use_deconvolution = false;
    params.use_median_background = false;
    
    params.plane_list.resize(params.num_planes);
    for (int i = 0; i < params.num_planes; ++i)
    {
        float z = params.start_plane + i * params.plane_stepsize;
        params.plane_list[i] = z;
    }
}// HoloUi

HoloUi::HoloUi(int argc, char* argv[])
{
    file_selected = false;

    params.start_plane = 0;
    params.plane_stepsize = 1.0;
    params.num_planes = 256;
    params.wavelength = 632;
    params.resolution = 1000;
    params.center_point.x = -1;
    params.center_point.y = -1;
    params.center_point.z = -1;
    params.roi_size = -1;
    params.roi = {0,0,0,0};
    params.beta = 0.1;
    params.zero_padding = 256;
    params.window_r = 0.5;
    params.window_alpha = 2.0;
    params.background_start = -1;
    params.background_end = -1;
    params.num_iipe_iterations = 7;
    params.num_inverse_iterations = 500;
    params.num_tv_iterations = 5;
    params.regularization_param = 0.01;
    params.max_svd = 1;
    params.sim_chamber_radius = 16000000;
    params.sim_particle_concentration = 5e-10;
    params.sim_particle_diameter = 7000;
    params.sim_particle_length = 70000;
    params.sim_particle_resolution = 1000;
    params.sim_hologram_image_size = 512;
    params.sim_rotation_step = 0.001;
    params.sim_oversample = 4;
    params.segment_min_area = 2;
    params.segment_min_intensity = 1;
    params.segment_close_size = 3;
    params.remove_background = false;
    params.segment_particles = true;
    params.output_planes = false;
    params.use_standard_recon = false;
    params.use_inverse_recon = true;
    params.use_deconvolution = false;
    params.use_median_background = false;

    if (argc == 2)
    {
        strcpy(input_file, argv[1]);
        file_selected = true;
    }
    
    params.plane_list.resize(params.num_planes);
    for (int i = 0; i < params.num_planes; ++i)
    {
        float z = params.start_plane + i * params.plane_stepsize;
        params.plane_list[i] = z;
    }
}


//============================= OPERATORS ====================================

/*std::ostream& operator<<(std::ostream& os, const Parameters& obj)
{
    // write obj to stream
    os << "Parameters:" << std::endl;
    os << "  image_path: " << obj.image_path << std::endl;
    os << "  image_formatted_name: " << obj.image_formatted_name << std::endl;
    os << "  image_filename: " << obj.image_filename << std::endl;
    os << "  image_count: " << obj.image_count << std::endl;
    os << "  start_image: " << obj.start_image << std::endl;
    os << "  output_path: " << obj.output_path << std::endl;
    os << "  start_plane: " << obj.start_plane << std::endl;
    os << "  plane_stepsize: " << obj.plane_stepsize << std::endl;
    os << "  num_planes: " << obj.num_planes << std::endl;
    os << "  wavelength: " << obj.wavelength << std::endl;
    os << "  resolution: " << obj.resolution << std::endl;
    return os;
}*/

//============================= OPERATIONS ===================================
int HoloUi::run()
{
    if (file_selected)
    {
        readFile(input_file);
        return 0;
    }
    cout << "Use parameter file or text based ui? [1|2]: ";
    int choice;
    cin >> choice;

    if (choice == 2)
    {
        textBasedUi();
        return 0;
    }

    string temp_filename;
    cout << "Enter parameter filename: ";
    cin >> temp_filename;
    strcpy(input_file, temp_filename.c_str());
    file_selected = true;
    readFile(input_file);

    return 0;
}

int HoloUi::readFile(char* filename)
{
    strcpy(input_file, filename);
    file_selected = true;
    
    FILE* fid = fopen(filename, "r");
    if (fid == NULL)
    {
        printf("Unable to read file <%s>\n", filename);
        return -1;
    }

    char buff[FILENAME_MAX];
    char description[FILENAME_MAX];
    HoloParameter id;
    float value;
    char char_value[FILENAME_MAX];
    bool set_full_filename = false;

    while (!feof(fid))
    {
        fgets(buff, FILENAME_MAX, fid);

        // Exclude comment lines
        if ((buff[0] == '%') || (buff[0] == '#')) continue;

        sscanf(buff, "%s %d %f", &description, &id, &value);
        sscanf(buff, "%s %d %[^\t\n]", &description, &id, &char_value);

        switch (id)
        {
        case HOLO_PARAM_NUM_IMAGES:
        {
            params.image_count = value;
            break;
        }
        case HOLO_PARAM_START_IMAGE:
        {
            params.start_image = value;
            break;
        }
        case HOLO_PARAM_BG_START:
        {
            params.background_start = value;
            break;
        }
        case HOLO_PARAM_BG_END:
        {
            params.background_end = value;
            break;
        }
        case HOLO_PARAM_INPUT_FN:
        {
            strcpy(params.image_filename, char_value);
            set_full_filename = true;
            break;
        }
        case HOLO_PARAM_INPUT_PATHN:
        {
            strcpy(params.image_path, char_value);
            break;
        }
        case HOLO_PARAM_INPUT_FORMATFN:
        {
            strcpy(params.image_formatted_name, char_value);
            break;
        }
        case HOLO_PARAM_OUTPUT_PATHN:
        {
            strcpy(params.output_path, char_value);
            break;
        }
        case HOLO_PARAM_INPUT_BACKGROUND:
        {
            strcpy(params.background_filename, char_value);
            params.remove_background = true;
            break;
        }
        case HOLO_PARAM_EXE_PATHN:
        {
            strcpy(params.exe_path, char_value);
            break;
        }
        case HOLO_PARAM_START_PLANE:
        {
            params.start_plane = value;
            break;
        }
        case HOLO_PARAM_PLANE_STEP:
        {
            params.plane_stepsize = value;
            break;
        }
        case HOLO_PARAM_NUM_PLANES:
        {
            params.num_planes = value;
            break;
        }
        case HOLO_PARAM_RESOLUTION:
        {
            params.resolution = value;
            break;
        }
        case HOLO_PARAM_WAVELENGTH:
        {
            params.wavelength = value;
            break;
        }
        case HOLO_PARAM_ROI_X:
        {
            params.center_point.x = value;
            break;
        }
        case HOLO_PARAM_ROI_Y:
        {
            params.center_point.y = value;
            break;
        }
        case HOLO_PARAM_ROI_Z:
        {
            params.center_point.z = value;
            break;
        }
        case HOLO_PARAM_ROI_SIZE:
        {
            params.roi_size = value;
            break;
        }
        case HOLO_PARAM_ZERO_PADDING:
        {
            params.zero_padding = value;
            break;
        }
        case HOLO_PARAM_ROI_RECT_X:
        {
            params.roi.x = value;
            break;
        }
        case HOLO_PARAM_ROI_RECT_Y:
        {
            params.roi.y = value;
            break;
        }
        case HOLO_PARAM_ROI_RECT_W:
        {
            params.roi.width = value;
            break;
        }
        case HOLO_PARAM_ROI_RECT_H:
        {
            params.roi.height = value;
            break;
        }
        case HOLO_PARAM_BETA:
        {
            params.beta = value;
            break;
        }
        case HOLO_PARAM_IIPE_ITERATIONS:
        {
            params.num_iipe_iterations = value;
            break;
        }
        case HOLO_PARAM_PARTICLE_DIAMETER:
        {
            params.particle_diameter = value;
            params.sim_particle_diameter = value;
            break;
        }
        case HOLO_SIM_CHAMBER_RADIUS:
        {
            params.sim_chamber_radius = value;
            break;
        }
        case HOLO_SIM_PARTICLE_CONCENTRATION:
        {
            params.sim_particle_concentration = value;
            break;
        }
        case HOLO_SIM_PARTICLE_LENGTH:
        {
            params.sim_particle_length = value;
            break;
        }
        case HOLO_SIM_PARTICLE_RESOLUTION:
        {
            params.sim_particle_resolution = value;
        }
        case HOLO_SIM_IMAGE_SIZE:
        {
            params.sim_hologram_image_size = value;
            break;
        }
        case HOLO_SIM_ROTATION_STEP:
        {
            params.sim_rotation_step = value;
            break;
        }
        case HOLO_SIM_OVERSAMPLE:
        {
            params.sim_oversample = value;
            break;
        }
        case HOLO_PARAM_SEGMENT_MIN_AREA:
        {
            params.segment_min_area = value;
            break;
        }
        case HOLO_PARAM_SEGMENT_MIN_INTENSITY:
        {
            params.segment_min_intensity = value;
            break;
        }
        case HOLO_PARAM_SEGMENT_CLOSE_SIZE:
        {
            params.segment_close_size = value;
            break;
        }
        case HOLO_PARAM_NUM_INVERSE_ITS:
        {
            params.num_inverse_iterations = value;
            break;
        }
        case HOLO_PARAM_INVERSE_REG_TAU:
        {
            params.regularization_param = value;
            break;
        }
        case HOLO_PARAM_INVERSE_MAX_SVD:
        {
            params.max_svd = value;
            break;
        }
        case HOLO_PARAM_INVERSE_REG_TV:
        {
            params.regularization_TV_param = value;
            break;
        }
        case HOLO_PARAM_NUM_TV_ITS:
        {
            params.num_tv_iterations = value;
            break;
        }
        case HOLO_FLAG_OUTPUT_PLANES:
        {
            params.output_planes = value;
            if (strcmp(char_value, "true") == 0)
                params.output_planes = true;
            if (strcmp(char_value, "false") == 0)
                params.output_planes = false;
            break;
        }
        case HOLO_FLAG_OUTPUT_STEPS:
        {
            params.output_steps = value;
            if (strcmp(char_value, "true") == 0)
                params.output_steps = true;
            if (strcmp(char_value, "false") == 0)
                params.output_steps = false;
            break;
        }
        case HOLO_FLAG_STANDARD_RECON:
        {
            params.use_standard_recon = value;
            if (strcmp(char_value, "true") == 0)
                params.use_standard_recon = true;
            if (strcmp(char_value, "false") == 0)
                params.use_standard_recon = false;
            break;
        }
        case HOLO_FLAG_INVERSE_RECON:
        {
            params.use_inverse_recon = value;
            if (strcmp(char_value, "true") == 0)
                params.use_inverse_recon = true;
            if (strcmp(char_value, "false") == 0)
                params.use_inverse_recon = false;
            break;
        }
        case HOLO_FLAG_DECONV_RECON:
        {
            params.use_deconvolution = value;
            if (strcmp(char_value, "true") == 0)
                params.use_deconvolution = true;
            if (strcmp(char_value, "false") == 0)
                params.use_deconvolution = false;
            break;
        }
        case HOLO_FLAG_MEDIAN_BACKGROUND:
        {
            params.use_median_background = value;
            if (strcmp(char_value, "true") == 0)
                params.use_median_background = true;
            if (strcmp(char_value, "false") == 0)
                params.use_median_background = false;
            break;
        }
        default:
        {
            cerr << "Unknown or unsupported input argument: " << buff << endl;
            return -1;
        }
        }
    }

    if (!set_full_filename) sprintf(params.image_filename, "%s%s", params.image_path, params.image_formatted_name);
    
    params.plane_list.resize(params.num_planes);
    for (int i = 0; i < params.num_planes; ++i)
    {
        float z = params.start_plane + i * params.plane_stepsize;
        params.plane_list[i] = z;
    }

    return 0;
}

void HoloUi::checkGpu()
{
    cudaDeviceProp props;
    size_t avail_mem, total_mem;
    int count = 0;
    cudaGetDeviceCount(&count);
    cout << "Number of available devices: " << count << endl;

    for (int i = 0; i < count; ++i)
    {
        cout << "Device " << i << endl;
        cudaGetDeviceProperties(&props, 0);
        cout << "  description: " << props.name << endl;
        cout << "  shared memory per block: " << props.sharedMemPerBlock << endl;
        cout << "  Max threads X: " << props.maxThreadsDim[0] << endl;
        cout << "              Y: " << props.maxThreadsDim[1] << endl;
        cout << "              Z: " << props.maxThreadsDim[2] << endl;
        cout << "  Max GridSize X: " << props.maxGridSize[0] << endl;
        cout << "               Y: " << props.maxGridSize[1] << endl;
        cout << "               Z: " << props.maxGridSize[2] << endl;

        cudaMemGetInfo(&avail_mem, &total_mem);
        cout << "  total memory: " << (total_mem >> 20) << "MB" << endl;
        cout << "  available memory: " << (avail_mem >> 20) << "MB" << endl;
    }
}

/*struct PassData
{
    Reconstruction* recon;
    Hologram* holo;
};

void updateView(int zidx, void* ptr)
{
    PassData* data = (PassData*)ptr;
    Reconstruction* recon = data->recon;
    Hologram* holo = data->holo;
    Parameters params = getParams();
    int width = recon->getWidth();
    int height = recon->getHeight();
    
    if (recon->isState(OPTICALFIELD_STATE_RECONSTRUCTED))
    {
        CuMat single_plane = recon->getPlane(zidx);
        cv::Mat plane = single_plane.getMatData();
        cv::imshow("Reconstruction", plane);
    }
    else
    {
        CuMat single_plane;
        float z = params.start_plane + zidx*params.plane_stepsize;
        recon->reconstructTo(&single_plane, *holo, z);
        cv::Mat plane = single_plane.getMatData();
        cv::imshow("Reconstruction", plane);
    }
}

void HoloUi::viewReconstruction(Reconstruction& recon, Hologram holo)
{
    cv::namedWindow("Reconstruction", CV_WINDOW_AUTOSIZE);
    
    PassData* data = new PassData;
    data->recon = &recon;
    data->holo =  &holo};
    
    int plane_id = 0;
    cv::createTrackbar("Plane #", "Reconstruction",
        &plane_id, params.num_planes-1, updateView, (void*)data);
    
    updateView(0, (void*)data);
    
    while (true)
    {
        char key = (char)cv::waitKey(1);
        if (key == 27) // 'esc' key was pressed
        {
            cv::destroyAllWindows();
            break;
        }
    }
}*/

//============================= ACCESS     ===================================
Parameters HoloUi::getParameters()
{
    if (!validateParameters())
    {
        cout << "Warning: parameters not set properly or invalid" << endl;
    }

    return params;
}
//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
void HoloUi::textBasedUi()
{
    cout << "Text based UI not implemented. Use parameter file instead." << endl;
}

bool HoloUi::validateParameters()
{
    if (params.image_count < 1) return false;

    return true;
}
