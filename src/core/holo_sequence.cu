#include "holo_sequence.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

HoloSequence::HoloSequence(const Parameters params)
{
    state = UNINITIALIZED;
    this->params = params;
    store_count = -1;
    image_width = -1;
    image_height = -1;
    image_bytes = -1;
    use_roi = false;
    total_count = params.image_count;
    num_chunks = -1;
    current_chunk = 0;
    use_bg_sub = false;
    bg_scale1 = 5;
    bg_min_grey = 0;
    bg_mean = -1000;
    image_stepsize = 1;
    start_image = params.start_image;
    use_time_as_image_id = false;
    return;
}

void HoloSequence::destroy()
{
    if (state == UNINITIALIZED) return;

    if (store_count > 0)
    {
        for (int n = 0; n < store_count; ++n)
        {
            Hologram temp = holos[n];
            holos[n].destroy();
        }
    }

    return;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

int HoloSequence::findLargestCount(int multiplier)
{
    CHECK_FOR_ERROR("begin HoloSequence::findLargestCount");

    size_t free_memory;
    cudaMemGetInfo(&free_memory, NULL);

    // Determine space used by one hologram (requires reading one)
    char filename[FILENAME_MAX];
    sprintf(filename, params.image_filename, start_image);
    Hologram temp_holo(params);
    temp_holo.read(filename);

    if (use_roi)
    {
        Particle part = points.getParticle(0, 0);
        temp_holo = temp_holo.crop(cv::Point(round(part.x), round(part.y)), roi_size);
    }

    if (image_width == -1) // Indicates unloaded
    {
        image_width = temp_holo.getData().getWidth();
        image_height = temp_holo.getData().getHeight();
        image_bytes = temp_holo.getData().getDataSize();
    }
    else
    {
        size_t loaded_width = temp_holo.getData().getWidth();
        size_t loaded_height = temp_holo.getData().getHeight();
        size_t elem_size = temp_holo.getData().getDataSize() / (loaded_height * loaded_width);
        image_bytes = image_width * image_height * elem_size;
    }

    // Account for some overhead due to memory layout
    size_t max_image_bytes = image_bytes * 1.0002;
    this->store_count = free_memory / (multiplier * max_image_bytes);
    this->num_chunks = ceil((double)total_count / (double)store_count);

    temp_holo.destroy();
    return this->store_count;
}

void HoloSequence::useRoi(int size, PointCloud centers)
{
    use_roi = true;
    roi_size = size;
    points = centers;
    total_count = centers.getNumObjects();
    return;
}

void HoloSequence::init()
{
    image_stepsize = 1;
    start_image = params.start_image;
    if (store_count < 0)
    {
        std::cout << "Must run HoloSequence::findLargestCount before allocating space" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    CUDA_SAFE_CALL( cudaMalloc((void**)&holo_data_d, store_count * image_bytes * sizeof(uint8_t)) );

    //printf("HoloSequence::init creating space for %d holograms\n", store_count);
    holos = new Hologram[store_count];
    for (int n = 0; n < store_count; ++n)
    {
        holos[n].init(params);
        uint8_t* image_d = ((uint8_t*)holo_data_d) + n * image_bytes;
        holos[n].setDevicePtr((void*)image_d, image_bytes);
    }

    state = INITIALIZED;
    CHECK_FOR_ERROR("HoloSequence::init");
    return;
}

bool HoloSequence::loadNextChunk(HoloSequenceProcess norm)
{
    if (this->state != INITIALIZED)
    {
        std::cout << "HoloSequence::loadNextChunk: Must initilize first" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }

    DECLARE_TIMING(read_images);
    START_TIMING(read_images);

    if (current_chunk*image_stepsize >= num_chunks) return false;

    // Chunk size may not evenly line up with number of images to process
    int num_processed = store_count * current_chunk * image_stepsize;
    num_loaded = store_count;
    if (total_count - num_processed < store_count*image_stepsize)
        num_loaded = (total_count - num_processed) / image_stepsize;

    Hologram full_image(params);
    char image_filename[FILENAME_MAX];

    int prev_img_idx = -1;
    for (int n = 0; n < num_loaded; ++n)
    {
        //printf("n = %d of %d\n", n, num_loaded);
        int global_idx = n + store_count * current_chunk;
        //printf("loadNextChunk: global_idx was %d, ", global_idx);
        global_idx *= image_stepsize;
        //printf("now %d\n", global_idx);
        
        // If the full image is different than that used for the previous roi, 
        // load it. Otherwise, reuse the one from before.
        int img_idx = global_idx;
        int absolute_img_num = start_image + img_idx;
        if (use_roi) 
        {
            img_idx = points.getFrameIdx(global_idx);
            if (use_time_as_image_id)
            {
                absolute_img_num = points.getParticle(global_idx).time;
            }
            else
            {
                absolute_img_num = start_image + img_idx;
            }

            if (image_stepsize != 1)
            {
                std::cout << "Warning!: Non-consecutive images not properly supported when using ROI" << std::endl;
                throw HOLO_ERROR_CRITICAL_ASSUMPTION;
            }
        }

        if (img_idx != prev_img_idx)
        {
            sprintf(image_filename, params.image_filename, absolute_img_num);
            //printf("Read image %s\n", image_filename);
            full_image.read(image_filename);

            if (norm == LOAD_BG_DIVISION)
            {
                full_image.divideBackground(background);
            }
            else
            {
                if (use_bg_sub)
                {
                    full_image.subtractBackground(background, bg_alpha, bg_beta, bg_min_grey, bg_mean);
                }
                if (norm == LOAD_NORM_INVERSE)
                {
                    full_image.round();
                    full_image.normalize_inverse();
                }
            }
        }
        prev_img_idx = img_idx;
        //printf("attempting to set holos[%d]\n", n);
        if (use_roi)
        {
            // Place ROI in array. 
            holos[n] = full_image.crop(points.getPoint(global_idx), roi_size);
        }
        else
        {
            holos[n] = full_image;
        }
        //printf("  succeeded\n", n);
    }

    SAVE_TIMING(read_images);

    current_chunk++;
    //printf("finished loadNextChunk\n");

    full_image.destroy();

    return true;
}

bool HoloSequence::reloadChunk(HoloSequenceProcess norm)
{
    if (current_chunk < 1)
    {
        std::cout << "HoloSequence::reloadChunk: must run loadNextChunk first" << std::endl;
        return false;
    }
    current_chunk--;
    return loadNextChunk(norm);
}

void HoloSequence::reset()
{
    current_chunk = 0;
    return;
}

void HoloSequence::view()
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].show();
    }

    return;
}

/*void HoloSequence::normalize_inverse()
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].normalize_inverse();
    }
    return;
}*/

void HoloSequence::window(WindowMethod method, float window_param)
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].window(method, window_param);
    }
    return;
}

void HoloSequence::lowpass(WindowMethod method, float window_param)
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].lowpass(method, window_param);
    }
    return;
}

void HoloSequence::fft(int Nx, int Ny)
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].fft(Nx, Ny);
    }
    return;
}

void HoloSequence::fftshift()
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].fftshift();
    }
    return;
}

void HoloSequence::subtractBackground(Hologram bg, double alpha, double beta, double LL_GS, double bgmean)
{
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].subtractBackground(bg, alpha, beta, LL_GS, bgmean);
    }
    return;
}

void HoloSequence::useBackgroundSubtraction()
{
    // Avoid recalculating stats if it's already been done
    if (use_bg_sub) return;

    CuMat background_data;
    CuMat temp_image;

    // Adjust image size information if roi is used
    int original_store_count = this->store_count;
    int original_use_roi = this->use_roi;
    size_t original_image_width = this->image_width;
    size_t original_image_height = this->image_height;
    size_t original_image_bytes = this->image_bytes;
    int original_start_image = this->start_image;
    int original_total_count = this->total_count;
    int original_num_chunks = this->num_chunks;
    int original_stepsize = this->image_stepsize;
    if (use_roi)
    {
       // Find the raw image width
       this->use_roi = false;
       this->image_width = -1;
       this->image_height = -1;
       this->store_count = this->findLargestCount(2);
    }

    // Allow for different sets of images to be used for background calculation
    // and full processing
    if (params.background_start >= 0 && params.background_end >= 0)
    {
        this->start_image = params.background_start;
        this->total_count = params.background_end - params.background_start - 1;
        this->num_chunks = ceil((double)total_count / (double)store_count);
    }
    else if (original_use_roi)
    {
        this->start_image = points.getFrameIdx(0);
        if (use_time_as_image_id)
            this->start_image = points.getParticle(0).time;
        this->total_count = points.getNumFrames();
        this->num_chunks = ceil((double)total_count / (double)store_count);
    }

    // If there is a large number of images to process, only need to process
    // some to get background summary statistics.
    if (total_count > 500)
    {
        // We want around 200 images for processing
        this->image_stepsize = total_count / 200;
    }

    // Calculate background as mean of all images
    //printf("useBackgroundSubtraction: calculating mean, stepsize = %d\n", image_stepsize);
    this->reset();
    int num_processed = 0;
    cv::Mat bg_mat(cv::Size(image_width, image_height), CV_32FC2, cv::Scalar::all(0));
    background_data.setMatData(bg_mat);
    while (this->loadNextChunk(LOAD_UNCHANGED))
    {
        for (int i = 0; i < num_loaded; ++i)
        {
            //printf("  getHologram(%d)\n", i);
            temp_image = this->getHologram(i).getData();
            //printf("  after getHologram(%d)\n", i);
            sum(background_data, temp_image, background_data);
            num_processed++;
            //printf("  bgsub: processed %d images\n", num_processed);
        }
    }
    std::cout << "Used " << num_processed << " images for background calculation" << std::endl;
    std::cout << "  stepsize = " << this->image_stepsize << std::endl;
    int num_bg_images = num_processed;

    bg_mat = background_data.getMatData();
    bg_mat /= num_processed;
    bg_mean = cv::mean(bg_mat)[0];
    background_data.setMatData(bg_mat);

    // Determine mean and standard deviation of each subtracted image
    this->reset();
    num_processed = 0;
    cv::Mat img;
    cv::Mat all_means(cv::Size(1, num_bg_images), CV_64F, cv::Scalar::all(0));
    cv::Mat all_stds(cv::Size(1, num_bg_images), CV_64F, cv::Scalar::all(0));
    while (this->loadNextChunk(LOAD_UNCHANGED))
    {
        for (int i = 0; i < num_loaded; ++i)
        {
            img = this->getHologram(i).getData().getMatData();
            img = img - bg_mat;
            cv::Scalar new_mean, new_std;
            cv::meanStdDev(img, new_mean, new_std);

            all_means.at<double>(num_processed) = new_mean[0];
            all_stds.at<double>(num_processed) = new_std[0];

            num_processed++;
        }
    }
    
    cv::Scalar mean_mean = mean(all_means);
    cv::Scalar mean_std = mean(all_stds);

    // Calulate and store summary statistics
    bg_alpha = mean_mean[0] - bg_scale1*mean_std[0];
    bg_beta = 255 / (2 * mean_std[0] * bg_scale1);
    //printf("background alpha = %f, beta = %f\n", bg_alpha, bg_beta);

    background.setData(background_data);
    background.setScale(this->getHologram(0).getScale());
    background.setState(this->getHologram(0).getState());

    this->reset();
    use_bg_sub = true;
    this->image_stepsize = original_stepsize;
    this->total_count = original_total_count;
    this->start_image = original_start_image;
    this->num_chunks = original_num_chunks;

    if (original_use_roi)
    {
       // Find the raw image width
       this->use_roi = original_use_roi;
       this->image_width = original_image_width;
       this->image_height = original_image_height;
       this->image_bytes = original_image_bytes;
       this->store_count = original_store_count;
    }

    return;
}

//============================= ACCESS     ===================================

void* HoloSequence::getCuData()
{
    // First, make sure that device data is up-to-date
    for (int n = 0; n < num_loaded; ++n)
    {
        holos[n].getData().getCuData();
    }

    return holo_data_d;
}

void HoloSequence::setSize(size_t width, size_t height)
{
    image_width = width;
    image_height = height;
    return;
}

Hologram HoloSequence::getBackground()
{
    if (!use_bg_sub)
    {
        std::cout << "HoloSequence::getBackground: No background set" << std::endl;
        background = this->getHologram(0);
    }
    
    return background;
}

void HoloSequence::getBackgroundStats
        (double* alpha, double* beta, double* minimum, double* average)
{
    *alpha = bg_alpha;
    *beta = bg_beta;
    *minimum = bg_min_grey;
    *average = bg_mean;
    return;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
