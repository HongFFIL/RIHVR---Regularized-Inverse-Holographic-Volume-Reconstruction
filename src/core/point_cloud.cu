#include "point_cloud.h"  // class implemented

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

PointCloud::PointCloud()
{
    num_frames = 0;
    num_objects = 0;
    return;
}

PointCloud::PointCloud(char* filename, PointCloudType type)
{
    read(filename, type);
    return;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

void PointCloud::read(char* filename, PointCloudType type)
{
    // Handle difference between C and Matlab indexing
    if ((type != C_ZERO_INDEX) && (type != MATLAB_ONE_INDEX))
    {
        std::cout << "PointCloud::read: invalid argument type" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    int shift = 0;
    if (type == MATLAB_ONE_INDEX)
        shift = 1;
    
    CuMat data_cumat;
    try { data_cumat.load(filename, FILEMODE_ASCII); }
    catch (HoloError err)
    {
        if (err == HOLO_ERROR_BAD_FILENAME)
            throw err;
        else throw HOLO_ERROR_UNKNOWN_ERROR;
    }
    cv::Mat data = data_cumat.getMatData();

    num_objects = data.rows;
    num_frames = 1;
    frame_starts.push_back(0);

    frame_id.resize(num_objects);
    time.resize(num_objects);
    x.resize(num_objects);
    y.resize(num_objects);
    size.resize(num_objects);

    // Count the number of unique times
    float prev_time = data.at<float>(0, 0);
    for (int n = 0; n < data.rows; ++n)
    {
        float current = data.at<float>(n, 0);
        if (current != prev_time)
        {
            if (current < prev_time) throw HOLO_ERROR_INVALID_ARGUMENT;

            num_frames++;
            frame_starts.push_back(n);
        }
        prev_time = current;

        frame_id[n] = num_frames - 1;
        time[n] = data.at<float>(n, 0);
        x[n] = data.at<float>(n, 1) - shift;
        y[n] = data.at<float>(n, 2) - shift;
    }

    data.deallocate();
    data_cumat.destroy();
    return;
}

void PointCloud::write(char* filename, PointCloudType type)
{
    // Handle difference between C and Matlab indexing
    if ((type != C_ZERO_INDEX) && (type != MATLAB_ONE_INDEX))
    {
        std::cout << "PointCloud::read: invalid argument type" << std::endl;
        throw HOLO_ERROR_INVALID_ARGUMENT;
    }
    int shift = 0;
    if (type == MATLAB_ONE_INDEX)
        shift = 1;
    
    int num_vars = 4;
    cv::Mat data(num_objects, num_vars, CV_32F);

	printf("PointCloud::write: writing %d objects to %s\n", num_objects, filename);
	std::cout << "Data matrix has size: " << data.size() << std::endl;
    for (int n = 0; n < num_objects; ++n)
    {
        Particle part = this->getParticle(n);
        data.at<float>(n, 0) = part.time;
        data.at<float>(n, 1) = part.x;
        data.at<float>(n, 2) = part.y;
        data.at<float>(n, 3) = part.size;
    }

    CuMat data_cumat;
    data_cumat.setMatData(data);
    bool limit_length = false;
    data_cumat.save(filename, FILEMODE_ASCII, limit_length);

    return;
}

//============================= ACCESS     ===================================

int PointCloud::getNumFrames()
{
    return num_frames;
}

int PointCloud::getCountAtFrame(int frame_idx)
{
    int count = 0;
    if ((frame_idx < 0) || (frame_idx >= num_frames))
        count = 0;
    else if (frame_idx < num_frames-1)
        count = frame_starts[frame_idx + 1] - frame_starts[frame_idx];
    else
        count = num_objects - frame_starts[frame_idx];
    
    return count;
}

Particle PointCloud::getParticle(int frame_idx, int part_idx)
{
    Particle part;

    if (part_idx > getCountAtFrame(frame_idx)-1)
    {
        part.x = 0;
        part.y = 0;
        part.time = 0;
        part.size = 0;
    }

    int idx = frame_starts[frame_idx] + part_idx;

    part.x = x[idx];
    part.y = y[idx];
    part.time = time[idx];
    part.size = size[idx];

    return part;
}

Particle PointCloud::getParticle(int global_idx)
{
    Particle part;
    if (global_idx > num_objects)
    {
        part.x = 0;
        part.y = 0;
        part.time = 0;
        part.size = 0;
    }

    part.x = x[global_idx];
    part.y = y[global_idx];
    part.time = time[global_idx];
    part.size = size[global_idx];

    return part;
}

void PointCloud::setParticle(int frame_idx, int part_idx, Particle part)
{
    // Expand vectors if necessary
    if (frame_idx >= num_frames) // Need to add new frame
    {
        std::vector<int>::iterator it = frame_starts.end();
        frame_starts.insert(it, num_frames-frame_idx+1, num_objects);
        num_frames = frame_idx+1;
    }
    if (part_idx >= getCountAtFrame(frame_idx)) // Add part in frame
    {
        num_objects++;
        for (int i = frame_idx+1; i < num_frames; ++i)
            frame_starts[i]++;
        
        frame_id.resize(num_objects);
        time.resize(num_objects);
        x.resize(num_objects);
        y.resize(num_objects);
        size.resize(num_objects);
    }

    int global_idx = frame_starts[frame_idx] + part_idx;
    frame_id[global_idx] = frame_idx;
    time[global_idx] = part.time;
    x[global_idx] = part.x;
    y[global_idx] = part.y;
    size[global_idx] = part.size;
}

cv::Point PointCloud::getPoint(int frame_idx, int part_idx)
{
    cv::Point pt;

    if (part_idx > getCountAtFrame(frame_idx)-1)
    {
        pt.x = 0;
        pt.y = 0;
    }

    int idx = frame_starts[frame_idx] + part_idx;

    pt.x = round(x[idx]);
    pt.y = round(y[idx]);

    return pt;
}

cv::Point PointCloud::getPoint(int global_idx)
{
    cv::Point pt;

    if (global_idx > num_objects)
    {
        pt.x = 0;
        pt.y = 0;
    }

    pt.x = round(x[global_idx]);
    pt.y = round(y[global_idx]);

    return pt;
}

int PointCloud::getFrameIdx(int global_idx)
{
    if (global_idx > num_objects)
        return 0;

    return frame_id[global_idx];
}

int PointCloud::getPartIdx(int global_idx)
{
    if (global_idx > num_objects)
        return 0;

    int fid = frame_id[global_idx];
    return global_idx - frame_starts[fid];
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
