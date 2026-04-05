#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

// SYSTEM INCLUDES
#include <vector>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"

// LOCAL INCLUDES
#include "cumat.h"

namespace umnholo {

    enum PointCloudType
    {
        C_ZERO_INDEX = 0,
        MATLAB_ONE_INDEX = 1
    };

    /**
     * @brief Important data for a single object
     */
    struct Particle
    {
        float x;
        float y;
        float time;
        float size;
    };

    /**
     * @brief Data holder for extracted particle positions
     */
    class CV_EXPORTS PointCloud
    {
    public:
        // LIFECYCLE

        /**
         * @brief Creates empty PointCloud with zero points
         */
        PointCloud();

        /**
         * @brief Initialize by loading file. Same as calling read after
         *        default constructor
         */
        PointCloud(char* filename, PointCloudType type = MATLAB_ONE_INDEX);

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Read point data from a text file
         * @param filename Name of file to read. File must be an ASCII text
         *                 file with no header and columns time, x, and y.  
         *                 Returns HoloError if filename does not exist.
         */
        void read(char* filename, PointCloudType type = MATLAB_ONE_INDEX);

        /**
         * @brief Write point data to a text file (same format as read)
         * @param filename Name of file to write to. File must be unopened.
         *                 Will overwrite any existing file contents.
         */
        void write(char* filename, PointCloudType type = MATLAB_ONE_INDEX);

        // ACCESS

        /**
         * @brief Returns the number of frames for which data exists
         */
        int getNumFrames();

        int getNumObjects() { return num_objects; }

        /**
         * @brief Returns number of objects in a given frame
         * @param frame_idx Index to the query frame (zero-based)
         * @returns Zero if frame_idx is invalid
         */
        int getCountAtFrame(int frame_idx);

        /**
         * @brief Returns particle specified by frame and particle index
         * @param frame_idx Index to the query frame (zero-based). Should be 
         *                  between 0 and result of getNumFrames
         * @param part_idx Index to query particle. Index is frame-specific
         *                 Should be betwen 0 and result of getCountAtFrame
         */
        Particle getParticle(int frame_idx, int part_idx);

        /**
         * @brief Returns particle at the global array index
         * @param global_idx Unique index to a single particle in the cloud
         */
        Particle getParticle(int global_idx);

        /**
         * @brief Fill cloud one element at a time
         * @param frame_idx Index to the push frame (zero-based). Should be
         *                  non-negative.
         * @param part_idx Index to push particle. Must be non-negative.
         * @param part Particle data to insert into the cloud
         */
        void setParticle(int frame_idx, int part_idx, Particle part);

        /**
        * @brief Returns particle specified by frame and particle index
        * @param frame_idx Index to the query frame (zero-based). Should be
        *                  between 0 and result of getNumFrames
        * @param part_idx Index to query particle. Index is frame-specific
        *                 Should be betwen 0 and result of getCountAtFrame
        */
        cv::Point getPoint(int frame_idx, int part_idx);

        /**
        * @brief Returns particle at the global array index
        * @param global_idx Unique index to a single particle in the cloud
        */
        cv::Point getPoint(int global_idx);

        /**
        * @brief Returns index of the frame containing the query particle
        * @param global_idx Unique index to a single particle in the cloud
        */
        int getFrameIdx(int global_idx);

        /**
        * @brief Returns index of the particle index within its frame
        * @param global_idx Unique index to a single particle in the cloud
        */
        int getPartIdx(int global_idx);

        // INQUIRY

    protected:
    private:
        int num_frames;
        int num_objects;

        std::vector<int> frame_id;
        std::vector<int> frame_starts;
        std::vector<float> time;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> size;
    };

} // namespace umnholo

#endif // POINT_CLOUD_H
