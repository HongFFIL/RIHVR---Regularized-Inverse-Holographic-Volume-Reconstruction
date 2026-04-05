#ifndef POINT_CLOUD_3D_H
#define POINT_CLOUD_3D_H

// SYSTEM INCLUDES
#include <vector>
#include <iostream>
#include <math.h>
//

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"

// LOCAL INCLUDES
#include "cumat.h"
#include "hologram.h"
//

namespace umnholo {
    
    /** 
     * @brief Structure for tracked particle trajectories
    */
    class CV_EXPORTS PointCloud3d
    {
        public:
            // LIFECYCLE
            
            /**
             * @brief Default constructor need not do anything
             */
            PointCloud3d() { return; }
            
            /**
             * @brief Initialize with enough space for count points
             * @param count Number of points in cloud
             */
            PointCloud3d(int count);
            
            /**
             * @brief Initialize positions from matrix
             * @param centroids Matrix of particle centroids. May have 1-3
             *      columns and N rows. Columns correspond to x, y, and z
             *      and are filled in that order.
             */
            PointCloud3d(cv::Mat centroids);
            
            // OPERATORS
            // OPERATIONS
            
            /**
             * @brief Estimates a hologram and compares to the input
             * @param holo Recorded hologram (after background removal)
             *      state should be HOLOGRAM_STATE_NORM_MEAND_ZERO
             * @returns residual Hologram after subtracting the estimated hologram
             */
            void calcResidual(Hologram* residual, Hologram& holo);
            
            /**
             * @brief Optimizes the particle locations to minimize the residual
             *      This is the 'shake' step of shake-the-box
             * @param residual Residual calculated using current positions
             * @returns Modifies internal data
             */
            void optimize(Hologram& residual);
            
            /**
             * @brief Removes particles that are likely to be ghosts
             *      Uses an automatic intensity cutoff
             * @returns Modifies internal data
             */
            void pruneGhosts();
            
            /**
             * @brief Add points from one cloud into this one
             * @param new_cloud PointCloud3d will be added to this one
             * @returns Modifies internal data
             */
            void mergeIn(PointCloud3d& new_cloud);
            
            /**
             * @brief Rounds all point positions to nearest integer
             */
            void round();
            
            // ACCESS
            
            int getNumPoints() { return num_points; }
            
            cv::Point3f getPosition(int idx);
            
            void setPointX(int idx, float x);
            void setPointY(int idx, float y);
            void setPointZ(int idx, float z);
            
            void setPosition(int idx, cv::Point3f pos);
            
            float getIntensity(int idx);
            
            int getId(int idx);
            void setId(int idx, int track_id);
            
            // INQUIRY

        protected:
        private:
            
            int num_points;
            std::vector<int> track_ids;
            std::vector<cv::Point3f> positions;
            std::vector<float> intensities;
            
            CuMat buffer1;
            CuMat buffer2;
            
            /**
             * @brief Add particle to estimated hologram
             * @param pid Index of particle to add, in range 0:num_points-1
             * @returns Updates data in estimated
             */
            void buildEstimatedHologram
                (CuMat* estimated, size_t idx, const Parameters& params);
            
            /**
             * @brief Same as above but with explicit parameter passing
             */
            void buildEstimatedHologram
                (CuMat* estimated, cv::Point3f pos, float intensity,
                float reso, float lambda, size_t nx, size_t ny);
            
            /**
             * @brief Evaluate the norm of the residual hologram to be formed
             * @returns Squared L2 norm of the error
             */
            float evaluatePosition
                (cv::Point3f pos, float intensity, Hologram& holo,
                 Parameters& params);
    };
} // namespace umnholo

#endif // POINT_CLOUD_3D_H