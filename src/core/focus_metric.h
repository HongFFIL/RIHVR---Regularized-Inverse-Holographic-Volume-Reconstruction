#ifndef FOCUS_METRIC_H
#define FOCUS_METRIC_H

// SYSTEM INCLUDES
#include <vector>
#include <iostream>
#include <fstream>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
#include "cuda.h"
#include "npp.h"

// LOCAL INCLUDES
#include "umnholo.h"

namespace umnholo {

    enum FocusMetricState
    {
        FOCUS_STATE_UNKNOWN = -1,
        FOCUS_STATE_EMPTY = 0,
        FOCUS_STATE_RAW = 1
    };

    /**
     * @brief Metric is maximized when object is in focus. This class may
     *        include data conditioning to get best possible peak.
     */
    class CV_EXPORTS FocusMetric
    {
    public:
        // LIFECYCLE

        FocusMetric() { state = FOCUS_STATE_UNKNOWN; }

        /**
         * @brief Allocates space for a vector of size points
         */
        FocusMetric(int size);

        /**
         * @brief Performs same operations as FocusMetric(int size)
         */
        void init(int size);

        // OPERATORS
        // OPERATIONS

        /**
         * @brief Writes focus metric as text file
         * @param filename Destination file
         */
        void write(char* filename);

        /**
         * @brief Writes focus metric as binary file to save space
         * @param filename Destination file
         *        Format of output file is as yet unspecified
         */
        void write_binary(char* filename);

        /**
         * @brief Reads file of the sort output by write_binary
         * @param filename Source file
         *        File must be formated a output by write_binary
         *        The number of elements read is limited by the initialized size
         */
        void read_binary(char* filename);

        // ACCESS

        /**
         * @brief Set the raw metric data using device arrays
         * @param real_d Device allocated array of size num_points
         * @param imag_d Device allocated array of size num_points
         * @returns Changes internal state, device data is not used after call
         */
        void setRawDevice(double* real_d, double* imag_d);

        /**
         * @brief Primarily for debugging purposes only
         */
        double* getRawReal() { return raw_real; }
        double* getRawImag() { return raw_imag; }

        // INQUIRY

    protected:
    private:
        FocusMetricState state;
        int num_points;
        std::vector<float> metric;

        double* raw_real;
        double* raw_imag;
    };

} // namespace umnholo

#endif // FOCUS_METRIC_H
