#ifndef CUMAT_H
#define CUMAT_H

// SYSTEM INCLUDES
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

// PROJECT INCLUDES
#include "opencv2/core/core.hpp"
//#include "opencv2/core/traits.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nppi.h"

// LOCAL INCLUDES
#include "umnholo.h"

namespace umnholo {

    enum CuMatFileMode
    {
        FILEMODE_BINARY = 0,
        FILEMODE_ASCII = 1,
        FILEMODE_YML = 2,
        FILEMODE_CSV = 3
    };

    const int MAX_SAVED_ROWS = 1024;
    const int SAVED_NUM_CHARS = 16;

    /** 
     * @brief Utility class for converting from OpenCV Mat data to CUDA device
     *        variables. For use in high level functions or classes to 
     *        simplify conversion to CUDA code
     */
    class CV_EXPORTS CuMat
    {
    public:
        size_t available_device_mem;
        // LIFECYCLE

        /**
         * @brief By defualt, all data is empty and unallocated
         */
        CuMat();

        /**
         * @brief Destructor must be called manually to return device data
         */
        void destroy();

        // OPERATORS

        CuMat operator=(CuMat rhs);

        // OPERATIONS

        /**
         * @brief Converts type of data using OpenCV type names
         * @param new_type OpenCV type to convert to
         * @param duplicate When output has more channels than input, indicates
         *        whether additional channels should be copies of the original
         *        (true) or zero (false, default)
         * @returns Modifies data and type value. Destroys CUDA device data if
         *          existing. Warns user if that happens.
         */
        void convertTo(int new_type, bool duplicate = false);

        /**
         * @brief Allocates space on gpu device for matrix data
         *        Uses internal size and type information for allocation
         */
        void allocateCuData();

        /**
         * @brief Allocates space on gpu for a matrix of the requested size
         * @param width Number of data elements in width
         * @param height Number of data elements in height
         * @param depth Number of data elements in depth
         * @param elem_size Size in bytes of each element as returned by sizeof
         */
        void allocateCuData(size_t width, size_t height, size_t depth, size_t elem_size);

        /**
         * @brief Converts complex array to real by removing complex component
         *        Reallocates device data, may be inefficient
         *        New value is the real component of the original data
         * @requires Data type must be CV_32FC
         */
        void makeReal();

        /**
         * @brief Saves the data to a file
         * @param filename Full or relative path to storage file. If a file 
         *                 with name exists, it will be overwritten. If the
         *                 path does not exist, will throw HOLO_ERROR
         */
        void save(char* filename, CuMatFileMode format = FILEMODE_YML, bool limit_size = false);

        /**
         * @brief Loads data from a binary file
         * @param filename Full or relative path to storage file. If the file
         *                 does not exist, will throw HOLO_ERROR
         * @param format Specifies format of the file
         *        FILEMODE_BINARY Saves data in a binary file using fwrite
         *        FILEMODE_ASCII Saves data in a space delimited ASCII file
         *        FILEMODE_CSV Uses standard comma separated value format
         */
        void load(char* filename, CuMatFileMode format = FILEMODE_ASCII);

        /**
         * @brief Apply threshold to data, mimicing behavior of OpenCV function
         * @param thresh Threshold value
         * @param type See documentation for cv::threshold for possible options
         *             Currently, only THRESH_TOZERO, THRESH_BINARY, and
         *             THRESH_BINARY_INV are implemented
         * @param set_value Value used for non-zero output of binary threshold
         */
        void threshold(double thresh, int type, float set_value = 0.0);
        
        /**
         * @brief Simulates result of saving and reloading image
         * @returns Data is rounded to nearest multiple of 1/255
         */
        void simulateUint8Cast();
        
        /**
         * @brief Simulates result of matlab dilation with strel('disk', radius)
         * @param radius Radius of structuring element disk. Total width 
         *               of structuring element is 2*radius + 1
         * @returns Performs operation in-place
         */
        void dilateDisk(int radius);

        /**
         * @brief Morphological opening with arbitrary structuring element
         * @param opening_element Binary data describing neighborhood used for
         *        opening. Can be arbitrary size and shape.
         * @param buffer Buffer data for operation, must be of same size as this
         * @returns Performs operation in-place
         */
        void morphOpening(CuMat opening_element, CuMat buffer);
        
        /**
         * @brief Replace the masked region with the mean of the array
         * @param mask Operation mask of the same size as *this
         */
        void maskMean(CuMat mask);
        
        /**
         * @brief Replace the entire region with the mean of the array
         */
        void setToMean();
        
        /**
         * @brief Returns the smallest and largest values in the array
         * @param T Template class must be of same size as stored data type
         * @param min_d and max_d are pointers to device data to store the
         *        min and max results respectively
         */
        template<class T>
        void minMax(T* min_d, T* max_d)
        {
            CHECK_FOR_ERROR("before CuMat::minMax");
            // Check that class matches stored data
            if (sizeof(T) != elem_size)
            {
                std::cout << "CuMat::minMax Error: Requested class must matched stored data" << std::endl;
                throw HOLO_ERROR_INVALID_ARGUMENT;
            }
            
            NppStatus err1, err2;
            
            // Compute min and max using NPPI functions
            //NppiSize roi = {cols, rows*depth};
            NppiSize roi = {
                            static_cast<int>(cols),
                            static_cast<int>(rows * depth)
                            };
            Npp8u* buffer;
            //int buffer_size_h = 0;
            size_t buffer_size_h = 0;
            T* src = (T*)this->getCuData();
            int src_step = cols*sizeof(T);
            
            switch (sizeof(T))
            {
                case 1:
                {
                    err1 = nppiMinMaxGetBufferHostSize_8u_C1R(roi, &buffer_size_h);
                    CUDA_SAFE_CALL( cudaMalloc((void**)&buffer, buffer_size_h) );
                    err2 = nppiMinMax_8u_C1R((Npp8u*)src, src_step, roi, (Npp8u*)min_d, (Npp8u*)max_d, buffer);
                    break;
                }
                case 2:
                {
                    err1 = nppiMinMaxGetBufferHostSize_16s_C1R(roi, &buffer_size_h);
                    CUDA_SAFE_CALL( cudaMalloc((void**)&buffer, buffer_size_h) );
                    err2 = nppiMinMax_16s_C1R((Npp16s*)src, src_step, roi, (Npp16s*)min_d, (Npp16s*)max_d, buffer);
                    break;
                }
                case 4:
                {
                    err1 = nppiMinMaxGetBufferHostSize_32f_C1R(roi, &buffer_size_h);
                    CUDA_SAFE_CALL( cudaMalloc((void**)&buffer, buffer_size_h) );
                    err2 = nppiMinMax_32f_C1R((Npp32f*)src, src_step, roi, (Npp32f*)min_d, (Npp32f*)max_d, buffer);
                    break;
                }
                default:
                {
                    std::cout << "CuMat::minMax data of size " << sizeof(T) << " is not supported" << std::endl;
                    throw HOLO_ERROR_INVALID_ARGUMENT;
                    break;
                }
            }
            
            if ((err1 != NPP_SUCCESS) || (err2 != NPP_SUCCESS))
            {
                std::cout << "CuMat::minMax Error Nppi functions with error codes";
                std::cout << err1 << " and " << err2 << std::endl;
                throw HOLO_ERROR_UNKNOWN_ERROR;
            }
            
            CHECK_FOR_ERROR("CuMat::minMax");
        }
        
        /**
         * @brief Counts the number of non-zero elements
         *        Complex or multi-channel data counts only once
         *        Same as Matlab nnz
         * @returns Integer number of non-zeros
         */
        size_t countNonZeros();
        
        
        // ACCESS

        /** 
         * @brief Returns the matrix data as an OpenCV Mat object
         * @returns Empty matrix if data has not been set
         * @param format Specifies format of the file
         *        FILEMODE_BINARY Reads data from a binary file
         *        FILEMODE_ASCII Reads data from a space delimited ASCII file
         */
        cv::Mat getMatData();

        /**
         * @brief Sets the data referenced by the CuMat object to be equal to 
         *        that of the input array
         */
        void setMatData(cv::InputArray input);

        /**
         * @brief Returns a pointer to CUDA device memory storing the data
         * @returns Pointer to device memory regardless of whether that memory
         *          has been allocated to the desired size or contains data
         */
        void* getCuData();

        /** 
         * @brief Replaces the data stored by the object with the data in the 
         *        input pointer to cuda device memory
         */
        void setCuData(void* input);

        /**
        * @brief Sets size of array in addition to changing stored data
        * @param input Pointer to cuda device memory
        * @param size Allocated space in bytes
        */
        void setCuData(void* input, size_t size);

        /**
         * @brief Sets size of array in addition to changing stored data
         * @param input Pointer to cuda device memory
         * @param r Number of rows in input array
         * @param c Number of columns in input array
         */
        void setCuData(void* input, size_t r, size_t c, int type);

        /**
        * @brief Sets size of 3D array in addition to changing stored data
        * @param input Pointer to cuda device memory
        * @param r Number of rows in input array
        * @param c Number of columns in input array
        * @param d Depth of input array
        */
        void setCuData(void* input, size_t r, size_t c, size_t d, int type);

        /**
         * @brief Sets number of rows and adjusts size of data accordingly.
         *        Does not modify existing data
         * @param r New number of rows
         */
        void setRows(size_t r);
        void setHeight(size_t h) { setRows(h); }

        /**
        * @brief Sets number of columns and adjusts size of data accordingly.
        *        Does not modify existing data
        * @param c New number of columns
        */
        void setCols(size_t c);
        void setWidth(size_t w) { setCols(w); }

        /**
        * @brief Sets 3D depth and adjusts size of data accordingly.
        *        Does not modify existing data
        * @param d New matrix depth
        */
        void setDepth(size_t d);
        
        /**
         * @brief Sets size of each voxel (in bytes) and adjusts size of
         *        data accordingly. Does not modify existing data.
         *        Data type is inferred from size as floating point
         * @param es New element size
         */
        void setElemSize(size_t es);

        size_t getDataSize() { return data_size; }

        size_t getRows() { return rows; }
        size_t getCols() { return cols; }
        size_t getDepth() { return depth; }

        size_t getHeight() { return rows; }
        size_t getWidth() { return cols; }

        int getDataType() { return data_type; }
        
        /**
         * @brief Sets the type of data to be stored Does not modify 
         * existing data. Data size is adjusted accordingly
         * @param type One of the OpenCV types (i.e. CV_32FC2)
         */
        void setDataType(int type);
        
        /**
         * @brief Returns size of single element (voxel) in bytes
         *        e.g. Returns 1 if uint8 or CV_8U, 8 if float2 or CV_32FC2
         */
        size_t getElemSize() { return elem_size; }

        /**
        * @brief Returns real component of complex data
        * @returns CuMat with element depth 1
        */
        CuMat getReal();

        /**
        * @brief Returns imaginary component of complex data
        * @returns CuMat with element depth 1
        */
        CuMat getImag();

        /**
        * @brief Returns magnitude of complex data
        * @returns CuMat with element depth 1
        */
        CuMat getMag();
        
        /**
         * @brief Returns phase of complex data with no unwrapping
         * @returns CuMat with element depth 1, data scaled [0, 2*pi]
         */
        CuMat getPhase();

        // INQUIRY

        /**
         * @brief Returns true if data is stored on device
         */
        bool isStoredCu() const { return (stored_cu) ? true : false; }

        /**
         * @brief Returns true if device data has been allocated. The only way
         *        to deallocate is by using destoy. If cudaFree is called
         *        outside of this class, it will result in unspecified behavior
         */
        bool isAllocated() const { return (is_allocated) ? true : false; }
        
        bool isEmpty() const { return (rows == 0) || (cols == 0); }
        
        bool hasNan();
        
        bool isZero() const { return (is_zero)? true : false; }
        
        /**
         * @brief Data will be assumed zero until next modification
         *      Highly dangerous, use with caution
         */
        void declareZero() { is_zero = true; }
        
        // Used for simple communication between functions (e.g. current plane)
        int identifier;

    protected:
    private:
        cv::Mat mat_data;
        void* cu_data;

        bool stored_mat;
        bool stored_cu;
        bool is_allocated;
        bool is_zero;

        size_t data_size;
        size_t rows;
        size_t cols;
        size_t depth;

        int data_type;
        size_t elem_size;
        size_t allocated_size;
        
        // Small buffer (64 bytes) to save allocation time for temp storage
        void* buffer_64_d;
        bool is_allocated_buffer_64_d;
    };

    // OVERLOADS

    /**
     * @brief Returns the minimum of 2 CuMat arrays
     * @param src1 Input data to compare. Must be of type float
     * @param src2 Second input data to compare to src1. Must have same type
     *             as src1
     * @param dst Destination array
     */
    void min(CuMat src1, CuMat src2, CuMat dst);

    /**
     * @brief Returns the maximum of 2 CuMat arrays
     * @param src1 Input data to compare. Must be of type float
     * @param src2 Second input data to compare to src1. Must have same type
     *             as src1
     * @param dst Destination array
     */
    void max(CuMat src1, CuMat src2, CuMat dst);

    /**
     * @brief Returns the maximum of 2 CuMat arrays and a map indicating which
     *        elements belong to which source array
     * @param src1 Input data to compare. Must be of type float
     * @param src2 Second input data to compare to src1. Must have same type
     *             as src1
     * @param dst Destination array for the max
     * @param id2 Identifier for src2. Output elements originating from src2
     *            will be set to this value.
     * @param arg Desitination array for the identifier. Where the element in
     *            dst originated from src2 it will be set to id2, otherwise it
     *            is unchanged.
     */
    void argmax(CuMat src1, CuMat src2, CuMat dst, float id2, CuMat arg);

    /**
     * @brief Elementwise multiplication of two arrays
     * @param src1 First array for multiplication. Must be of type float or float2
     * @param src2 Second arry. Must be same size and type as src1
     * @param dst Destination array. May be same as src1 or src2 for in place
     */
    void mult(CuMat &src1, CuMat &scr2, CuMat &dst);

    /**
     * @brief Multiplication of arrays and a scalar
     * @param src1 Array for multiplication. Must be of type float or float2
     * @param dst Destination array. May be same as src1 for in place
     */
    void mult(CuMat &src1, float scalar, CuMat &dst);

    /**
     * @brief Returns the sum of 2 CuMat arrays
     * @param src1 Input data to sum
     * @param src2 Second input data for sum
     * @param dst Destination array
     */
     void sum(CuMat &src1, CuMat &src2, CuMat &dst);

    /**
     * @brief Returns the difference of 2 CuMat arrays
     * @param src1 Input data to subtract
     * @param src2 Second input data for subtract
     * @param dst Destination array
     */
     void subtract(CuMat &src1, CuMat &src2, CuMat &dst);

} // namespace umnholo

#endif // CUMAT_H
