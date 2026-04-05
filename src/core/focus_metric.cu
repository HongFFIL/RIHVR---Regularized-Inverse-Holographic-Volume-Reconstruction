#include "focus_metric.h"  // class implemented
#include <float.h>
#include <iomanip>

using namespace umnholo;

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

FocusMetric::FocusMetric(int size)
{
    num_points = size;
    metric.reserve(size);

    raw_real = (double*)malloc(size * sizeof(double));
    raw_imag = (double*)malloc(size * sizeof(double));
    
    state = FOCUS_STATE_EMPTY;

    return;
}

void FocusMetric::init(int size)
{
    num_points = size;
    metric.reserve(size);

    raw_real = (double*)malloc(size * sizeof(double));
    raw_imag = (double*)malloc(size * sizeof(double));

    state = FOCUS_STATE_EMPTY;

    return;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

void FocusMetric::write(char* filename)
{
    if (state == FOCUS_STATE_RAW)
    {
        FILE* fid = fopen(filename, "w");
        if (fid == NULL)
        {
            std::cout << "FocusMetric::write: Unable to open file: " << filename << std::endl;
            throw HOLO_ERROR_BAD_FILENAME;
        }

        // Old code:
        // #if defined(WIN32) || defined(_WIN32)
        //     _set_output_format(_TWO_DIGIT_EXPONENT);
        // #endif

#if defined(_MSC_VER) && !defined(__CUDACC__)
        // Set output format to two-digit exponent on MSVC host builds only
        _set_output_format(_TWO_DIGIT_EXPONENT);
#endif

        for (int i = 0; i < num_points; ++i)
        {
            fprintf(fid, "   %0.7E", raw_real[i]);
            fprintf(fid, "   %0.7E", raw_imag[i]);
            fprintf(fid, "\n");
        }

        fclose(fid);
    }
    return;
}

void FocusMetric::write_binary(char* filename)
{
    if (state == FOCUS_STATE_RAW)
    {
        FILE* fid = fopen(filename, "w");
        if (fid == NULL)
        {
            std::cout << "FocusMetric::write: Unable to open file: " << filename << std::endl;
            throw HOLO_ERROR_BAD_FILENAME;
        }

        /*
        unsigned char foo = 0x8C;
        printf("\nfoo = %02x\n", foo);
        unsigned char bar = foo << 4;
        printf("left shift = %02x\n", bar);
        printf("right shift = %02x\n", foo >> 4);
        printf("left then right = %02x\n", bar >> 4);

        char char_data[8];
        double num_data[1];
        printf("\nsize of char: %d\n", sizeof(char));
        printf("\nsize of double: %d\n", sizeof(double));
        printf("\nsize of unsigned long long int: %d\n", sizeof(unsigned long long int));
        
        for (int i = 0; i < num_points; ++i)
        {
            // *num_data = raw_real[i];
            //char_data = reinterpret_cast<char*>(num_data);
            //char_data
            //double x = raw_real[i];
            double dreal1 = raw_real[i];
            double dreal2 = raw_real[i];
            double dimag1 = raw_imag[i];
            double dimag2 = raw_imag[i];
            long int* real1 = (long int*)&dreal1;
            long int* real2 = (long int*)&dreal2;
            long int* imag1 = (long int*)&dimag1;
            long int* imag2 = (long int*)&dimag2;
            *real1 >>= 4*8;
            *imag1 >>= 4*8;
            fprintf(fid, "%08x%08x%08x%08x\n", *real1, *real2, *imag1, *imag2);
        }
        */

        // Print as characters instead to save some space
        for (int i = 0; i < num_points; ++i)
        {
            double real = raw_real[i];
            double imag = raw_imag[i];
            char* char_data = (char*)&real;
            for (int n = 0; n < sizeof(double); ++n)
            {
                int idx = sizeof(double) - n - 1;
                fprintf(fid, "%c", char_data[idx]);
            }
            char_data = (char*)&imag;
            for (int n = 0; n < sizeof(double); ++n)
            {
                int idx = sizeof(double) - n - 1;
                fprintf(fid, "%c", char_data[idx]);
            }
            //fprintf(fid, "\n");
        }

        /*
        // Print as character num instead to check
        fprintf(fid, "\n-----\n");
        for (int i = 0; i < num_points; ++i)
        {
            double real = raw_real[i];
            double imag = raw_imag[i];
            uint8_t* char_data = (uint8_t*)&real;
            for (int n = 0; n < sizeof(double); ++n)
            {
                int idx = sizeof(double) - n - 1;
                fprintf(fid, "%x ", char_data[idx]);
            }
            char_data = (uint8_t*)&imag;
            for (int n = 0; n < sizeof(double); ++n)
            {
                int idx = sizeof(double) - n - 1;
                fprintf(fid, "%x ", char_data[idx]);
            }
            fprintf(fid, "\n");
        }

        // Print the true values for comparison
        fprintf(fid, "\n-----\n");
        for (int i = 0; i < num_points; ++i)
        {
            fprintf(fid, "%f + %fi\n", raw_real[i], raw_imag[i]);
        }
        */

        fclose(fid);
    }
    return;
}

void FocusMetric::read_binary(char* filename)
{

    FILE* fid = fopen(filename, "r");
    if (fid == NULL)
    {
        std::cout << "FocusMetric::write: Unable to open file: " << filename << std::endl;
        throw HOLO_ERROR_BAD_FILENAME;
    }

    int line_counter = 0;
    double real = 0;
    double imag = 0;

    while (!feof(fid))
    {
        //fread(&real, sizeof(double), 1, fid);
        //fread(&imag, sizeof(double), 1, fid);
        //fread(NULL, sizeof(char), 1, fid); // newline
        
        uint8_t* temp_char = (uint8_t*)&real;
        for (int i = 0; i < sizeof(double); ++i)
        {
            int idx = sizeof(double) - i - 1;
            temp_char[idx] = fgetc(fid);
            //printf("Read char: %d\n", temp_char[i]);
        }

        temp_char = (uint8_t*)&imag;        
        for (int i = 0; i < sizeof(double); ++i)
        {
            int idx = sizeof(double) - i - 1;
            temp_char[idx] = fgetc(fid);
            //printf("Read char: %d\n", temp_char[i]);
        }

        //char newline = fgetc(fid);

        // eof isn't set until until actully attempting to read past end
        if (feof(fid))
        {
            break;
        }

        raw_real[line_counter] = real;
        raw_imag[line_counter] = imag;
        //printf("count %d found: %f, %f\n", line_counter, real, imag);
        line_counter++;

        if (line_counter > num_points)
        {
            std::cout << "FocusMetric::read_binary:"
                << "Found more than the expeced values in file"
                << filename << std::endl;
            return;
        }
    }
}

//============================= ACCESS     ===================================

void FocusMetric::setRawDevice(double* real_d, double* imag_d)
{
    DECLARE_TIMING(setRawDevice);
    START_TIMING(setRawDevice);

    cudaMemcpy(raw_real, real_d, num_points * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(raw_imag, imag_d, num_points * sizeof(double), cudaMemcpyDeviceToHost);

    state = FOCUS_STATE_RAW;

    SAVE_TIMING(setRawDevice);
    CHECK_FOR_ERROR("FocusMetric::setRawDevice");
    return;
}

//============================= INQUIRY    ===================================
/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
