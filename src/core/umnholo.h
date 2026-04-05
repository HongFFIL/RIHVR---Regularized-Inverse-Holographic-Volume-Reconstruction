#ifndef UMNHOLO_H
#define UMNHOLO_H

//#include "holo_ui.h"
//#include "cumat.h"
//#include "hologram.h"
//#include "optical_field.h"
#include <time.h>
#include <string.h>
#include <iostream>

#include "cuda.h"

#define CHECK_FOR_ERROR(x)\
{\
cudaDeviceSynchronize();\
cudaError_t err = cudaGetLastError();\
if(err != cudaSuccess)\
    {\
	fprintf(stderr, "%s: Error: %s\n", x, cudaGetErrorString(err));\
	throw CUDA_ERROR_UNKNOWN;\
    }\
}

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(x)\
{\
	if(cudaSuccess != x)\
    	{\
		printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(x));\
		throw CUDA_ERROR_UNKNOWN;\
    	}\
}
#endif

#ifndef CURAND_SAFE_CALL
#define CURAND_SAFE_CALL(x)\
{\
	if(cudaSuccess != x)\
    	{\
		printf("CURAND error at %s:%d: %s\n", __FILE__, __LINE__);\
		throw CUDA_ERROR_UNKNOWN;\
    	}\
}
#endif

#ifndef NPP_SAFE_CALL
#define NPP_SAFE_CALL(x)\
{\
	if(NPP_SUCCESS != x)\
    	{\
		printf("NPP error at %s:%d: %s\n", __FILE__, __LINE__);\
		throw CUDA_ERROR_UNKNOWN;\
    	}\
}
#endif

#ifndef CUFFT_SAFE_CALL
#define CUFFT_SAFE_CALL(x)\
{\
	if(CUFFT_SUCCESS != x)\
    	{\
		printf("CUFFT error at %s:%d: %d\n", __FILE__, __LINE__, (x));\
		throw CUDA_ERROR_UNKNOWN;\
    	}\
}
#endif

#define CHECK_MEMORY(x)\
{\
    size_t free_byte, total_byte;\
    cudaMemGetInfo( &free_byte, &total_byte );\
    printf("Free memory %s: %d kB of %d\n", x, (free_byte/1024), (total_byte/1024));\
    /*std::cout << "waiting for input: ";\
    char a;\
    std::cin >> a;*/\
}

// Adapted from http://aishack.in/tutorials/timing-macros/ by Shervin Emami
// Record the execution time of some code, in milliseconds.
//#define RECORD_TIMING
#ifdef RECORD_TIMING
#define DECLARE_TIMING(s)  clock_t timeStart_##s; double timeDiff_##s; double timeTally_##s = 0; int countTally_##s = 0
#define START_TIMING(s)    timeStart_##s = clock()
#define STOP_TIMING(s)     cudaDeviceSynchronize(); timeDiff_##s = (double)(clock() - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++
#define GET_TIMING(s)      (double)(timeDiff_##s / (CLOCKS_PER_SEC/1000.0))
#define GET_AVERAGE_TIMING(s)   (double)(countTally_##s ? timeTally_##s/ ((double)countTally_##s * CLOCKS_PER_SEC/1000.0) : 0)
#define CLEAR_AVERAGE_TIMING(s) timeTally_##s = 0; countTally_##s = 0
#define PRINT_TIMING(s)   STOP_TIMING(s); std::cout << "Time taken for " << #s << ": " << GET_TIMING(s) << " ms" << std::endl;
#define SAVE_TIMING(s)\
{\
    FILE* timefid = fopen("../TestResult/timing.csv", "a");\
    if (timefid == NULL)\
    {\
        timefid = fopen("timing.csv", "a");\
    }\
    STOP_TIMING(s);\
    fprintf(timefid, "%s, %f\n", #s, GET_TIMING(s));\
    fclose(timefid);\
}
#define OPEN_TIMING_FILE()\
{\
    FILE* timefid = fopen("../TestResult/timing.csv", "w");\
    if (timefid == NULL)\
    {\
        timefid = fopen("timing.csv", "w");\
        if (timefid == NULL) {\
            printf("Unable to open timing file\n");\
            throw HOLO_ERROR_BAD_FILENAME;\
        }\
    }\
    fclose(timefid);\
}
#else
#define DECLARE_TIMING(s) {}
#define START_TIMING(s) {}
#define STOP_TIMING(s) {}
#define GET_TIMING(s) { 0.0 }
#define GET_AVERAGE_TIMING(s) { 0.0 }
#define CLEAR_AVERAGE_TIMING(s) {}
#define PRINT_TIMING(s) {}
#define SAVE_TIMING(s) {}
#define OPEN_TIMING_FILE() {}
#endif

//#define STRINGIFY(x) #x
//#define TOSTRING(x) STRINGIFY(x)
//#define HERE __FILE__ ":" TOSTRING(__LINE__) ":(" __FUNC__ ")"

namespace umnholo
{
    enum HoloError
    {
        HOLO_ERROR_UNKNOWN_TYPE = 1,
        HOLO_ERROR_BAD_FILENAME = 2,
        HOLO_ERROR_MISSING_DATA = 3,
        HOLO_ERROR_INVALID_STATE = 4,
        HOLO_ERROR_OUT_OF_MEMORY = 5,
        HOLO_ERROR_UNKNOWN_MODE = 6,
        HOLO_ERROR_INVALID_ARGUMENT = 7,
        HOLO_ERROR_UNKNOWN_ERROR = 8,
        HOLO_ERROR_INVALID_FILE = 9,
        HOLO_ERROR_INVALID_DATA = 10,
        HOLO_ERROR_CRITICAL_ASSUMPTION = 11,
        HOLO_ERROR_INFINITE_LOOP = 12
    };
    
    enum CmbMethod
    {
        MIN_CMB = 0,
        MAX_CMB = 1,
        PHASE_CMB = 2
    };
}
#endif // UMNHOLO_H
