@echo off
REM ============================================
REM CUDA + C++ build for RIHVR project
REM Builds only:
REM   - make-background.exe   (from main_create_background.cu)
REM   - segmentation.exe      (from main_segmentation.cu)
REM Warnings are fully suppressed; errors still stop the build
REM ============================================

setlocal enabledelayedexpansion

REM --- Paths ---
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "OPENCV_PATH=C:\OpenCVLib\install\opencv"
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"

REM --- Initialize MSVC environment (auto-sets WindowsSDKVersion + VCToolsInstallDir) ---
call "%VS_PATH%\vcvarsall.bat" x64

REM --- Add CUDA bin to PATH ---
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM --- Compiler flags ---
set NVCC_FLAGS=-std=c++17 -m64 -O3 -use_fast_math -w ^
 -gencode arch=compute_120,code=compute_120 ^
 -gencode arch=compute_75,code=sm_75 ^
 -gencode arch=compute_75,code=compute_75 ^
 -gencode arch=compute_86,code=sm_86 ^
 -gencode arch=compute_86,code=compute_86
set CC_FLAGS=/O2 /EHsc /w /MT /std:c++17

REM --- Include paths ---
set INCLUDES=-I. ^
 -I"%CUDA_PATH%\include" ^
 -I"%CUDA_PATH%\samples\common\inc" ^
 -I"%OPENCV_PATH%\include"

REM --- Library paths ---
set LIB_PATHS=-L"%CUDA_PATH%\lib\x64" ^
 -L"%OPENCV_PATH%\x64\vc17\lib" 
 -L"%VCToolsInstallDir%lib\x64" ^
 -L"%WindowsSdkDir%Lib\%WindowsSDKVersion%\ucrt\x64" ^
 -L"%WindowsSdkDir%Lib\%WindowsSDKVersion%\um\x64"
	
REM --- Libraries ---
REM CUDA-related libraries
set CUDA_LIBS=cufft.lib curand.lib nppc.lib nppial.lib nppist.lib nppig.lib nppim.lib ^
 nppitc.lib nppisu.lib nppidei.lib npps.lib
REM OpenCV library
set CV_LIBS=opencv_world4130.lib
REM Legacy and standard runtime libs
set LEGACY_LIBS=legacy_stdio_definitions.lib ucrt.lib vcruntime.lib msvcrt.lib
REM Combine everything into one LIBS variable
set LIBS=%CUDA_LIBS% %CV_LIBS% %LEGACY_LIBS%


REM --- Collect CUDA and C++ sources ---
set CUDA_SRC=
for %%F in (*.cu) do set CUDA_SRC=!CUDA_SRC! %%F

set CPP_SRC=
for %%F in (*.cpp *.c) do (
    if /I "%%~xF"==".cpp" if /I not "%%~nF"=="ccbg" if /I not "%%~nF"=="tracking" if /I not "%%~nF"=="particletracking"  set CPP_SRC=!CPP_SRC! %%F
    if /I "%%~xF"==".c" if /I not "%%~nF"=="getopt"  set CPP_SRC=!CPP_SRC! %%F
)


REM =======================================================
REM  Compile all .cu → .obj
REM =======================================================
echo Compiling CUDA sources...
for %%F in (!CUDA_SRC!) do (
    echo Compiling %%F...
    nvcc %INCLUDES% %NVCC_FLAGS% -Xcompiler "%CC_FLAGS%" -c %%F -o %%~nF.obj
    if errorlevel 1 (
        echo ERROR compiling %%F
        exit /b 1
    )
)

REM =======================================================
REM  Compile all .cpp → .obj
REM =======================================================
echo Compiling C++ sources...
for %%F in (!CPP_SRC!) do (
    echo Compiling %%F...
    cl /c /EHsc /MT /std:c++17 /O2 /w %INCLUDES% %%F
    if errorlevel 1 (
        echo ERROR compiling %%F
        exit /b 1
    )
)

REM =======================================================
REM  Link targets
REM =======================================================

set TARGETS=main_create_background.cu main_segmentation.cu main_sparse_inverse.cu

for %%M in (%TARGETS%) do (
    if exist %%M (
        echo.
        echo Linking executable for %%M...

        set OBJ_LIST=

        REM --- include all CUDA objects except any main_*.obj ---
		for %%F in (!CUDA_SRC!) do (
			echo %%~nF | findstr /R "^main_.*$" >nul
			if errorlevel 1 (
				set OBJ_LIST=!OBJ_LIST! %%~nF.obj
			)
		)

		REM --- include all C++ objects except any file with main or image_tiling ---
		for %%F in (!CPP_SRC!) do (
			echo %%~nF | findstr /R "^main_.*$ ^main$ ^image_tiling$" >nul
			if errorlevel 1 (
				set OBJ_LIST=!OBJ_LIST! %%~nF.obj
			)
		)

		REM --- add only the main_*.obj for this target ---
		set OBJ_LIST=!OBJ_LIST! %%~nM.obj

        REM --- define executable name ---
        if /I "%%~nM"=="main_create_background" (
            set EXE_NAME=make-background.exe
        ) else if /I "%%~nM"=="main_segmentation" (
            set EXE_NAME=segmentation.exe
		) else if /I "%%~nM"=="main_sparse_inverse" (
            set EXE_NAME=sparse-inverse-recon.exe	
        ) else (
            set EXE_NAME=%%~nM.exe
        )

        nvcc %LIB_PATHS% -Xlinker /NODEFAULTLIB:LIBCMT -o !EXE_NAME! !OBJ_LIST! %LIBS%
        if errorlevel 1 (
            echo ERROR linking !EXE_NAME!
            exit /b 1
        )
        echo Successfully built !EXE_NAME!
    )
)

REM =======================================================
REM  Compile ccbg and particle tracking exe files
REM =======================================================
set EXE_NAME=ccbg.exe
cl /EHsc /O2 /std:c++17 %INCLUDES% ccbg.cpp /link /LIBPATH:%OPENCV_PATH%\x64\vc17\lib %CV_LIBS%
if errorlevel 1 (
	echo ERROR linking %EXE_NAME%
	exit /b 1
	)
echo Successfully built %EXE_NAME%

set EXE_NAME=particletracking.exe
cl /EHsc /O2 /std:c++17 particletracking.cpp tracking.cpp
if errorlevel 1 (
	echo ERROR linking %EXE_NAME%
	exit /b 1
	)
echo Successfully built %EXE_NAME%


REM =======================================================
REM  Clean up intermediate files and move executables
REM =======================================================

REM --- Create lib folder if it doesn't exist ---
if not exist lib mkdir lib

REM --- Move each built executable and delete intermediary files ---
for %%M in (%TARGETS%) do (
    if /I "%%~nM"=="main_create_background" (
        set EXE_NAME=make-background.exe
    ) else if /I "%%~nM"=="main_segmentation" (
        set EXE_NAME=segmentation.exe
    ) else if /I "%%~nM"=="main_sparse_inverse" (
        set EXE_NAME=sparse-inverse-recon.exe
    ) else (
        set EXE_NAME=%%~nM.exe
    )

    if exist !EXE_NAME! (
        move /Y !EXE_NAME! lib\
    )
)

set EXE_NAME=ccbg.exe
if exist %EXE_NAME% (move /Y !EXE_NAME! lib\)

set EXE_NAME=particletracking.exe
if exist %EXE_NAME% (move /Y !EXE_NAME! lib\)


REM --- Delete all .obj, .lib, .exp files generated during build ---
del /Q *.obj *.lib *.exp

echo.
echo All builds complete.
pause
