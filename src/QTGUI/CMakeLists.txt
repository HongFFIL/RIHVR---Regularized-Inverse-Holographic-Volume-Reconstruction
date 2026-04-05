cmake_minimum_required(VERSION 3.16)

project(RIHVR_GUI VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools OpenGL OpenGLWidgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools OpenGL OpenGLWidgets)

# === OpenCV setup ===
# tell CMake where OpenCV is
set(OpenCV_DIR "C:/OpenCVLib/install/opencv/x64/vc17/lib")  # path to OpenCVConfig.cmake
find_package(OpenCV REQUIRED)
# include headers
include_directories(${OpenCV_INCLUDE_DIRS})
# ====================

set(TS_FILES RIHVR_GUI_en_US.ts)

# include the icon for the executable
set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_CURRENT_SOURCE_DIR}/Icons/resources.rc")

set(PROJECT_SOURCES
        main.cpp
        rihvr.cpp
        rihvr.h
        rihvr.ui
        ${TS_FILES}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(RIHVR_GUI
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        Resources.qrc
        settingsdialog.ui
        settingsdialog.cpp
        bgcompute.h
        bgcompute.cpp
        reconstructioncompute.h
        reconstructioncompute.cpp
        trackingcompute.h
        trackingcompute.cpp
        visualizationutils.h
        visualizationutils.cpp
        tracksplot.h
        tracksplot.cpp
        ${APP_ICON_RESOURCE_WINDOWS}         # icon for the executable
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET RIHVR_GUI APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation

    qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
else()
    if(ANDROID)
        add_library(RIHVR_GUI SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(RIHVR_GUI
            ${PROJECT_SOURCES}
        )
    endif()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
endif()


# link OpenCV libraries, data visualization and OpenGL
target_link_libraries(RIHVR_GUI PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::OpenGLWidgets Qt${QT_VERSION_MAJOR}::OpenGL ${OpenCV_LIBS})

#target_link_libraries(RIHVR_GUI PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.RIHVR_GUI)
endif()
set_target_properties(RIHVR_GUI PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS RIHVR_GUI
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(RIHVR_GUI)
endif()
