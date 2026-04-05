#include "reconstructioncompute.h"
#include <QFile>
#include <QTextStream>
#include <QProcess>
#include <QDebug>

bool ReconstructionCompute::generateParameterFile(
    const QString &paramFilePath,
    int numImages,
    int startImage,
    const QString &inputPath,
    const QString &inputFileFormat,
    const QString &outputPath,
    double startZ,
    double stepZ,
    int numPlanes,
    double wavelength,
    double resolution,
    bool useROICenter,
    int roiX, int roiY, int roiSize,
    bool useROIRect,
    int roiRectX, int roiRectY, int roiRectW, int roiRectH,
    int SegMinIntensity, int SegMinVoxelSz, int SegCloseSz,
    int zeroPadding,
    int inverseIterations,
    double regularizationTau,
    double regularizationTV,
    int numTVIterations,
    bool outputPlanesFlag,
    bool outputEachStepFlag,
    bool useEnhancementFlag,
    bool extractCentroidsFlag,
    const QString &backgroundFilename
    )
{
    QFile paramFile(paramFilePath);
    if (!paramFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open parameter file:" << paramFilePath;
        return false;
    }

    QTextStream out(&paramFile);

    // Core parameters
    out << "num_images            1001 " << numImages << "\n";
    out << "start_image           1002 " << startImage << "\n";
    out << "input_pathn           2002 " << inputPath << "\n";
    out << "input_filename_format 2003 " << inputFileFormat << "\n";
    out << "output_pathn          2004 " << outputPath << "\n";

    // Z-plane parameters
    out << "start_plane_z         3001 " << startZ << "\n";
    out << "plane_step_dz         3002 " << stepZ << "\n";
    out << "num_planes            3003 " << numPlanes << "\n";
    out << "wavelength            3004 " << wavelength << "\n";
    out << "resolution            3005 " << resolution << "\n";

    // ROI (center)
    if (useROICenter)
    {
        out << "roi_x                 3006 " << roiX << "\n";
        out << "roi_y                 3007 " << roiY << "\n";
        out << "roi_size              3009 " << roiSize << "\n";
    }/*
    else
    {
        out << "roi_x                 3006 \n";
        out << "roi_y                 3007 \n";
        out << "roi_size              3009 \n";
    }*/

    // ROI (rectangle)
    if (useROIRect)
    {
        out << "roi_rect_x            3011 " << roiRectX << "\n";
        out << "roi_rect_y            3012 " << roiRectY << "\n";
        out << "roi_rect_w            3013 " << roiRectW << "\n";
        out << "roi_rect_h            3014 " << roiRectH << "\n";
    }/*
    else
    {
        out << "roi_rect_x            3011 \n";
        out << "roi_rect_y            3012 \n";
        out << "roi_rect_w            3013 \n";
        out << "roi_rect_h            3014 \n";
    }*/

    // Misc parameters
    out << "zero_padding          3010 " << zeroPadding << "\n";
    out << "inverse_iterations    6001 " << inverseIterations << "\n";
    out << "regularization_tau    6002 " << regularizationTau << "\n";
    out << "regularization_TV     6004 " << regularizationTV << "\n";
    out << "num_TV_iterations     6005 " << numTVIterations << "\n";
    out << "output_planes         7001 " << (outputPlanesFlag ? "true" : "false") << "\n";
    out << "output_each_step      7005 " << (outputEachStepFlag ? "true" : "false") << "\n";

    // Enhancement
    if (useEnhancementFlag)
        out << "background_image      2005 " << backgroundFilename << "\n";
    /*
    else
        out << "background_image      2005 \n";
    */

    // Centroid Extraction
    if (extractCentroidsFlag)
    {
        out << "segment_min_voxels    5009 " << SegMinVoxelSz << "\n";
        out << "segment_min_intensity 5010 " << SegMinIntensity << "\n";
        out << "segment_close_size    5011 " << SegCloseSz << "\n";
    }
    paramFile.close();
    return true;
}

bool ReconstructionCompute::runReconstructionExe(
    const QString &exePath,
    const QString &paramFilePath,
    int savePlanesSubfolderFlag,
    QString &stdOutput,
    QString &stdError
    )
{
    QProcess process;
    process.start(exePath, QStringList() << "-F" << paramFilePath << "-S" << QString::number(savePlanesSubfolderFlag));

    if (!process.waitForFinished(-1)) // wait indefinitely for now
    {
        qDebug() << "Inverse reconstruction process did not finish correctly.";
        return false;
    }

    stdOutput = process.readAllStandardOutput();
    stdError  = process.readAllStandardError();

    qDebug() << "sparse-inverse-recon.exe output:" << stdOutput;
    if (!stdError.isEmpty())
        qDebug() << "sparse-inverse-recon.exe errors:" << stdError;

    return true;
}
