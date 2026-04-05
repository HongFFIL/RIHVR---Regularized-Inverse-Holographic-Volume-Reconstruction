#ifndef RECONSTRUCTIONCOMPUTE_H
#define RECONSTRUCTIONCOMPUTE_H

#include <QString>

class ReconstructionCompute
{
public:
    static bool generateParameterFile(
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
        );

    static bool runReconstructionExe(
        const QString &exePath,
        const QString &paramFilePath,
        int savePlanesSubfolderFlag,
        QString &stdOutput,
        QString &stdError
        );
};

#endif // RECONSTRUCTIONCOMPUTE_H
