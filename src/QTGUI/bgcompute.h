#ifndef BGCOMPUTE_H
#define BGCOMPUTE_H

#include <QString>

class BGCompute
{
public:
    static bool generateParameterFile(
        const QString &paramFilePath,
        int numImages,      // 1001
        int startImage,     // 1002
        int bgStartImage,   // 1003
        int bgEndImage,     // 1004
        const QString &inputPath, // 2002
        const QString &inputFileFormat, // 2003
        const QString &outputPath, // 2004
        bool useMedian // 7006
        );

    static bool generateCCBGParameterFile(
        const QString &paramFilePath,
        int bgStartImage,
        int bgEndImage,
        double resizePct,
        int minFrames,
        int maxFrames,
        const QString &inputPath,
        const QString &inputFileFormat,
        const QString &outputPath,
        bool saveEnhanced
        );


    static bool runBackgroundExe(
        const QString &exePath,
        const QString &paramFilePath,
        QString &stdOutput,
        QString &stdError
        );
};

#endif
