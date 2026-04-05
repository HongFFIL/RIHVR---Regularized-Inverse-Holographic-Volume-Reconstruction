#include "bgcompute.h"
#include <QFile>
#include <QTextStream>
#include <QProcess>
#include <QDebug>

bool BGCompute::generateParameterFile(
    const QString &paramFilePath,
    int numImages,
    int startImage,
    int bgStartImage,
    int bgEndImage,
    const QString &inputPath,
    const QString &inputFileFormat,
    const QString &outputPath,
    bool useMedian
    )
{
    QFile paramFile(paramFilePath);
    if (!paramFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open parameter file:" << paramFilePath;
        return false;
    }

    QTextStream out(&paramFile);
    out << "num_images            1001 " << numImages << "\n";
    out << "start_image           1002 " << startImage << "\n";
    out << "bg_start_image        1003 " << bgStartImage << "\n";
    out << "bg_end_image          1004 " << bgEndImage << "\n";
    out << "input_pathn           2002 " << inputPath << "\n";
    out << "input_filename_format 2003 " << inputFileFormat << "\n";
    out << "output_pathn          2004 " << outputPath << "\n";
    out << "use_median_background 7006 " << (useMedian ? 1 : 0) << "\n";

    paramFile.close();
    return true;
}

bool BGCompute::generateCCBGParameterFile(
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
    )
{
    QFile paramFile(paramFilePath);
    if (!paramFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open parameter file:" << paramFilePath;
        return false;
    }

    QTextStream out(&paramFile);
    out << "inputFolder      = " << inputPath << "\n";
    out << "outputFolder     = " << outputPath << "\n";
    out << "filePattern      = " << inputFileFormat << "\n";
    out << "startIdx         = " << bgStartImage << "\n";
    out << "endIdx           = " << bgEndImage << "\n";
    out << "resizePct        = " << resizePct << "\n";
    out << "minFrames        = " << minFrames << "\n";
    out << "maxFrames        = " << maxFrames << "\n";
    out << "saveEnhanced     = " << (saveEnhanced ? true : false) << "\n";
    paramFile.close();
    return true;
}

bool BGCompute::runBackgroundExe(
    const QString &exePath,
    const QString &paramFilePath,
    QString &stdOutput,
    QString &stdError
    )
{
    QProcess process;
    process.start(exePath, QStringList() << "-F" << paramFilePath);

    if (!process.waitForFinished(-1)) // wait indefinitely for now
    {
        qDebug() << "make-background.exe did not finish correctly.";
        return false;
    }

    stdOutput = process.readAllStandardOutput();
    stdError  = process.readAllStandardError();

    qDebug() << "make-background.exe output:" << stdOutput;
    if (!stdError.isEmpty())
        qDebug() << "make-background.exe errors:" << stdError;

    return true;
}
