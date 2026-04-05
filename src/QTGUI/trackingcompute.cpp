#include "trackingcompute.h"
#include <QFile>
#include <QTextStream>
#include <QProcess>
#include <QDebug>


bool TrackingCompute::generateParameterFile(
    const QString &paramFilePath,
    int trackStartIdX, int trackEndIdX,
    int maxParticleSize, int minParticleSize,
    int maxDisplacement, int minTrajectorySize,
    int memoryParticle, int dimensions,
    const QString &inputPath, const QString &inputFileFormat,
    const QString &outputPath,
    bool quiet
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
    out << "startIdx         = " << trackStartIdX << "\n";
    out << "endIdx           = " << trackEndIdX << "\n";
    out << "min_size         = " << minParticleSize << "\n";
    out << "max_size         = " << maxParticleSize << "\n";
    out << "maxdisp          = " << maxDisplacement << "\n";
    out << "good             = " << minTrajectorySize << "\n";
    out << "memory           = " << memoryParticle << "\n";
    out << "dim              = " << dimensions << "\n";
    out << "quiet            = " << (quiet ? true : false) << "\n";
    paramFile.close();
    return true;

}
