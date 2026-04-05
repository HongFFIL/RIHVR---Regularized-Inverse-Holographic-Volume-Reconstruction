#ifndef TRACKINGCOMPUTE_H
#define TRACKINGCOMPUTE_H

#include <QString>

class TrackingCompute
{
public:
    static bool generateParameterFile(
        const QString &paramFilePath,
        int trackStartIdX, int trackEndIdX,
        int maxParticleSize, int minParticeSize,
        int maxDisplacement, int minTrajectorySize,
        int memoryParticle, int dimensions,
        const QString &inputPath, const QString &inputFileFormat,
        const QString &outputPath,
        bool quiet
        );
};




#endif // TRACKINGCOMPUTE_H
