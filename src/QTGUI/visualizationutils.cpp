#include "visualizationutils.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QStringList>
#include <cstdio>
#include <QRegularExpression>

namespace VizUtils {

// --- existing functions ---
QImage matToQImage(const cv::Mat &mat)
{
    if (mat.empty()) {
        qWarning() << "matToQImage: empty image";
        return QImage();
    }

    switch (mat.type()) {
    case CV_8UC1: // grayscale
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    case CV_8UC3: // color (BGR → RGB)
    {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }
    case CV_8UC4: // BGRA → RGBA
    {
        cv::Mat rgba;
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows, rgba.step, QImage::Format_RGBA8888).copy();
    }
    default:
        qWarning() << "matToQImage: unsupported image type" << mat.type();
        return QImage();
    }
}

QImage loadImageAsQImage(const QString &path)
{
    cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        qWarning() << "Failed to load image:" << path;
        return QImage();
    }
    return matToQImage(img);
}

//  load image as cv::Mat ---
cv::Mat loadImageAsMat(const QString &path)
{
    cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        qWarning() << "Failed to load image as cv::Mat:" << path;
    }
    return img;
}

// Load a background image using a C-style pattern
cv::Mat loadBackgroundImage(const QString &bgFolder,
                            const QString &pattern,
                            int index,
                            bool pickLargestIfMissing)
{
    cv::Mat bgMat;
    QString bgFile;

    if (pattern.isEmpty()) {
        qWarning() << "loadBackgroundImage: empty pattern";
        return bgMat;
    }

    // Use C-style formatting if pattern contains %
    if (pattern.contains("%")) {
        QByteArray patternBA = pattern.toUtf8();
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), patternBA.constData(), index);
        bgFile = QDir(bgFolder).filePath(QString::fromUtf8(buffer));
    } else {
        // pattern is just a plain filename
        bgFile = QDir(bgFolder).filePath(pattern);
    }

    if (!QFileInfo::exists(bgFile)) {
        if (pickLargestIfMissing) {
            // pick the file with the largest number in the folder
            QDir dir(bgFolder);
            QStringList files = dir.entryList(QDir::Files);
            int maxIndex = -1;
            QString maxFile;
            QRegularExpression re("\\d+"); // find digits
            for (const QString &f : files) {
                QRegularExpressionMatch match = re.match(f);
                if (match.hasMatch()) {
                    int val = match.captured(0).toInt();
                    if (val > maxIndex) {
                        maxIndex = val;
                        maxFile = f;
                    }
                }
            }
            if (!maxFile.isEmpty()) {
                bgFile = QDir(bgFolder).filePath(maxFile);
            } else {
                qWarning() << "No suitable background found in" << bgFolder;
                return bgMat;
            }
        } else {
            qWarning() << "Background file not found:" << bgFile;
            return bgMat;
        }
    }

    bgMat = cv::imread(bgFile.toStdString(), cv::IMREAD_UNCHANGED);
    if (bgMat.empty()) {
        qWarning() << "Failed to load background image:" << bgFile;
    }

    return bgMat;
}

} // namespace VizUtils
