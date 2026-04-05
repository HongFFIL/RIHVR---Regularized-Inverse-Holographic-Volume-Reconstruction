#ifndef VISUALIZATIONUTILS_H
#define VISUALIZATIONUTILS_H

#include <QImage>
#include <QString>
#include <opencv2/opencv.hpp>

namespace VizUtils {

// Convert OpenCV Mat → QImage
QImage matToQImage(const cv::Mat &mat);

// Load image from file as QImage
QImage loadImageAsQImage(const QString &path);

// Load image from file as cv::Mat
cv::Mat loadImageAsMat(const QString &path);

// Load background image for subtraction
// bgPath: folder where backgrounds live
// templateName: e.g., "background_%04d.png" (empty for single background.png)
// index: number to substitute into template
// pickLargestIfMissing: if true, pick largest numbered file if exact index not found
cv::Mat loadBackgroundImage(const QString &bgPath, const QString &templateName = "", int index = 0, bool pickLargestIfMissing = false);

} // namespace VizUtils

#endif // VISUALIZATIONUTILS_H
