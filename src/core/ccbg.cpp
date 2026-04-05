#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric> 

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>


namespace fs = std::filesystem;

// Utility function: load a grayscale image
cv::Mat loadImage(const std::string &path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return img;
}

// Compute normalized cross-correlation between two images (flattened)
float computeCorrelation(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) throw std::runtime_error("Size mismatch");
    float meanA = std::accumulate(a.begin(), a.end(), 0.0f) / a.size();
    float meanB = std::accumulate(b.begin(), b.end(), 0.0f) / b.size();
    float num = 0.0f, denA = 0.0f, denB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float da = a[i] - meanA;
        float db = b[i] - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
    }
    return num / (std::sqrt(denA * denB) + 1e-8f);
}

// Find optimal number of frames to average based on entropy gradient
int findOptimalFrameCount(const std::vector<float> &entropies) {
    for (size_t i = 1; i < entropies.size(); ++i) {
        float grad = entropies[i] - entropies[i-1];
        if (grad < 0)
            return static_cast<int>(i);
    }
    return static_cast<int>(entropies.size());
}

// Compute entropy of a flattened image
float computeEntropy(const std::vector<float> &img) {
    std::vector<int> hist(256,0);
    for (auto v : img) hist[std::min(255, std::max(0,int(std::round(v))))]++;
    float entropy = 0;
    int N = img.size();
    for (int h : hist) if (h > 0) {
        float p = float(h)/N;
        entropy -= p * std::log(p);
    }
    return entropy;
}

// Flatten cv::Mat to std::vector<T> (templated)
template <typename T>
std::vector<T> flattenImage(const cv::Mat &img) {
    std::vector<T> out;
    out.reserve(img.rows * img.cols);

    if (img.type() == CV_8UC1) {
        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                out.push_back(static_cast<T>(img.at<uint8_t>(i,j)));
    } else if (img.type() == CV_32FC1) {
        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                out.push_back(static_cast<T>(img.at<float>(i,j)));
    } else {
        throw std::runtime_error("Unsupported cv::Mat type in flattenImage");
    }

    return out;
}

// Main correlation-based background computation with entropy smoothing and min/max frame limits
void computeBackgrounds(const std::string &inputFolder,
                        const std::string &outputFolder,
                        const std::string &filePattern,
                        int startIdx, int endIdx,
                        int minFrames, int maxFrames,
                        float resizePct = 1.0f,
                        bool saveEnhanced = false)
{
    int numImages = endIdx - startIdx + 1;

    std::vector<cv::Mat> imagesFull(numImages);    // Full-resolution images
    std::vector<cv::Mat> imagesResized(numImages); // Resized images for correlation
    std::vector<std::vector<float>> imagesFlat(numImages); // Flattened resized images

    // Load images
    for (int i = 0; i < numImages; ++i) {
        char filename[512];
        snprintf(filename, sizeof(filename), (inputFolder + "/" + filePattern).c_str(), startIdx + i);
        cv::Mat img = loadImage(filename); // CV_8UC1
        cv::Mat imgF;
        img.convertTo(imgF, CV_32FC1);    // Convert full-res to float
        imagesFull[i] = imgF;

        // Create resized version for correlation
        cv::Mat imgResized;
        if (resizePct < 1.0f) {
            cv::resize(imgF, imgResized, cv::Size(), resizePct, resizePct, cv::INTER_AREA);
        } else {
            imgResized = imgF.clone();
        }
        imagesResized[i] = imgResized;
        imagesFlat[i] = flattenImage<float>(imgResized);
    }

    int rows = imagesFull[0].rows;
    int cols = imagesFull[0].cols;

    for (int n = 0; n < numImages; ++n) {
        // Compute correlation with all other images
        std::vector<std::pair<float,int>> corrScores;
        for (int j = 0; j < numImages; ++j) {
            if (j == n) continue;
            float corr = computeCorrelation(imagesFlat[n], imagesFlat[j]);
            corrScores.emplace_back(corr,j);
        }
        std::sort(corrScores.begin(), corrScores.end(),
                  [](const auto &a, const auto &b){ return a.first > b.first; });

        int m_max = std::min(maxFrames, int(corrScores.size()));
        std::vector<float> entropies(m_max, 0.0f);

        cv::Mat bgResizedF = cv::Mat::zeros(imagesResized[n].size(), CV_32FC1);

        // Compute entropy for top frames (on resized images)
        for (int m = 0; m < m_max; ++m) {
            int idx = corrScores[m].second;
            bgResizedF += imagesResized[idx];
            cv::Mat avgResizedF = bgResizedF / float(m+1);
            cv::Mat enh = imagesResized[n] - avgResizedF;
            entropies[m] = computeEntropy(flattenImage<float>(enh));
        }

        // Smooth entropy with simple 3-point moving average
        std::vector<float> entSmooth(entropies.size(),0.0f);
        for (size_t i = 0; i < entropies.size(); ++i) {
            float sum = entropies[i];
            int count = 1;
            if (i > 0) { sum += entropies[i-1]; count++; }
            if (i+1 < entropies.size()) { sum += entropies[i+1]; count++; }
            entSmooth[i] = sum / count;
        }

        // Find first index where gradient < 0
        int optimalFrames = 1;
        for (size_t i = 1; i < entSmooth.size(); ++i) {
            float grad = entSmooth[i] - entSmooth[i-1];
            if (grad < 0) {
                optimalFrames = static_cast<int>(i);
                break;
            }
        }

        // Clamp optimalFrames to minFrames and maxFrames
        optimalFrames = std::max(minFrames-1, std::min(optimalFrames, m_max-1));

        // Compute final background using full-res original images
        cv::Mat finalBgF = cv::Mat::zeros(rows, cols, CV_32FC1);
        for (int k = 0; k <= optimalFrames; ++k) {
            int idx = corrScores[k].second;
            finalBgF += imagesFull[idx];
        }
        finalBgF /= float(optimalFrames+1);

        // Save background
        cv::Mat bg8U;
        finalBgF.convertTo(bg8U, CV_8UC1);
        char outName[512];
        snprintf(outName, sizeof(outName), (outputFolder + "/background_%04d.png").c_str(), startIdx + n);
        cv::imwrite(outName, bg8U);

        // Optional: save enhanced image (bg-subtracted)
        if (saveEnhanced) {
            cv::Mat enhancedF = imagesFull[n] - finalBgF;
            cv::normalize(enhancedF, enhancedF, 0, 255, cv::NORM_MINMAX);
            cv::Mat enhanced8U;
            enhancedF.convertTo(enhanced8U, CV_8UC1);
            snprintf(outName, sizeof(outName), (outputFolder + "/enhanced_%04d.png").c_str(), startIdx + n);
            cv::imwrite(outName, enhanced8U);
        }

        std::cout << "Saved background for image " << startIdx + n
                  << " using " << (optimalFrames+1) << " frames"
                  << (saveEnhanced ? " + enhanced image" : "") << "\n";
    }
}

// Trim whitespace from string ends
inline std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Reads a params file into a map<string,string>
std::map<std::string, std::string> readParams(const std::string &filename) {
    std::map<std::string, std::string> params;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open params file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue; // skip blank lines or comments
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue; // skip malformed lines
        std::string key = trim(line.substr(0, eqPos));
        std::string value = trim(line.substr(eqPos + 1));
        params[key] = value;
    }
    return params;
}


int main(int argc, char* argv[]) {
    try {
        // Check for command-line argument
        std::string paramFile = "params.txt"; // default
        if (argc > 1) {
            paramFile = argv[1]; // use provided file
        }
        //std::cout << "Reading params from: " << paramFile << std::endl;

        auto params = readParams(paramFile);

        // Extract parameters with defaults
        std::string inputFolder = params.count("inputFolder") ? params["inputFolder"] : "./input";
        std::string outputFolder = params.count("outputFolder") ? params["outputFolder"] : "./backgrounds";
        std::string filePattern = params.count("filePattern") ? params["filePattern"] : "im_%04d.tif";
        int startIdx = params.count("startIdx") ? std::stoi(params["startIdx"]) : 1;
        int endIdx = params.count("endIdx") ? std::stoi(params["endIdx"]) : 10;
        int minFrames = params.count("minFrames") ? std::stoi(params["minFrames"]) : 5;
        int maxFrames = params.count("maxFrames") ? std::stoi(params["maxFrames"]) : 100;
        float resizePct = params.count("resizePct") ? std::stof(params["resizePct"]) : 0.25f;

        // New flag: save enhanced (bg-subtracted) images
        bool saveEnhanced = false;
        if (params.count("saveEnhanced")) {
            std::string val = params["saveEnhanced"];
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            saveEnhanced = (val == "1" || val == "true" || val == "yes");
        }

        fs::create_directories(outputFolder);

        computeBackgrounds(inputFolder, outputFolder, filePattern,
                   startIdx, endIdx, minFrames, maxFrames, resizePct, saveEnhanced);

                   return 0; //sucess

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1; // failure

    }
}