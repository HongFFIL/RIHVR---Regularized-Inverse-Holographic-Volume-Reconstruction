#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include "tracking.h"

namespace fs = std::filesystem;

// ==========================
// Trim whitespace
// ==========================
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// ==========================
// Read key=value params
// ==========================
std::map<std::string, std::string> readParams(const std::string& filename) {
    std::map<std::string, std::string> params;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open parameter file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        size_t eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            params[key] = value;
        }
    }
    return params;
}

// ==========================
// Load CSV into vector of vectors
// ==========================
std::vector<std::vector<double>> loadCSV(const std::string& filename, bool hasHeader = true) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open CSV file: " + filename);

    std::string line;
    if (hasHeader) std::getline(file, line);

    int lineNumber = hasHeader ? 2 : 1;
    while (std::getline(file, line)) {
        if (line.empty()) { ++lineNumber; continue; }
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            cell = trim(cell);
            if (cell.empty()) continue;
            try { row.push_back(std::stod(cell)); }
            catch (...) { std::cerr << "Warning: skipping invalid number '" << cell
                                     << "' at line " << lineNumber << "\n"; }
        }
        if (!row.empty()) data.push_back(row);
        ++lineNumber;
    }
    return data;
}

// ==========================
// Remove zero particles (x=y=z=0)
// ==========================
void removeZeroParticles(std::vector<std::vector<double>>& data) {
    data.erase(std::remove_if(data.begin(), data.end(),
        [](const std::vector<double>& row) {
            return row.size() >= 3 &&
                   row[0] == 0.0 && row[1] == 0.0 && row[2] == 0.0;
        }), data.end());
}

// ==========================
// Filter particles by size
// ==========================
void filterBySize(std::vector<std::vector<double>>& data, double minSize, double maxSize) {
    data.erase(std::remove_if(data.begin(), data.end(),
        [minSize, maxSize](const std::vector<double>& row) {
            return row.empty() || row.back() < minSize || row.back() > maxSize;
        }), data.end());
}

// ==========================
// Tracking parameters
// ==========================
struct TrackParams {
    double maxdisp = 10.0;
    int good = 10;
    int mem = 10;
    int dim = 3;
    int quiet = 1;
};

// ==========================
// Load particles from CSVs into Particle structs
// ==========================
std::vector<Particle> loadParticles(const std::string& inputFolder,
                                    const std::string& filePattern,
                                    int startIdx, int endIdx,
                                    double minSize, double maxSize,
                                    bool printToConsole = true) {

    std::vector<Particle> particles;
    for (int i = startIdx; i <= endIdx; ++i) {
        char filename[256];
        snprintf(filename, sizeof(filename), filePattern.c_str(), i);
        fs::path filePath = fs::path(inputFolder) / filename;

        if (!fs::exists(filePath)) {
            std::cout << "File not found: " << filePath << "\n";
            continue;
        }

        auto data = loadCSV(filePath.string(), true);
        removeZeroParticles(data);
        filterBySize(data, minSize, maxSize);

        // Convert rows to Particle structs
        for (const auto& row : data) {
            if (row.size() >= 3) {
                Particle p;
                p.coords = { row[0], row[1], row[2] };
                p.t = static_cast<double>(i); // frame number 
                particles.push_back(p);

                //std::cout << " x = " << row[0] << " y = " << row[1] << " z = " << row[2] << "\n";
            }
        }

        if (printToConsole) {
            std::cout << "Frame " << i << " (" << data.size() << " particles loaded)\n";
        }
    }
    return particles;
}


// ==========================
// Main function
// Loads particles, tracks, and saves CSV like MATLAB
// ==========================
int main(int argc, char* argv[]) {
    try {
        std::string paramFile = "params.txt";
        if (argc > 1) paramFile = argv[1];

        auto params = readParams(paramFile);

        std::string inputFolder  = params.count("inputFolder")  ? params["inputFolder"]  : "./CentroiData";
        std::string outputFolder = params.count("outputFolder") ? params["outputFolder"] : "./tracks";
        std::string filePattern  = params.count("filePattern")  ? params["filePattern"]  : "centroids_%04d.csv";
        int startIdx             = params.count("startIdx")     ? std::stoi(params["startIdx"]) : 1;
        int endIdx               = params.count("endIdx")       ? std::stoi(params["endIdx"])   : 10;
        double minSize           = params.count("min_size")     ? std::stod(params["min_size"]) : 0.0;
        double maxSize           = params.count("max_size")     ? std::stod(params["max_size"]) : 1e6;

        TrackParams trackParams;
        trackParams.maxdisp  = params.count("maxdisp") ? std::stod(params["maxdisp"]) : 10.0;
        trackParams.good     = params.count("good")    ? std::stoi(params["good"])    : 10;
        trackParams.mem      = params.count("mem")     ? std::stoi(params["mem"])     : 10;
        trackParams.dim      = params.count("dim")     ? std::stoi(params["dim"])     : 3;
        trackParams.quiet    = params.count("quiet")   ? std::stoi(params["quiet"])   : 1;

        fs::create_directories(outputFolder);

        // Load all particles
        auto particleList = loadParticles(inputFolder, filePattern, startIdx, endIdx, minSize, maxSize);
        std::cout << "\nTotal particles loaded: " << particleList.size() << "\n";

        // Prepare TrackParam
        TrackParam param;
        param.mem = trackParams.mem;
        param.good = trackParams.good;
        param.dim = trackParams.dim;
        param.quiet = (trackParams.quiet != 0);

        //std::cout << "particleList[1] coords: " << particleList[0].coords[0] << ", " << particleList[0].coords[1] << ", " << particleList[0].coords[2] << std::endl;
        
        // Call tracking
        auto trackedResult = track(particleList, trackParams.maxdisp, param);

        // debug print
        // std::cout << "After track():\n" << std::flush;
        std::cout << "Number of particles returned: " << trackedResult.particles.size() << "\n";

        // ==========================
        // Save CSV :  x,y,z,frame,ID
        // ==========================
        std::ofstream outFile(outputFolder + "/tracked_particles.csv");
        outFile << "x,y,z,frame,ID\n";

        for (size_t i = 0; i < trackedResult.particles.size(); ++i) {
            const Particle& p = trackedResult.particles[i];
            int id = trackedResult.ids[i];
            outFile << p.coords[0] << "," << p.coords[1] << "," << p.coords[2]
                    << "," << p.t << "," << id << "\n";
        }
        outFile.close();

        std::cout << "Tracked particles saved to: " << outputFolder + "/tracked_particles.csv\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
