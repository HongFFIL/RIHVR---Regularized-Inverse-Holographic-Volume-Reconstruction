#pragma once
#include <vector>

// Parameters for tracking
struct TrackParam {
    int mem = 0;
    int good = 0;
    int dim = 2;
    bool quiet = false;
};

// Individual particle or measurement
struct Particle {
    std::vector<double> coords; // x, y, z...
    double t; // frame/time
};

// Tracking result
struct TrackResult {
    std::vector<Particle> particles;
    int t;
    std::vector<int> ids;
};

// Main tracking function
TrackResult track(const std::vector<Particle>& xyzs, double maxdisp, const TrackParam& param = TrackParam());
