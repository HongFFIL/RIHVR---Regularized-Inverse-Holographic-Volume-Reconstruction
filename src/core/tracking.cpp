#include "tracking.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <set>
#include <deque>
#include <map>
#include <array>


template<typename T>
std::vector<T> circshift(const std::vector<T>& a, int shift)
{
   int n = static_cast<int>(a.size());
     if (n == 0) return {};

    std::vector<T> b(n);
    shift = ((shift % n) + n) % n; // normalize shift to [0, n-1]
    for (int i = 0; i < n; ++i) {
        b[i] = a[(i - shift + n) % n];
    }
    return b;
}



// =========================================================================================================================================
// Main tracking function that returns TrackResult with particles + ids
// =========================================================================================================================================
TrackResult track(
    const std::vector<Particle>& xyzs,
    double maxdisp,
    const TrackParam& param
) {
    if (xyzs.empty()) {
        std::cout << "No particles sent to the tracking function! Returning empty TrackResult.\n";
        return TrackResult{};
    }

    if (param.quiet == 0) { std::cout << "Going to track " << xyzs.size() << " particles\n";}
    
    // Determine frame column index (MATLAB dd equivalent) — we still keep it conceptually
    int dd = (param.dim == 3) ? 4 : 3; // not strictly needed since Particle::t exists

    // Extract time vector t
    std::vector<double> t;
    t.reserve(xyzs.size());
    for (const auto& p : xyzs) {
        t.push_back(p.t);
    }
    
    // Compute differences between consecutive frames
    std::vector<double> st;
    st.reserve(xyzs.size() - 1);
    for (size_t i = 1; i < t.size(); ++i) {
        st.push_back(t[i] - t[i-1]);
    }

    // Check for negative differences (time not in order)
    double negSum = 0;
    for (double dt : st) {
        if (dt < 0) negSum += dt;
    }
    if (negSum != 0) {
        std::cout << "The time vectors is not in order\n";
        return TrackResult{};  
    }

    // info flag
    int info = 1;

    // Find indices where frame difference > 0
    std::vector<size_t> w;
    for (size_t i = 0; i < st.size(); ++i) {
        if (st[i] > 0) w.push_back(i);
    }

    int z = static_cast<int>(w.size()); // matlab has z = length(w)+1, but as we are doing 0-indexing, we don't need the +1

    if (w.empty()) {
        std::cout << "All positions are at the same time... Cannot track! Go back!\n";
        return TrackResult{};
    }

    // ================================
    // Partitioning the data with unique times
    // ================================
    std::vector<size_t> res;
    size_t nnn = t.size();

    // Find indices where the current time differs from the next (like MATLAB circshift)
    for (size_t i = 0; i < nnn - 1; ++i) {
        if (t[i] != t[i + 1]) {
            res.push_back(i);
        }
    }

    // Handle edge case: if no differences found, use last index
    if (res.empty()) {
        res.push_back(nnn - 1);
    }

    // Count of unique time intervals
    size_t count = res.size();

    // =======================================
    // Preallocation and setup
    // =======================================

    // Extend 'res' to include 1 at start and n at end
    res.insert(res.begin(), 0);          // MATLAB 1-based but we have C++ 0-based
    res.push_back(t.size() - 1);         // add last index

    // Number of particles in the first frame
    int ngood = static_cast<int>(res[1] - res[0] + 1);

    // Eyes: indices of particles in the first frame
    std::vector<int> eyes(ngood);
    for (int i = 0; i < ngood; ++i) eyes[i] = i;

    // pos: positions of particles in first frame
    std::vector<Particle> pos;
    pos.reserve(ngood);

    for (int i : eyes) {
        Particle p;
        p.coords.resize(param.dim);
        for (int d = 0; d < param.dim; ++d) {
            p.coords[d] = xyzs[i].coords[d];
        }
        p.t = 0.0;  // optional, but we'll keep it
        pos.push_back(p);
    }

    int istart = 1; // in matlab, this was 2, for us 1 because of 0 based indexing
    int n = ngood;

    // Determine zspan based on the number of particles
    int zspan = 50;
    if (n > 200) zspan = 20;
    if (n > 500) zspan = 10;

    // resx: zspan x n, initialized to -1
    std::vector<std::vector<int>> resx(zspan, std::vector<int>(n, -1));

    // bigresx: z x n, initialized to -1
    std::vector<std::vector<int>> bigresx(z+1, std::vector<int>(n, -1)); // as we need the index to start from 0 for z, we need z+1 x n size actually in C++
    
    // mem: n x 1 zeros
    std::vector<int> mem(n, 0);

    // uniqid: 1:n
    std::vector<int> uniqid(n);
    std::iota(uniqid.begin(), uniqid.end(), 1);

    // maxid = n
    int maxid = n;

    // olist = [0.,0.]
    std::vector<std::pair<double, double>> olist = {{0.0, 0.0}};

    // If goodenough > 0, allocate dumphash and nvalid
    std::vector<int> dumphash, nvalid;
    if (param.good > 0) {
        dumphash.assign(n, 0);
        nvalid.assign(n, 1);
    }

    // resx first row = eyes
    for (int j = 0; j < n; ++j) resx[0][j] = eyes[j];

    // Setup constants
    double maxdisq = maxdisp * maxdisp;

    // Matlab comment: John calls this the setup for "fancy code" ??? -- so let's just say it's a setup for the fancy code
    // but this necessarily decides if we will need to split the problem once the number of particles is like 40k
    bool notnsqrd = (std::sqrt(n * ngood) > 200) && (param.dim < 7);
    
    // initialize some variables we are going use both inside the if statement and later on
    // blocksize -- default
    double blocksize = maxdisp; 
    // construct the vertices of a 3x3x3... d-dimensional hypercube
    int cubeSize = static_cast<int>(std::pow(3, param.dim));
    std::vector<std::vector<int>> cube(cubeSize, std::vector<int>(param.dim, 0));
    // the flag to check if the case is nontrivial or not
    int nontrivial = 0;
    std::vector<int> labelx;
    std::vector<int> labely;
    int nclust = 0;
    std::vector<int> bmap;
    int ntrk = 0;
    int npull = 0;
    std::vector<int> wpull;
    bool SkipSubNet;

    if (notnsqrd) {
        if (param.quiet == 0) { std::cout << "Filling in the cube vertices.\n";}
        // Fill cube vertices
        for (int d = 0; d < param.dim; ++d) {
            int numb = 0;
            for (int j = 0; j < cubeSize; j += static_cast<int>(std::pow(3, d))) {
                int block = static_cast<int>(std::pow(3, d));
                for (int k = 0; k < block; ++k) {
                    if (j + k < cubeSize)
                        cube[j + k][d] = numb;
                }
                numb = (numb + 1) % 3;
            }
        }
        // update the blocksize which may be greater than maxdisp, but which
        // keeps nblocks reasonably small.
        double volume = 1.0;
        for (int d = 0; d < param.dim; ++d) {
            double minn = std::numeric_limits<double>::max();
            double maxx = std::numeric_limits<double>::lowest();

            // Loop over indices in w to find min/max for this dimension
            for (size_t idx : w) {
                double val = xyzs[idx].coords[d];
                if (val < minn) minn = val;
                if (val > maxx) maxx = val;
            }
            volume *= (maxx - minn);
        }
        // Compute blocksize: max(maxdisp, (volume/(20*ngood))^(1/dim))
        double blocksize = std::max(maxdisp, std::pow(volume / (20.0 * ngood), 1.0 / param.dim));
        
    }

    if (param.quiet == 0) {std::cout << "Entering loop over the frames. z = " << z <<" \n";}    

    for (int i = istart; i <= z; ++i) {
        if (param.quiet == 0) {std::cout << "Frame # " << i << " out of " << z <<".\n";}    

        // span index for resx
        int ispan = (i - 1) % zspan + 1;

        // get new particle positions
        int m = static_cast<int>(res[i+1] - res[i]); // number of new particles

        // eyes = 1:m + res(i) (again, we need to remember that MATLAB uses 1-based indexing)
        std::vector<size_t> eyes(m);
        for (int j = 0; j < m; ++j) {
            eyes[j] = res[i] + j + 1;  // C++ 0-based indexing by including + 1
        }

        if (param.quiet == 0) {std::cout << "number of new particles = " << m <<"\n";}    

        if (m > 0) {
            // Extract current particle positions (xyi = xyzs(eyes,1:dim))
            std::vector<std::vector<double>> xyi(m, std::vector<double>(param.dim));
            for (int j = 0; j < m; ++j) {
                for (int d = 0; d < param.dim; ++d) {
                    xyi[j][d] = xyzs[eyes[j]].coords[d];
                }
            }
            
            std::vector<int> found(m, 0);            

            // THE TRIVIAL BOND CODE BEGINS
            if (notnsqrd) {
                if (param.quiet == 0) {std::cout << "Inside the trivial bond statement. Using raster metric to do a 1D parametrize of the space. \n";}    

                // Use the raster metric code to do trivial bonds
                // construct "s", a one dimensional parameterization of the space 
                // which consists of the d-dimensional raster scan of the volume.

                std::vector<std::vector<int>> abi(m, std::vector<int>(param.dim));
                std::vector<std::vector<int>> abpos(n, std::vector<int>(param.dim));
                for (int j = 0; j < m; ++j)
                    for (int d = 0; d < param.dim; ++d)
                        abi[j][d] = static_cast<int>(std::floor(xyi[j][d] / blocksize));
                for (int j = 0; j < n; ++j)
                    for (int d = 0; d < param.dim; ++d)
                           abpos[j][d] = static_cast<int>(std::floor(pos[j].coords[d] / blocksize));  

                std::vector<int> si(m, 0);
                std::vector<int> spos(n, 0);
                std::vector<int> dimm(param.dim, 0);
                int coff = 1;

                for (int d = 0; d < param.dim; ++d) {
                    // find min/max over combined abi(:,d) and abpos(:,d)
                    int minn = abi[0][d];
                    int maxx = abi[0][d];
                    for (int j = 0; j < m; ++j) {
                        if (abi[j][d] < minn) minn = abi[j][d];
                        if (abi[j][d] > maxx) maxx = abi[j][d];
                    }
                    for (int j = 0; j < n; ++j) {
                        if (abpos[j][d] < minn) minn = abpos[j][d];
                        if (abpos[j][d] > maxx) maxx = abpos[j][d];
                    }

                    // shift positions so min is 0
                    for (int j = 0; j < m; ++j) abi[j][d] -= minn;
                    for (int j = 0; j < n; ++j) abpos[j][d] -= minn;

                    dimm[d] = maxx - minn + 1;

                    // compute 1D raster indices
                    for (int j = 0; j < m; ++j) si[j] += abi[j][d] * coff;
                    for (int j = 0; j < n; ++j) spos[j] += abpos[j][d] * coff;

                    coff *= dimm[d];
                }
                int nblocks = coff;

                // trim down (intersect) the hypercube if its too big to fit in the particle volume
                std::vector<std::vector<int>> cub_trim = cube;
                std::vector<int> deg;
                for (int d = 0; d < param.dim; ++d)
                    if (dimm[d] < 3) deg.push_back(d);

                if (!deg.empty()) {
                    for (int k = 0; k < deg.size(); ++k) {
                        int col = deg[k];
                        std::vector<std::vector<int>> cub_filtered;
                        for (auto &row : cub_trim) {
                            if (row[col] < dimm[col]) cub_filtered.push_back(row);
                        }
                        cub_trim = cub_filtered;
                    }
                }

                // calculate the "s" coordinates of hypercube (with a corner @ the origin)
                std::vector<int> scube(cub_trim.size(), 0);
                coff = 1;
                for (int d = 0; d < param.dim; ++d) {
                    for (size_t k = 0; k < cub_trim.size(); ++k) {
                        scube[k] += cub_trim[k][d] * coff;
                    }
                    coff *= dimm[d];
                }

                // shift the hypercube "s" coordinates to be centered around the origin
                coff = 1;
                for (int d = 0; d < param.dim; ++d) {
                    if (dimm[d] > 3) {
                        for (size_t k = 0; k < scube.size(); ++k) scube[k] -= coff;
                    }
                    coff *= dimm[d];
                }

                // wrap around
                for (size_t k = 0; k < scube.size(); ++k) {
                    scube[k] = (scube[k] + nblocks) % nblocks;
                }
                
                // get the sorting for the particles by their "s" positions
                std::vector<int> isort(m);
                std::iota(isort.begin(), isort.end(), 0);
                std::sort(isort.begin(), isort.end(),
                        [&si](int a, int b) { return si[a] < si[b]; });

                std::vector<int> strt(nblocks, -1);
                std::vector<int> fnsh(nblocks, 0);

                // si == 0 
                std::vector<int> h_indices;
                for (int j = 0; j < m; ++j) if (si[j] == 0) h_indices.push_back(j);
                int lh = h_indices.size();
                if (lh>0) {
                    for (int hhh = 0; hhh <h_indices.size(); ++hhh) si[h_indices[hhh]] = 1;
                }

                // fill hash table
                for (int j = 0; j < m; ++j) {
                    int sidx = si[isort[j]]; 
                    if (strt[sidx-1] == -1) {
                        strt[sidx-1] = j;
                        fnsh[sidx-1] = j;
                    } else {
                        fnsh[sidx-1] = j;
                    }
                }

                // restore si == 0
                if (lh>0) {
                    for (int hhh = 0; hhh <h_indices.size(); ++hhh) si[h_indices[hhh]] = 0;
                }
                
                // initialize column/row totals and mapping arrays
                std::vector<int> coltot(m, 0);
                std::vector<int> rowtot(n, 0);
                std::vector<int> which1(n, 0);
                
                for (int j = 0; j < n; ++j) {

                    std::vector<int> map;
                    map.push_back(-1); // in MATLAB we had fix(-1)

                    // compute scube + spos(j)
                    std::vector<int> s(scube.size());
                    for (size_t k = 0; k < scube.size(); ++k) {
                        s[k] = ((scube[k] + spos[j]) % nblocks + nblocks) % nblocks;
                    }

                    // remove zeros
                    std::vector<int> s_nonzero;
                    for (auto val : s) if (val != 0) s_nonzero.push_back(val);
                    if (!s_nonzero.empty()) s = s_nonzero;

                    // select blocks that have particles
                    std::vector<int> stretTemp;
                    for (int sss = 0; sss<s.size(); ++sss) stretTemp.push_back(strt[s[sss]-1]);
                    std::vector<int> w;
                    for (int www = 0; www<stretTemp.size(); ++www)if (stretTemp[www] != -1) w.push_back(www);

                    int ngood = w.size();
                    if (ngood > 0) {

                        std::vector<int> s_good;
                        for (auto idx : w) s_good.push_back(s[idx]);
                            s = s_good;                        

                        // build map of candidate new particle indices
                        for (int sss = 0; sss<s.size(); ++sss) {
                            int start_idx = strt[s[sss]-1];
                            int end_idx = fnsh[s[sss]-1];
                            for (int k = start_idx; k <= end_idx; ++k) {
                                map.push_back(isort[k]);
                            }
                        }

                        // remove the initial -1
                        map.erase(map.begin());

                        // calculate squared distances to filter trivial bonds
                        std::vector<bool> ltmax(map.size(), false);
                        for (size_t idx = 0; idx < map.size(); ++idx) {
                            double dist_sq = 0.0;
                            for (int d = 0; d < param.dim; ++d) {
                                //double diff = xyi[map[idx]][d] - pos[j][d];
                                double diff = xyi[map[idx]][d] - pos[j].coords[d];
                                dist_sq += diff * diff;
                            }
                            if (dist_sq < maxdisq) ltmax[idx] = true;
                        }

                        // update rowtot, coltot, which1
                        int count = std::count(ltmax.begin(), ltmax.end(), true);
                        rowtot[j] = count;

                        if (count >= 1) {
                            for (size_t k = 0; k < map.size(); ++k) {
                                if (ltmax[k]) coltot[map[k]] += 1;
                            }
                            for (size_t k = 0; k < map.size(); ++k) {
                                if (ltmax[k]) {
                                    which1[j] = map[k];
                                    break; // take the first one
                                }
                            }
                        }
                    }
                }

                // Count how many previous particles have at least one trivial bond
                ntrk = static_cast<int>(n - std::count(rowtot.begin(), rowtot.end(), 0));

                // Find indices of rows with exactly one bond
                std::vector<size_t> w;
                for (size_t j = 0; j < rowtot.size(); ++j) {
                    if (rowtot[j] == 1) w.push_back(j);
                }
                ngood = static_cast<int>(w.size());
                
                if (ngood != 0) {
                    // Find which of these have a column with exactly one bond
                    std::vector<size_t> ww;
                    for (int idx =0; idx<w.size(); ++idx) {
                        if (coltot[which1[w[idx]]] == 1) ww.push_back(idx);
                    }
                    ngood = static_cast<int>(ww.size());
                    
                    if (ngood != 0) {
                        for (int idx = 0; idx<ww.size(); ++idx) {
                            resx[ispan][w[ww[idx]]] = eyes[which1[w[ww[idx]]]];  // update trivial bonds
                            found[which1[w[ww[idx]]]] = 1;
                            rowtot[w[ww[idx]]] = 0;
                            coltot[which1[w[ww[idx]]]] = 0;
                        }
                    }
                    
                }

                labely.clear();
                labelx.clear();

                // Check for remaining nontrivial bonds
                for (size_t j = 0; j < rowtot.size(); ++j) {
                    if (rowtot[j] > 0) labely.push_back(j);
                }
                ngood = static_cast<int>(labely.size());
                
                if (ngood != 0) {
                    for (size_t j = 0; j < coltot.size(); ++j) {
                        if (coltot[j] > 0) labelx.push_back(j);
                    }
                    nontrivial = 1;
                } else {
                    nontrivial = 0;
                }
                
                if (param.quiet == 0) {std::cout << "non-trivial flag = " << nontrivial << "\n";}    

            }
            else 
            {   
                if (param.quiet == 0) {std::cout << "Inside the trivial bond calculation via a N^2 time routine\n";}    
                // Use simple N^2 time routine to calculate trivial bonds
                // let's try a nice, loopless way!
                // don't bother tracking perm. lost guys.

                // find valid particles (pos(:,1) >= 0)
                std::vector<size_t> wh;
                for (size_t j = 0; j < pos.size(); ++j) {
                   if (pos[j].coords[0] >= 0) wh.push_back(j);
                }

                int ntrack = static_cast<int>(wh.size());
                if (ntrack == 0) {
                    std::cout << "There are no valid particles to track idiot!\n";
                    break; // exit the main loop
                }

                // Create the index matrices xmat (ntrack x m) and ymat (ntrack x m) [meshgrid like]
                std::vector<std::vector<int>> xmat(ntrack, std::vector<int>(m, 0));
                std::vector<std::vector<int>> ymat(ntrack, std::vector<int>(m, 0));
                // fill in xmat
                int count = 0;
                for (int kk = 0; kk < ntrack; ++kk) {
                    for (int ll = 0; ll < m; ++ll) {
                        xmat[kk][ll] = count++;
                    }
                    count = 0;
                }
                // fill in ymat
                for (int kk = 0; kk < ntrack; ++kk) {
                    for (int ll = 0; ll < m; ++ll) {
                        ymat[kk][ll] = kk;
                    }
                }
                int lenxn = static_cast<int>(xmat.size());        // number of rows
                int lenxm = (lenxn > 0) ? static_cast<int>(xmat[0].size()) : 0; // number of columns

                // Compute distance matrix dq
                // Initialize dq matrix: size ntrack x m
                // lenxn = number of rows to use from ymat; lenxm = number of columns to use from xmat/ymat
                std::vector<std::vector<double>> dq(lenxn, std::vector<double>(lenxm, 0.0));
                for (int d = 0; d < param.dim; ++d) {
                    
                    // Extract coordinate for this dimension
                    std::vector<double> x(m);
                    
                    for (int j = 0; j < m; ++j){
                        x[j] = xyi[j][d];
                    } 
                    
                    std::vector<double> y(ntrack);
                    for (int i = 0; i < ntrack; ++i){
                        y[i] = pos[wh[i]].coords[d];
                    }
                    
                    // Compute differences with bounds checking
                    for (int i = 0; i < lenxn; ++i) {
                        for (int j = 0; j < lenxm; ++j) {
                            int xi = xmat[i][j]; // MATLAB->C++ index
                            int yi = ymat[i][j];
                                    
                            double diff = x[xi] - y[yi];
                            if (d == 0){
                                dq[i][j] = diff * diff;
                            }else{
                                dq[i][j] += diff * diff;};
                        
                        }
                    }
                }

                labely.clear();
                labelx.clear();
                
                // Compute trivial bonds mask
                std::vector<std::vector<bool>> ltmax(ntrack, std::vector<bool>(m, false));
                for (int i = 0; i < ntrack; ++i) {
                    for (int j = 0; j < m; ++j) {
                        ltmax[i][j] = dq[i][j] < maxdisq;
                        //std::cout << "ltmax = " << ltmax[i][j] << "\n";
                    }
                }

                // Figure out which trivial bonds go with which
                // Compute rowtot (number of trivial bonds per existing particle)
                std::vector<int> rowtot(n, 0);
                for (int idx = 0; idx < ntrack; ++idx) {
                    int count = 0;
                    for (int j = 0; j < m; ++j) {
                        if (ltmax[idx][j]) ++count;
                    }
                    rowtot[wh[idx]] = count;
                }

                // Compute coltot (number of trivial bonds per new particle)
                std::vector<int> coltot(m, 0);
                if (ntrack > 1) {
                    for (int j = 0; j < m; ++j) {
                        int count = 0;
                        for (int i = 0; i < ntrack; ++i) {
                            if (ltmax[i][j]) ++count;
                        }
                        coltot[j] = count;
                    }
                } else {
                    for (int j = 0; j < m; ++j) {
                        coltot[j] = ltmax[0][j] ? 1 : 0;
                    }
                }

                // Compute which1 mapping
                std::vector<int> which1(n, -1);
                for (int idx = 0; idx < ntrack; ++idx) {
                    int best_j = -1;
                    double max_val = -1;
                    for (int j = 0; j < m; ++j) {
                        if (ltmax[idx][j] && 1 > max_val) { // MATLAB max returns first occurrence
                            max_val = 1;
                            best_j = j;
                        }
                    }
                    if (best_j != -1) {
                        which1[wh[idx]] = best_j;
                    }
                }

                // Number of tracked particles
                ntrk = n - std::count(rowtot.begin(), rowtot.end(), 0);

                // Find particles with exactly one trivial bond
                std::vector<int> w;
                for (int i = 0; i < n; ++i) {
                    if (rowtot[i] == 1) w.push_back(i);
                }
                int ngood = static_cast<int>(w.size());

                if (ngood > 0) {
                    // Keep only those for which the corresponding column has exactly one connection
                    std::vector<int> ww;
                    for (int idx : w) {
                        if (coltot[which1[idx]] == 1) ww.push_back(idx);
                    }
                    ngood = static_cast<int>(ww.size());

                    if (ngood > 0) {
                        for (int idx : ww) {
                            resx[ispan][idx] = eyes[which1[idx]];
                            found[which1[idx]] = 1;
                            rowtot[idx] = 0;
                            coltot[which1[idx]] = 0;
                        }
                    }
                }
                
                // Determine nontrivial particles
                for (int i = 0; i < n; ++i) {
                    if (rowtot[i] > 0) labely.push_back(i); 
                }
                ngood = static_cast<int>(labely.size());
                
                if (ngood != 0) {
                    for (int j = 0; j < coltot.size(); ++j) {
                        if (coltot[j] > 0) labelx.push_back(j); 
                    }
                    nontrivial = 1;
                } else {
                    nontrivial = 0;
                }
                if (param.quiet == 0) {std::cout << "non-trivial flag = "<< nontrivial << " \n";}                    
            }
            // THE TRIVIAL BOND CODE ENDS

            if (nontrivial) {
                if (param.quiet == 0) {std::cout << "Procesisng non-trivial bonds.\n";}                    

                int xdim = static_cast<int>(labelx.size());
                int ydim = static_cast<int>(labely.size());

                // make a list of the non-trivial bonds
                std::vector<std::array<int,2>> bonds; // each bond is [x_index, y_index]
                std::vector<double> bondlen;          // bond distances
                
                for (int j = 0; j < ydim; ++j) {
                    std::vector<double> distq(xdim, 0.0);
                    for (int d = 0; d < param.dim; ++d) {
                        for (int xi = 0; xi < xdim; ++xi) {
                            double diff = xyi[labelx[xi]][d] - pos[labely[j]].coords[d];
                            distq[xi] += diff * diff;
                        }
                    }

                    // find bonds below maxdisq
                    std::vector<int> w;
                    for (size_t i = 0; i < distq.size(); ++i) {
                        if (distq[i] < maxdisq) {
                            w.push_back(static_cast<int>(i)); // already 0-based
                        }
                    }
                    int ngood = static_cast<int>(w.size());

                    // In MATLAB, newb was a 2×ngood matrix where: row1 = w and row2 = (j+1) repeated ngood times; (matlab just had j)
                    std::vector<std::vector<int>> newb(2, std::vector<int>(ngood, 0));
                    // Fill first row with w
                    for (int k = 0; k < ngood; ++k) {
                        newb[0][k] = w[k];
                        newb[1][k] = j + 1;}

                    // append bondlen
                    for (int www = 0; www < w.size(); ++www){bondlen.push_back(distq[w[www]]);}

                    // find bonds 
                    for (int xi = 0; xi < newb[0].size(); ++xi) {bonds.push_back({newb[0][xi], newb[1][xi]});}
                }

                int numbonds = static_cast<int>(bonds.size());
                auto mbonds = bonds; // make a copy of bonds
                
                // Determine if we can skip the subnetwork step
                int mxsz = 0; int mysz = 0;
                if (std::max(xdim, ydim) < 4) {
                    int nclust = 1;
                    int maxsz = 0;
                    int mxsz = xdim;
                    int mysz = ydim;

                    SkipSubNet = true;

                    if (param.quiet == 0) {std::cout << "Skipped the subnet processing for small dimensions.\n";}                    
                    // instead of defining bmaps here, we just define it afterwards using the SkipSubNetFlag
                    // (Subnetwork step skipped for small dimensions)
                } else {
                    if (param.quiet == 0) {std::cout << "Cannot skip the subnet. Processing the subnet.\n";}                    

                    // THE SUBNETWORK CODE BEGINS
                    std::vector<int> lista(numbonds, 0);
                    std::vector<int> listb(numbonds, 0);
                    nclust = 0;
                    int maxsz = 0;
                    int thru = xdim;
                    // continue with the subnetwork handling

                    while (thru != 0) {
                        // the following code extracts connected
                        // sub-networks of the non-trivial bonds.  
                        // NB: lista/b can have redundant entries due to multiple-connected subnetworks
                        // extract connected sub-networks
                        std::vector<int> w;
                        for (int k = 0; k < static_cast<int>(bonds.size()); ++k) {
                            if (bonds[k][1] >= 0) w.push_back(k);
                        }

                        lista[0] = bonds[w[0]][1]; // y-index
                        listb[0] = bonds[w[0]][0]; // x-index
                        bonds[w[0]][0] = -(nclust + 1);
                        bonds[w[0]][1] = -(nclust + 1);

                        int adda = 1, addb = 1, donea = 0, doneb = 0;
                        bool finished = (donea == adda) && (doneb == addb);
                        while (!finished) {
                            // process lista
                            if (donea != adda) {
                                w.clear();
                                for (int k = 0; k < static_cast<int>(bonds.size()); ++k) {
                                    if (bonds[k][1] == lista[donea]) w.push_back(k);
                                }
                                int ngood = static_cast<int>(w.size());
                                if (ngood != 0) {
                                    for (int k = 0; k < ngood; ++k) {
                                        listb[addb + k] = bonds[w[k]][0];
                                        bonds[w[k]][0] = -(nclust + 1);
                                        bonds[w[k]][1] = -(nclust + 1);
                                    }
                                    addb += ngood;
                                }
                                donea++;
                            }

                            // process listb
                            if (doneb != addb) {
                                w.clear();
                                for (int k = 0; k < static_cast<int>(bonds.size()); ++k) {
                                    if (bonds[k][0] == listb[doneb]) w.push_back(k);
                                }
                                int ngood = static_cast<int>(w.size());
                                if (ngood != 0) {
                                    for (int k = 0; k < ngood; ++k) {
                                        lista[adda + k] = bonds[w[k]][1];
                                        bonds[w[k]][0] = -(nclust + 1);
                                        bonds[w[k]][1] = -(nclust + 1);
                                    }
                                    adda += ngood;
                                }
                                doneb++;
                            }

                            finished = (donea == adda) && (doneb == addb);
                            
                        }
                        // unique indices for x
                        std::vector<int> pqx(doneb);
                        std::iota(pqx.begin(), pqx.end(), 0);
                        std::sort(pqx.begin(), pqx.end(), [&](int a, int b){ return listb[a] < listb[b]; });

                        std::vector<double> arr_1(listb.begin(), listb.begin() + doneb);
                        std::vector<double> q_1;
                        for (int i : pqx) q_1.push_back(arr_1[i]);

                        std::vector<double> q_1_shifted = circshift(q_1,-1);
                        std::vector<double> indices_1;
                        for (int iii = 0; iii<q_1.size();++iii){
                            if (q_1[iii] != q_1_shifted[iii]){indices_1.push_back(iii);}
                        }

                        std::vector<double> unx;
                        if (indices_1.size()>0)
                            {for (int pyy=0; pyy<indices_1.size(); ++pyy){unx.push_back(pqx[indices_1[pyy]]);}}
                        else{
                            unx.push_back(q_1.size()-1);
                        }
                        int xsz = unx.size();
                        
                        // unique indices for y
                        std::vector<int> pqy(donea);
                        std::iota(pqy.begin(), pqy.end(), 0);
                        std::sort(pqy.begin(), pqy.end(), [&](int a, int b){ return lista[a] < lista[b]; });

                        std::vector<double> arr_2(lista.begin(), lista.begin() + donea);
                        std::vector<double> q_2;
                        for (int i : pqy) q_2.push_back(arr_2[i]);

                        std::vector<double> q_2_shifted = circshift(q_2,-1);
                        std::vector<double> indices_2;
                        for (int iii = 0; iii<q_2.size();++iii){
                            if (q_2[iii] != q_2_shifted[iii]){indices_2.push_back(iii);}
                        }

                        std::vector<double> uny;
                        if (indices_2.size()>0)
                            {for (int pyy=0; pyy<indices_2.size(); ++pyy){uny.push_back(pqy[indices_2[pyy]]);}}
                        else{
                            uny.push_back(q_2.size()-1);
                        }
                        int ysz = uny.size();

                        if (xsz * ysz > maxsz) {
                            maxsz = xsz * ysz;
                            mxsz = xsz;
                            mysz = ysz;
                        }

                        thru -= xsz;
                        nclust++;
                    }   

                    SkipSubNet = false;  // instea of defining bmap here, we just define it afterwards using the skipsubnet flag
                    nclust = nclust - 1; // it seems that we get an extra nclust in this case compared to matlab's case.                  
                }
                // % THE SUBNETWORK CODE ENDS. put verbose in for Jaci
                
                // Compute the map of clusters
                if (SkipSubNet){
                    bmap = std::vector<int>(bonds.size(), -1);
                }else{
                    bmap = std::vector<int>(bonds.size(), 0); // filled with zeros
                    for (size_t i = 0; i < bonds.size(); ++i){bmap[i] = bonds[i][1];}   
                }

                if (param.quiet == 0) {std::cout << "Computing permutations.\n";}    
                // THE PERMUTATION CODE BEGINS
                for (int nc = 0; nc <= nclust; ++nc) {
                    // find bonds in this cluster
                    std::vector<int> w;
                    for (size_t i = 0; i < bmap.size(); ++i) {
                        if (bmap[i] == -((nc+1)*1)) w.push_back(static_cast<int>(i));
                    }

                    // get clusterbonds (aka bonds) and lenseq
                    int nbonds = static_cast<int>(w.size());
                    std::vector<std::array<int, 2>> clusterBonds(nbonds); // bonds are redefined here by Matlab, we are using clusterBonds, as we aren't going to reuse the same variable.
                    std::vector<double> lensq(nbonds);
                    for (int i = 0; i < nbonds; ++i) {
                        clusterBonds[i] = mbonds[w[i]];
                        lensq[i] = bondlen[w[i]];
                    }

                    // Extract the first column of clusterbonds
                    std::vector<int> arr(clusterBonds.size());
                    for (size_t i = 0; i < clusterBonds.size(); ++i)
                        arr[i] = clusterBonds[i][0];

                    // Create sorting indices st (like MATLAB’s "st")
                    std::vector<int> st_1(clusterBonds.size());
                    std::iota(st_1.begin(), st_1.end(), 0);
                    std::sort(st_1.begin(), st_1.end(), [&](int a, int b) {return arr[a] < arr[b];});

                    // Create q = arr(st)
                    std::vector<int> q(st_1.size());
                    for (size_t i = 0; i < st_1.size(); ++i)
                        q[i] = arr[st_1[i]];

                    // Circular shift by -1 (i.e., shift left)
                    std::vector<int> qshift = circshift(q, -1);

                    // Find indices where elements differ
                    std::vector<int> indices_1;
                    for (int i = 0; i < q.size(); ++i) {
                        if (q[i] != qshift[i]) {
                            indices_1.push_back(static_cast<int>(i)); // store index
                        }
                    }

                    int count_1 = static_cast<int>(indices_1.size());
                    std::vector<int> un;

                    if (count_1 > 0) {
                        for (int idx = 0; idx<indices_1.size(); ++idx){
                            un.push_back(st_1[indices_1[idx]]);}
                    } else {
                        un.push_back(static_cast<int>(q.size()) - 1);
                    }  
                    
                    // uold = bonds(un,1);
                    std::vector<int> uold;
                    uold.reserve(un.size());
                    for (int idx : un) {
                        if (idx >= 0 && idx < (int)clusterBonds.size())
                            uold.push_back(clusterBonds[idx][0]);  // column 1 in MATLAB
                    }

                    // uold = length(uold);
                    int nold = static_cast<int>(uold.size());

                    // Extract the 2nd column of clusterbonds
                    std::vector<int> bonds_col2(clusterBonds.size());
                    for (size_t i = 0; i < clusterBonds.size(); ++i)
                        bonds_col2[i] = clusterBonds[i][1];

                    // Circular shift by -1 (i.e., shift left)
                    std::vector<int> bonds_col2shift = circshift(bonds_col2, -1);

                    // Find indices where elements differ
                    std::vector<int> indices_2;
                    for (int i = 0; i < q.size(); ++i) {
                        if (bonds_col2shift[i] != bonds_col2[i]) {
                            indices_2.push_back(static_cast<int>(i)); // store index
                        }
                    }

                    // count = length(indices);
                    int count_2 = static_cast<int>(indices_2.size());

                    // if count > 0 then un = indices else un = length(bonds(:,2)) - 1
                    if (count_2 > 0) {
                        un = indices_2;
                        
                    } else {
                        un.clear();
                        un.push_back(static_cast<int>(clusterBonds.size()) - 1);
                    }

                    std::vector<int> unew(un.size());
                    for (size_t i = 0; i < un.size(); ++i) unew[i] = clusterBonds[un[i]][1];
                    int nnew = static_cast<int>(unew.size());
                    unew.resize(nnew);

                    // Check combinatorics
                    if (nnew > 5) {
                        long long rnsteps = 1;
                        for (int ii = 0; ii < nnew; ++ii) {
                            int count = 0;
                            for (int b = 0; b < nbonds; ++b) {
                                if (clusterBonds[b][1] == unew[ii]) count++;
                            }
                            rnsteps *= count;
                            if (rnsteps > 50000) {
                                std::cout << "Warning: difficult combinatorics encountered.\n";
                            }
                            if (rnsteps > 200000) {
                                std::cerr << "'Excessive Combinitorics you FOOL LOOK WHAT YOU HAVE DONE TO ME!!!\n";
                                throw std::runtime_error("Excessive combinatorics encountered in track!");
                            }
                        }
                    }

                    // Initialize arrays
                    std::vector<int> stNew(nnew, 0);
                    std::vector<int> fi(nnew, 0);
                    std::vector<int> h(nbonds, 0);
                    std::vector<int> ok(nold, 1);
                    bool nlost = (nnew - nold) > 0;

                    // Fill h: map bonds(:,1) to uold index
                    for (int ii = 0; ii < nold; ++ii) {
                        for (int b = 0; b < nbonds; ++b) {
                            if (clusterBonds[b][0] == uold[ii]) h[b] = ii;
                        }
                    }

                    // Set st and fi
                    stNew[0] = 1; // MATLAB 1 maps to C++ 0
                    fi[nnew - 1] = nbonds;
                    if (nnew > 1) {
                        // Extract bonds(:,2)
                        std::vector<int> sb(nbonds);
                        for (int b = 0; b < nbonds; ++b) sb[b] = clusterBonds[b][1];

                        // Circular shifts
                        std::vector<int> sbr = circshift(sb, 1);
                        std::vector<int> sbl = circshift(sb, -1);

                        // compute st(2:end)=find( sb(2:end) ~= sbr(2:end)) + 1; => st[1:end] in C++
                        for (int sss = 1; sss<sb.size(); ++sss){
                            if (sb[sss] != sbr[sss]){stNew[sss] = (sss+1);}
                        }
                                            
                        // compute fi(1:nnew-1) = find( sb(1:nbonds-1) ~= sbl(1:nbonds-1)); => fi[0:nnew-1]
                        std::vector<int> sTemp;
                        for (int sss = 0; sss < nbonds-1; ++sss){
                            if (sb[sss] != sbl[sss]){
                                //std::cout << "sb[sss] = " << sb[sss] << ", sbl[sss] = " << sbl[sss] << "\n";
                                sTemp.push_back(sss+1);}
                        }
                        for (int fff = 0; fff<=nnew-2;++fff){fi[fff] = sTemp[fff];}                        
                    }

                    int checkflag = 0;
                    std::vector<int> minbonds; // will hold the minimal bond selection

                    while (checkflag != 2) {
                        
                        // Convert MATLAB 1-based st to 0-based C++ pt
                        std::vector<int> pt = stNew;
                        for (int i = 0; i < pt.size(); ++i) pt[i] -= 1;

                        std::vector<int> lost(nnew, 0);
                        int who = 0;
                        int losttot = 0;
                        double mndisq = nnew * maxdisq;

                        while (who != -1) {
                            if (pt[who] != fi[who]) {
                                // Find valid ok indices in h(pt(who):fi(who))
                                std::vector<int> w;
                                std::vector<int> hval;
                                std::vector<int> ok_sub;

                                w.clear();
                                
                                int start = pt[who];
                                int end = fi[who]-1;
                                for (int hhh = start; hhh<=end; ++hhh){hval.push_back(h[hhh]);}    
                                for (int ooo = 0; ooo<hval.size(); ++ooo){ok_sub.push_back(ok[hval[ooo]]);}  
                                for (int www = 0; www<ok_sub.size(); ++www){if(ok_sub[www]>0){w.push_back(www);}}

                                int ngood = static_cast<int>(w.size());
                                
                                if (ngood > 0) {                                    
                                    if (pt[who] != stNew[who]-1) {
                                        ok[h[pt[who]-1]] = 1;
                                    }
                                    pt[who] = w[0]+pt[who]+1;          // advance to first good
                                    ok[h[pt[who]-1]] = 0;

                                    if (who == nnew - 1) {    
                                        std::vector<int> ww;
                                        for (int i = 0; i < nnew; ++i) if (lost[i] == 0) ww.push_back(i);
                                    
                                        double dsq = 0.0;
                                        for (int i : ww) dsq += lensq[pt[i]-1];
                                        dsq += losttot * maxdisq;
                                        
                                        if (dsq < mndisq) {
                                            minbonds.clear();
                                            for (int mmm = 0; mmm < ww.size(); ++mmm){minbonds.push_back(pt[ww[mmm]]);}
                                            mndisq = dsq;
                                        }
                                    } else {
                                        who++;
                                    }
                                } else {
                                    if (lost[who] == 0 && (losttot != static_cast<int>(nlost))){ //if (!lost[who] && (losttot != nlost))
                                        lost[who] = 1;
                                        losttot++;
                                        if (pt[who] != stNew[who] -1) {
                                            ok[h[pt[who]-1]] = 1;
                                        }

                                        if (who == nnew - 1) {
                                            std::vector<int> ww;
                                            for (int i = 0; i < nnew; ++i) if (lost[i] == 0) ww.push_back(i);

                                            double dsq = 0.0;
                                            for (int i : ww) dsq += lensq[pt[i]-1];
                                            dsq += losttot * maxdisq;

                                            if (dsq < mndisq) {
                                                minbonds.clear();
                                                for (int mmm = 0; mmm < ww.size(); ++mmm){minbonds.push_back(pt[ww[mmm]]);}
                                                mndisq = dsq;
                                            }
                                        } else {
                                            who++;
                                        }
                                    } else {
                                        if (pt[who] != stNew[who]-1) ok[h[pt[who]-1]] = 1;
                                        pt[who] = stNew[who] - 1;
                                        if (lost[who]) {
                                            lost[who] = 0;
                                            losttot--;
                                        }
                                        who--;
                                    }
                                }
                            } else { // pt[who] == fi[who]
                                if (!lost[who] && (losttot != static_cast<int>(nlost))) {//if (lost[who] == 0 && losttot != nlost)
                                    lost[who] = 1;
                                    losttot++;
                                    if (pt[who] != stNew[who]-1) ok[h[pt[who]-1]] = 1;

                                    if (who == nnew - 1) {
                                        std::vector<int> ww;
                                        for (int i = 0; i < nnew; ++i) if (lost[i] == 0) ww.push_back(i);

                                        double dsq = 0.0;
                                        for (int i : ww) dsq += lensq[pt[i]-1];
                                        dsq += losttot * maxdisq;

                                        if (dsq < mndisq) {
                                            minbonds.clear();
                                            for (int mmm = 0; mmm < ww.size(); ++mmm){minbonds.push_back(pt[ww[mmm]]);}
                                            mndisq = dsq;
                                        }
                                    } else {
                                        who++;
                                    }
                                } else {
                                    if (pt[who] != st[who]-1) {ok[h[pt[who]-1]] = 1;}
                                    pt[who] = st[who] - 1;
                                    if (lost[who]) {
                                        lost[who] = 0;
                                        losttot--;
                                    }
                                    who--;
                                }
                            }
                        } // end inner backtracking while

                        checkflag++;
                        if (checkflag == 1) {
                            int plost = std::min(static_cast<int>(std::floor(mndisq / maxdisq)), nnew - 1);
                            if (plost > static_cast<int>(nlost)) {//(plost > nlost)
                                nlost = plost;
                            } else {
                                checkflag = 2;
                            }
                        }
                    } // end outer checkflag loop
                    for (int i = 0; i < minbonds.size(); ++i) {
                        int rowIdx = labely[clusterBonds[minbonds[i]-1][1]-1];           // MATLAB: labely(bonds(minbonds,2))
                        int colIdx = labelx[clusterBonds[minbonds[i]-1][0]];           // MATLAB: labelx(bonds(minbonds,1)+1)
                        resx[ispan][rowIdx] = eyes[colIdx];                  // assign eye value
                        found[colIdx] = 1;
                    }
                }
                //  THE PERMUTATION CODE ENDS
            }

            if (param.quiet == 0) {std::cout << "Done computing permutations.\n";}                    
            if (param.quiet == 0) {std::cout << "Assigning particle IDs for the current time step.\n";}                    

            std::vector<int> wNew;
            for (int j = 0; j < resx[ispan].size(); ++j) {
                if (resx[ispan][j] >= 0.0) {
                    wNew.push_back(j);
                }
            }

            int nww = wNew.size();
            if (nww > 0) {
                for (int i = 0; i < nww; ++i) {
                    int idx = wNew[i];
                    for (int d = 0; d < param.dim; ++d) {
                        pos[idx].coords[d] = xyzs[resx[ispan][idx]].coords[d];//pos[idx][d] = xyzs[resx[ispan][idx]][d];
                    }
                    if (param.good > 0) {
                        nvalid[idx] += 1;
                    }
                }
            }

            // Identify new particles that were not found
            std::vector<int> newguys;
            for (int i = 0; i < found.size(); ++i) {
                if (found[i] == 0) {newguys.push_back(i);
                }
            }
            int nnew = newguys.size();

            if (nnew > 0) {

                // Expand resx with -1 for new entries
                for (int row = 0; row < zspan; ++row) {
                    resx[row].insert(resx[row].end(), nnew, -1);
                }

                // Assign new particle indices for this timespan
                for (int i = 0; i < nnew; ++i) {
                    resx[ispan][n + i] = eyes[newguys[i]];
                }
                
                // Append positions of new particles
                for (int i = 0; i < nnew; ++i) {
                    Particle p;
                    p.coords.resize(param.dim);
                    for (int d = 0; d < param.dim; ++d) {
                        p.coords[d] = xyzs[eyes[newguys[i]]].coords[d];
                    }
                    pos.push_back(p);
                }

                // Update memory and unique IDs
                for (int i = 0; i < nnew; ++i) {
                    mem.push_back(0);
                    uniqid.push_back(maxid + i + 1);  // MATLAB 1-based indexing
                }
                maxid += nnew;
                
                if (param.good > 0) {
                    for (int i = 0; i < nnew; ++i) {
                        dumphash.push_back(0);
                        nvalid.push_back(1);
                    }
                }
                
                n += nnew;

            }
        } 
        else 
        {
            std::cout << "Warning - No positions found for t=" << ispan << std::endl;
        }

        // If no positions were found for this span
        std::vector<int> w;
        for (int i = 0; i < resx[ispan].size(); ++i) {
            if (resx[ispan][i] != -1) w.push_back(i);
        }
        int nok = w.size();

        if (nok != 0) {
            for (int i : w) mem[i] = 0;
        }
        
        // Increment memory for particles not found
        for (int i = 0; i < mem.size(); ++i) {
            if (resx[ispan][i] == -1) mem[i] += 1;
        }
        
        // Find lost particles
        std::vector<int> wlost;
        for (int i = 0; i < mem.size(); ++i) {
            if (mem[i] == param.mem + 1) wlost.push_back(i);
        }
        int nlost = wlost.size();

        if (nlost > 0) {
            for (int i : wlost) {
                //for (int d = 0; d < param.dim; ++d) pos[i][d] = -maxdisp;
                for (int d = 0; d < param.dim; ++d) pos[i].coords[d] = -maxdisp;
            }
            
            if (param.good > 0) {
                std::vector<int> wdump;
                for (int i = 0; i < wlost.size(); ++i) {
                    if (nvalid[wlost[i]] < param.good) wdump.push_back(i);
                }
                for (int i : wdump) {
                    dumphash[wlost[i]] = 1;
                }
            }
        
        }
        
        // If last span or last global step
        if ((ispan == zspan) || (i == z)) {
        
            int nold = bigresx[0].size();
            int nnew = n - nold;
            //std::cout << "nnew = " << nnew << " and nold = " << nold << "\n";

            if (nnew > 0) {
                for (int row = 0; row <= z; ++row)
                    bigresx[row].insert(bigresx[row].end(), nnew, -1);
            }
            
            // Goodenough filtering
            if (param.good > 0) {
                int sum_dumphash = std::accumulate(dumphash.begin(), dumphash.end(), 0);
                if (sum_dumphash > 0) {
                    std::vector<int> wkeep;
                    for (int j = 0; j < dumphash.size(); ++j) if (dumphash[j] == 0) wkeep.push_back(j);

                    int nkeep = wkeep.size();

                    // Filter all arrays
                    std::vector<std::vector<int>> resx_new(resx.size(), std::vector<int>(nkeep));
                    std::vector<std::vector<int>> bigresx_new(bigresx.size(), std::vector<int>(nkeep));
                    std::vector<Particle> pos_new;
                    std::vector<int> mem_new, uniqid_new, nvalid_new;

                    for (int j = 0; j < nkeep; ++j) {
                        int idx = wkeep[j];

                        for (int row = 0; row < resx.size(); ++row) resx_new[row][j] = resx[row][idx];
                        for (int row = 0; row < bigresx.size(); ++row) bigresx_new[row][j] = bigresx[row][idx];

                        pos_new.push_back(pos[idx]);
                        mem_new.push_back(mem[idx]);
                        uniqid_new.push_back(uniqid[idx]);
                        if (param.good > 0) nvalid_new.push_back(nvalid[idx]);
                    }

                    resx = resx_new;
                    bigresx = bigresx_new;
                    pos = pos_new;
                    mem = mem_new;
                    uniqid = uniqid_new;
                    n = nkeep;
                    dumphash.assign(nkeep, 0);
                    if (param.good > 0) nvalid = nvalid_new;
                }
            }
        
            if (param.quiet == 0) {
                std::cout << i << " of " << z << " done. Tracking " << ntrk
                        << " particles, " << n << " tracks total" << std::endl;
            }
        
            // Copy current resx into bigresx
            for (int row = 0; row <= ispan; ++row)
                for (int col = 0; col < n; ++col){
                    int idx = (i - ispan + row);
                    //std::cout << "i - ispan + row = " << idx << "\n";
                    bigresx[i - ispan + row][col] = resx[row][col];}
            
            // Reset resx
            for (int row = 0; row < zspan; ++row)
                std::fill(resx[row].begin(), resx[row].end(), -1);

            // Handle particles with -maxdisp positions
            for (int j = 0; j < pos.size(); ++j) if (pos[j].coords[0] == -maxdisp) wpull.push_back(j);
            npull = wpull.size();

            if (npull > 0) {
                std::vector<std::pair<double, double>> lillist = {{0.0, 0.0}};
                for (int ipull = 0; ipull < npull; ++ipull) {
                    std::vector<int> wpull2;
                    for (int row = 0; row < z; ++row)
                        if (bigresx[row][wpull[ipull]] != -1) wpull2.push_back(row);

                    int npull2 = wpull2.size();
                    std::vector<std::vector<int>> thing(npull2, std::vector<int>(2));
                    for (int k = 0; k < npull2; ++k) {
                        thing[k][0] = bigresx[wpull2[k]][wpull[ipull]];
                        thing[k][1] = uniqid[wpull[ipull]];
                    }

                    for (const auto& row : thing) {
                       lillist.emplace_back(static_cast<double>(row[0]), static_cast<double>(row[1]));
                    }
                }
                // Remove first row of zeros and append to olist
                olist.insert(olist.end(), lillist.begin() + 1, lillist.end());
            }
            
            // Keep only particles with valid positions
            std::vector<int> wkeep;
            for (int j = 0; j < pos.size(); ++j) if (pos[j].coords[0] >= 0) wkeep.push_back(j);
            int nkeep = wkeep.size();

            if (nkeep == 0) {
                std::cerr << "Were going to crash now, no particles...." << std::endl;
            }

            // Filter all arrays by wkeep
            std::vector<std::vector<int>> resx_new(zspan, std::vector<int>(nkeep));
            std::vector<std::vector<int>> bigresx_new(z+1, std::vector<int>(nkeep));
            std::vector<Particle> pos_new;
            std::vector<int> mem_new, uniqid_new, nvalid_new;

            for (int j = 0; j < nkeep; ++j) {
                int idx = wkeep[j];

                for (int row = 0; row < zspan; ++row) resx_new[row][j] = resx[row][idx];
                for (int row = 0; row <= z; ++row) bigresx_new[row][j] = bigresx[row][idx];

                pos_new.push_back(pos[idx]);
                mem_new.push_back(mem[idx]);
                uniqid_new.push_back(uniqid[idx]);
                if (param.good > 0) nvalid_new.push_back(nvalid[idx]);
            }

            resx = resx_new;
            bigresx = bigresx_new;
            pos = pos_new;
            mem = mem_new;
            uniqid = uniqid_new;
            n = nkeep;
            dumphash.assign(nkeep, 0);

            if (param.good > 0) nvalid = nvalid_new;
        }
    }
    
    if (param.good > 0) {
        std::vector<int> nvalid(n, 0);
        auto nvalidTemp = bigresx;
        
        for (int i = 0; i < bigresx.size(); ++i)
            for (int j = 0; j < bigresx[0].size(); ++j){
                if (bigresx[i][j] >= 0){nvalidTemp[i][j] = 1;} else{nvalidTemp[i][j] = 0;}}
        for (int nnn = 0; nnn<nvalidTemp[0].size(); ++nnn){
            for (int ppp = 0; ppp<nvalidTemp.size();++ppp){nvalid[nnn] = nvalid[nnn] + nvalidTemp[ppp][nnn];}
        }

        std::vector<int> wkeep;
        for (int j = 0; j < n; ++j)
            if (nvalid[j] >= param.good) wkeep.push_back(j);

        int nkeep = static_cast<int>(wkeep.size());
        if (nkeep == 0) {
            throw std::runtime_error(
                "You are not going any further, check your params and data"
            );
        }

        if (nkeep < n) {
            std::vector<std::vector<int>> newBigresx(bigresx.size(), std::vector<int>(nkeep));
            for (int i = 0; i < newBigresx.size(); ++i)
                for (int j = 0; j < newBigresx[0].size(); ++j)
                    newBigresx[i][j] = bigresx[i][wkeep[j]];
            bigresx = std::move(newBigresx);

            std::vector<int> newUniqid(nkeep);
            for (int j = 0; j < nkeep; ++j)
                newUniqid[j] = uniqid[wkeep[j]];
            uniqid = std::move(newUniqid);

           std::vector<Particle> newPos;
            newPos.reserve(nkeep); // reserve memory upfront
            for (int j = 0; j < nkeep; ++j) {
                newPos.push_back(pos[wkeep[j]]); // copy only the kept particles
            }
            // replace old pos with the new one
            pos = std::move(newPos);
            n = nkeep;
        }
        //std::cout << "here";
    }

    // as we wabt the IDs to start from 0, rather than 1, let's subtract 1 from all the IDs
    for (int uuu = 0; uuu<uniqid.size();++uuu){uniqid[uuu] = uniqid[uuu]-1;}

    // Compute wpull from pos(:,1) ~= -2*maxdisp
    wpull.clear();
    for (int j = 0; j < pos.size(); ++j) {
        if (pos[j].coords[0] != -2 * maxdisp) {  // "not equal" as in MATLAB
            wpull.push_back(j);
        }
    }

    npull = wpull.size();

    if (npull > 0) {
        std::vector<std::pair<double, double>> lillist = {{0.0, 0.0}};  // first dummy row
        
        for (int ipull = 0; ipull < npull; ++ipull) {
            std::vector<int> wpull2;

            // Find all rows where bigresx(row, wpull(ipull)) != -1
            for (int row = 0; row < bigresx.size(); ++row) {
                if (bigresx[row][wpull[ipull]] != -1) {
                    wpull2.push_back(row);
                }
            }
            
            int npull2 = wpull2.size();

            if (npull2 == 0) continue;

            // Build "thing" as [bigresx(wpull2, wpull(ipull)), uniqid(wpull(ipull))]
            for (int k = 0; k < npull2; ++k) {
                double first  = static_cast<double>(bigresx[wpull2[k]][wpull[ipull]]);
                double second = static_cast<double>(uniqid[wpull[ipull]]);
                lillist.emplace_back(first, second);
            }
        }

        // Append lillist(2:end,:) to olist (skip dummy row)
        if (lillist.size() > 1) {
            olist.insert(olist.end(), lillist.begin() + 1, lillist.end());
        }
    }

    // remove the intial [0,0] => olist = olist(2:end,:)
    if (!olist.empty()) {olist.erase(olist.begin());}
   
    int nolist = static_cast<int>(olist.size()); // olist is vector of pairs<int,double>
    int ndat = dd + 1; // number of columns
 
    // resFinal is a 2D matrix: nolist x (dd+1) == [x,y,z,t,ID]
    std::vector<std::vector<double>> resFinal(nolist, std::vector<double>(ndat, 0.0));
    for (int i = 0; i < nolist; ++i) {
        int idx = olist[i].first;  // MATLAB olist(:,1)
        if (idx >= 0 && idx < xyzs.size()) {
            if (dd > 0) resFinal[i][0] = xyzs[idx].coords[0];
            if (dd > 0) resFinal[i][1] = xyzs[idx].coords[1];
            if (dd > 3) resFinal[i][2] = xyzs[idx].coords[2];
        }
        resFinal[i][3] = xyzs[olist[i].first].t;
        resFinal[i][4] = olist[i].second; // MATLAB olist(:,2)
    }
    
    if (param.quiet == 0) {std::cout << "Populating final data structure.\n";}                    

    //=== Uberize: assign unique track IDs in last column
    // Copy resFinal into newtracks
    auto newtracks = resFinal;

    // circshift by -1 → compare element i vs i+1 (with wraparound)
    std::vector<int> IDVec;
    for (const auto& row : newtracks)
        IDVec.push_back(static_cast<int>(row[ndat - 1]));

    // Circular shift by -1 (i.e., shift left)
    std::vector<int> circShiftedIDs = circshift(IDVec, -1);

    // Find indices where elements differ
    std::vector<int> indices;
    for (size_t i = 0; i < IDVec.size(); ++i) {
        if (IDVec[i] != circShiftedIDs[i]) {
            indices.push_back(static_cast<int>(i)); // store index
        }
    }
    
    // Get the count
    int NewCount = static_cast<int>(indices.size());
    //std::cout << "NewCount = " << NewCount << "\n";

    std::vector<int> u;

    if (!indices.empty()) {
        // if NewCount > 0, copy indices
        u = indices;
    } else {
        // if NewCount == 0, use length(newtracks(:, ndat)) - 1
        // In C++, length of a column = number of rows = newtracks.size()
        u.push_back(static_cast<int>(newtracks.size()) - 1);
    }
    
    // number of tracks
    int ntracks = static_cast<int>(u.size());
    std::cout << "length of ntracks: " << ntracks << "\n";
    // to keep things consistent with what we have in matlab, we need u to start from 1, so let's add 1 to it
    for (int uwu = 0; uwu < u.size(); ++uwu){u[uwu] = u[uwu]+1;}

    // prepend 0 to u
    u.insert(u.begin(), 0); // equivalent to [0; u] in MATLAB

    // assign newtrack IDs
    for (int i = 1; i < u.size(); ++i) { // MATLAB loop starts from 2, so i=1 in 0-based C++
        int start = u[i-1];           // corresponds to u(i-1) in MATLAB
        int end = u[i];               // corresponds to u(i) in MATLAB
        for (int row = start; row < end; ++row) { // note: C++ end index is exclusive
            newtracks[row][ndat-1] = (i); // MATLAB uses i-1,
        }
    }

    // fill up the final results strucuture
    TrackResult FinalResults;
    // Reserve space for efficiency
    FinalResults.particles.reserve(newtracks.size());
    FinalResults.ids.reserve(newtracks.size());


    for (size_t i = 0; i < newtracks.size(); ++i) {
        Particle p;
        p.coords.resize(ndat-2);  // or at least 3 if you use x, y, z

        // Fill particle coordinates
        if (dd > 0) p.coords[0] = newtracks[i][0];
        if (dd > 0) p.coords[1] = newtracks[i][1];
        if (dd > 3) p.coords[2] = newtracks[i][2];
        
        // Fill time
        p.t = newtracks[i][ndat-2];

        // Add particle to the result
        FinalResults.particles.push_back(p);

        // Add particle ID
        FinalResults.ids.push_back((newtracks[i][ndat-1]));
        
    }
    
    return FinalResults;    

}

