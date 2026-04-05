#include "test.h"

using namespace cv;
using namespace std;
using namespace umnholo;

int test_main(int argc, char* argv[])
{
    cout << "Running build tests" << endl;

    size_t initial_avail_mem, initial_total_mem;
    size_t final_avail_mem, final_total_mem;
    cudaMemGetInfo(&initial_avail_mem, &initial_total_mem);
    
    CHECK_MEMORY("begining of tests");
    
    // This region is for tests to run out of order for faster debugging
    if (!test_compressive_holo())
    {
        cerr << "Error testing compressive_holo.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_compressive_holo");
    
    if (!test_sparse_compressive_holo())
    {
        cerr << "Error testing sparse_compressive_holo.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_sparse_compressive_holo");

    //if (!test_complete());
    {
    //cerr << "Complete tests failed. Running unit tests" << endl;

    if (!test_cumat())
    {
        cerr << "Error testing cumat.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_cumat");

    if (!test_hologram())
    {
        cerr << "Error testing hologram.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_hologram");

    if (!test_focus_metric())
    {
        cerr << "Error testing focus_metric.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_focus_metric");

    if (!test_point_cloud())
    {
        cerr << "Error testing point_cloud.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_point_cloud");

    if (!test_holo_sequence())
    {
        cerr << "Error testing holo_sequence.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_holo_sequence");

    if (!test_optical_field())
    {
        cerr << "Error testing optical_field.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_optical_field");

    if (!test_reconstruction())
    {
        cerr << "Error testing reconstruction.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_reconstruction");

    if (!test_deconvolution())
    {
        cerr << "Error testing deconvolution.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_deconvolution");
    
    if (!test_blob3d())
    {
        cerr << "Error testing blob3d.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_blob3d.h");

    if (!test_object_cloud())
    {
        cerr << "Error testing object_cloud.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_object_cloud.h");

    if (!test_particle_extraction())
    {
        cerr << "Error testing particle_extraction.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_particle_extraction");
    
    if (!test_compressive_holo())
    {
        cerr << "Error testing compressive_holo.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_compressive_holo");
    
    if (!test_sparse_compressive_holo())
    {
        cerr << "Error testing sparse_compressive_holo.h" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_sparse_compressive_holo");

    if (!test_complete())
    {
        cerr << "Error running complete tests\n" << endl;
        return -1;
    }
    CHECK_MEMORY("after test_complete");

    } // first test_complete

    cout << "Build tests successful" << endl;

    cudaMemGetInfo(&final_avail_mem, &final_total_mem);

    cout << "  initial memory: " << initial_avail_mem << endl;
    cout << "  finally memory: " << final_avail_mem << endl;
    return 0;
}

// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B)
{
    if ((A.rows != B.rows) || (A.cols != B.cols) || (A.type() != B.type()))
    {
        cerr << "Inputs to getSimilarity are of different sizes" << endl;
        cerr << " A: rows = " << A.rows << ", cols = " << A.cols << ", type = " << A.type() << endl;
        cerr << " B: rows = " << B.rows << ", cols = " << B.cols << ", type = " << B.type() << endl;
        return 100000000.0;
    }

    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calculate the L2 relative error between images.
        double errorL2 = norm(A, B, NORM_L2);// / norm(A, NORM_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        //Images have a different size
        return 100000000.0;  // Return a bad value
    }
}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

double maxDiffFraction(Mat real, Mat test, Point* max_loc = NULL)
{
    double max_diff, min_diff;
    Mat diff = abs(real - test);
    minMaxLoc(diff, &min_diff, &max_diff);

    // Differences less than FLT_EPSILON could be cause by floating point
    // representation error
    if (max_diff > FLT_EPSILON)
    {
        // Get relative difference, normalize using mean of non-zeros
        //Mat mask = real != 0;
        //diff = diff / mean(abs(real), mask).val[0];
        diff = diff / mean(abs(real)).val[0];
        Point min_loc;
        if (max_loc == NULL)
            minMaxLoc(diff, &min_diff, &max_diff);
        else
            minMaxLoc(diff, &min_diff, &max_diff, &min_loc, max_loc);
    }

    return max_diff;
}

int countDiffFraction(Mat real, Mat test, double thr)
{
    Mat diff = abs(real - test);
    diff = diff / mean(abs(real)).val[0];

    cv::threshold(diff, diff, thr, 1, cv::THRESH_BINARY);
    int count = cv::sum(diff).val[0];

    //imshow("Thresholded Difference", diff);

    return count;
}

bool compareData(CuMat my_data, CuMat true_data, double thr, int count_thr, bool verbose)
{
    Mat my_mat = my_data.getReal().getMatData();
    Mat true_mat = true_data.getReal().getMatData();
    
    return compareData(my_mat, true_mat, thr, count_thr, verbose);
}

bool compareData(Mat my_mat, Mat true_mat, double thr, int count_thr, bool verbose)
{
    if (my_mat.empty())
    {
        cout << "Failed!" << endl;
        cout << "User matrix is empty!" << endl;
        return false;
    }
    if (true_mat.empty())
    {
        cout << "Failed!" << endl;
        cout << "Ground truth matrix is empty!" << endl;
        return false;
    }
    
    if (my_mat.channels() > 1)
    {
        cout << "Failed!" << endl;
        cout << "User matrix cannot be complex" << endl;
        return false;
    }
    
    // Make sure that they are the same size and type before continuing
    if ((true_mat.rows != my_mat.rows) || 
            (true_mat.cols != my_mat.cols) || 
            (true_mat.type() != my_mat.type()))
    {
        cout << "Failed!" << endl;
        cout << "User matrix is not of same size and type as input" << endl;
        cout << "True rows: " << true_mat.rows << 
                ", cols: " << true_mat.cols << 
                ", type: " << type2str(true_mat.type()) << endl;
        cout << "My rows: " << my_mat.rows << 
                ", cols: " << my_mat.cols << 
                ", type: " << type2str(my_mat.type()) << endl;
        cout << "True: " << endl << true_mat(Rect(0,0,5,5)) << endl;
        return false;
    }
    
    // Check for nan data
    bool quiet = true;
    Point bad_point;
    if (!cv::checkRange(my_mat, quiet, &bad_point))
    {
        cout << "Failed!" << endl;
        cout << "User matrix contains NaN or infinite values" << endl;
        cout << "Location of first bad value is " << bad_point << endl;
        cout << "True value: " << true_mat.at<float>(bad_point) << endl;
        cout << "My value: " << my_mat.at<float>(bad_point) << endl;
        return false;
    }

    //double diff = getSimilarity(true_mat, my_mat);
    Point max_loc;
    double diff = maxDiffFraction(true_mat, my_mat, &max_loc);
    int err_count = countDiffFraction(true_mat, my_mat, thr);
    cout << " diff = " << abs(diff) << ", err = " << err_count << " ";
    if ((abs(diff) > thr) && (err_count > count_thr))
    {
        if (verbose)
        {
            cout << "Failed!" << endl;
            cout << "Max difference is " << diff*100 << "%" << endl;
            cout << "Number of elements with difference greater than " << thr;
            double err_percent = 100*(double)err_count / (double)(true_mat.rows*true_mat.cols);
            cout << ": " << err_count << " (" << err_percent << "%)" << endl;
            cout << "  at " << max_loc <<
                ", true = " << true_mat.at<float>(max_loc) <<
                ", mine = " << my_mat.at<float>(max_loc) << endl;
            cout << "max_loc.x = " << max_loc.x << ", max_loc.y = " << max_loc.y << endl;
            max_loc.x += 3;
            max_loc.y += 3;
            
            int roi_size = 9;
            if ((true_mat.rows > roi_size) && (true_mat.cols > roi_size))
            {
                Rect roi;
                if ((max_loc.x > roi_size/2 - 1) && (max_loc.y > roi_size/2 - 1) && 
                        (max_loc.x < true_mat.cols-(roi_size/2)) && 
                        (max_loc.y < true_mat.rows-(roi_size/2)))
                    roi = Rect(max_loc.x-(roi_size/2), max_loc.y-(roi_size/2), roi_size, roi_size);
                else
                    roi = Rect(0, 0, roi_size, roi_size);
                
                if (max_loc.x <= (roi_size/2)-1) roi.x = 0;
                else if (max_loc.x >= true_mat.cols-(roi_size/2)) roi.x = true_mat.cols-roi_size;
                else roi.x = max_loc.x-(roi_size/2);
                if (max_loc.y <= (roi_size/2)-1) roi.y = 0;
                else if (max_loc.y >= true_mat.rows-(roi_size/2)) roi.y = true_mat.rows-roi_size;
                else roi.y = max_loc.y-(roi_size/2);
                
                cout << "ROI: " << roi << endl;
                cout << "true:" << endl << true_mat(roi) << endl;
                cout << "mine:" << endl << my_mat(roi) << endl;

                // If mean is high, assume it should be displayed as uint8
                if (mean(true_mat).val[0] > 10)
                {
                    true_mat.convertTo(true_mat, CV_8U);
                    my_mat.convertTo(my_mat, CV_8U);
                }
                
                //imshow("True", true_mat);
                //imshow("Mine", my_mat);
                
                //cv::normalize(true_mat, true_mat, 0, 1, CV_MINMAX);
                //cv::normalize(my_mat, my_mat, 0, 1, CV_MINMAX);
                //imshow("Scaled True", true_mat);
                //imshow("Scaled Mine", my_mat);
                //waitKey(0);
            }
            else
            {
                Rect roi(0, 0, roi_size, roi_size);
                roi.width = min(roi_size, true_mat.cols);
                roi.height = min(roi_size, true_mat.rows);
                roi.x = min(max_loc.x, true_mat.cols - roi.width);
                roi.y = min(max_loc.y, true_mat.rows - roi.height);
                cout << "roi = " << roi << endl;
                cout << "true:" << endl << true_mat(roi) << endl;
                cout << "mine:" << endl << my_mat(roi) << endl;
            }
        }

        return false;
    }

    return true;
}

bool compareDataSparse(Mat my_mat, Mat true_mat, double thr, double count_thr, bool verbose)
{
    // Make sure that they are the same size and type before continuing
    if ((true_mat.rows != my_mat.rows) || 
            (true_mat.cols != my_mat.cols) || 
            (true_mat.type() != my_mat.type()))
    {
        cout << "Failed!" << endl;
        cout << "User matrix is not of same size and type as input" << endl;
        cout << "True rows: " << true_mat.rows << 
                ", cols: " << true_mat.cols << 
                ", type: " << type2str(true_mat.type()) << endl;
        cout << "My rows: " << my_mat.rows << 
                ", cols: " << my_mat.cols << 
                ", type: " << type2str(my_mat.type()) << endl;
        cout << "True: " << endl << true_mat(Rect(0,0,5,5)) << endl;
        return false;
    }
    
    // Check for nan data
    bool quiet = true;
    Point bad_point;
    if (!cv::checkRange(my_mat, quiet, &bad_point))
    {
        cout << "Failed!" << endl;
        cout << "User matrix contains NaN or infinite values" << endl;
        cout << "Location of first bad value is " << bad_point << endl;
        cout << "True value: " << true_mat.at<float>(bad_point) << endl;
        cout << "My value: " << my_mat.at<float>(bad_point) << endl;
        return false;
    }
    
    int nnz_mine = cv::countNonZero(my_mat);
    int nnz_true = cv::countNonZero(true_mat);
    double err = abs(nnz_true - nnz_mine) / (double)nnz_true;
    cout << endl << "nnz_mine = " << nnz_mine << ", nnz_true = " << nnz_true << ", err = " << err << endl;
    if (err > 1e-3)
    {
        cout << "Failed!" << endl;
        cout << "Number of non-zero elements does not match" << endl;
        cout << "True matrix has " << nnz_true << endl;
        cout << "Mine has " << nnz_mine << endl;
        return false;
    }
    
    Mat temp_true_mat;
    Mat temp_my_mat;
    threshold(true_mat, temp_true_mat, FLT_EPSILON, 1, THRESH_TOZERO);
    threshold(my_mat, temp_my_mat, FLT_EPSILON, 1, THRESH_TOZERO);
    
    Mat diff = abs(temp_true_mat - temp_my_mat);
    Mat mask = temp_true_mat != 0;
    divide(diff, true_mat, diff);
    threshold(diff, diff, FLT_EPSILON, 1, THRESH_TOZERO);
    double max_diff, min_diff;
    Point max_loc;
    minMaxLoc(diff, &min_diff, &max_diff, NULL, &max_loc, mask);
    
    cv::threshold(diff, diff, thr, 1, cv::THRESH_BINARY);
    int err_count = cv::countNonZero(diff);
    double err_fraction = (double)err_count / (double)nnz_true;
    
    cout << " diff = " << abs(max_diff) << ", err = " << err_count << " ";
    if ((abs(max_diff) > thr) && (err_fraction > count_thr))
    {
        if (verbose)
        {
            cout << "Failed!" << endl;
            cout << "Epsilon = " << FLT_EPSILON << endl;
            cout << "Max difference is " << max_diff*100 << "%" << endl;
            cout << "  at " << max_loc <<
                ", true = " << true_mat.at<float>(max_loc) <<
                ", mine = " << my_mat.at<float>(max_loc) << endl;
            cout << "max_loc.x = " << max_loc.x << ", max_loc.y = " << max_loc.y << endl;
            max_loc.x += 3;
            max_loc.y += 3;
            
            int roi_size = 9;
            if ((true_mat.rows > roi_size) && (true_mat.cols > roi_size))
            {
                Rect roi;
                if ((max_loc.x > roi_size/2 - 1) && (max_loc.y > roi_size/2 - 1) && 
                        (max_loc.x < true_mat.cols-(roi_size/2)) && 
                        (max_loc.y < true_mat.rows-(roi_size/2)))
                    roi = Rect(max_loc.x-(roi_size/2), max_loc.y-(roi_size/2), roi_size, roi_size);
                else
                    roi = Rect(0, 0, roi_size, roi_size);
                
                if (max_loc.x <= (roi_size/2)-1) roi.x = 0;
                else if (max_loc.x >= true_mat.cols-(roi_size/2)) roi.x = true_mat.cols-roi_size;
                else roi.x = max_loc.x-(roi_size/2);
                if (max_loc.y <= (roi_size/2)-1) roi.y = 0;
                else if (max_loc.y >= true_mat.rows-(roi_size/2)) roi.y = true_mat.rows-roi_size;
                else roi.y = max_loc.y-(roi_size/2);
                
                cout << "ROI: " << roi << endl;
                cout << "true:" << endl << true_mat(roi) << endl;
                cout << "mine:" << endl << my_mat(roi) << endl;

                // If mean is high, assume it should be displayed as uint8
                if (mean(true_mat).val[0] > 10)
                {
                    true_mat.convertTo(true_mat, CV_8U);
                    my_mat.convertTo(my_mat, CV_8U);
                }
                
                //imshow("True", true_mat);
                //imshow("Mine", my_mat);
                
                //cv::normalize(true_mat, true_mat, 0, 1, CV_MINMAX);
                //cv::normalize(my_mat, my_mat, 0, 1, CV_MINMAX);
                //imshow("Scaled True", true_mat);
                //imshow("Scaled Mine", my_mat);
                //waitKey(0);
            }
            else
            {
                Rect roi(0, 0, roi_size, roi_size);
                roi.width = min(roi_size, true_mat.cols);
                roi.height = min(roi_size, true_mat.rows);
                roi.x = min(max_loc.x, true_mat.cols - roi.width);
                roi.y = min(max_loc.y, true_mat.rows - roi.height);
                cout << "roi = " << roi << endl;
                cout << "true:" << endl << true_mat(roi) << endl;
                cout << "mine:" << endl << my_mat(roi) << endl;
            }
        }

        return false;
    }

    return true;
}

bool compareFile(CuMat my_data, char* filename, double thr, int count_thr, bool verbose)
{
    CuMat true_data;
    true_data.load(filename);

    Mat true_mat = true_data.getMatData();
    Mat my_mat = my_data.getReal().getMatData();
    
    bool result = compareData(my_mat, true_mat, thr, count_thr, verbose);
    if (!result) cout << "User data does not match with file " << filename << endl;
    return result;
}

bool compareFileSparse(CuMat my_data, char* filename, double thr, double count_thr, bool verbose)
{
    CuMat true_data;
    true_data.load(filename);

    Mat true_mat = true_data.getMatData();
    Mat my_mat = my_data.getReal().getMatData();
    
    bool result = compareDataSparse(my_mat, true_mat, thr, count_thr, verbose);
    if (!result) cout << "User data does not match with file " << filename << endl;
    return result;
}

bool data_exists(Parameters params)
{
    // Open the first image and make sure it is there
    char filename[FILENAME_MAX];
    sprintf(filename, params.image_filename, params.start_image);
    FILE* fid = fopen(filename, "r");
    if (fid == NULL)
    {
        return false;
    }
    fclose(fid);

    return true;
}
