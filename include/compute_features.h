#include <Python.h>
#include <numpy/arrayobject.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc/slic.hpp>

// grayscale features
#define GRAY_AVG_COLOR 0
#define GRAY_STD_DEV_COLOR 1
#define GRAY_CENTROID_I 2
#define GRAY_CENTROID_J 3
#define GRAY_STD_DEV_CENTROID_I 4
#define GRAY_STD_DEV_CENTROID_J 5
#define GRAY_NUM_PIXELS 6

#define FEATURES_GRAYSCALE 7

PyArrayObject *get_edge_index(cv::Mat s);

cv::Mat grayscale_features(cv::Mat s, int n, cv::Mat img);

cv::Mat from_numpy(PyArrayObject *a);
PyArrayObject *to_numpy_int32(cv::Mat a);
PyArrayObject *to_numpy_float64(cv::Mat a);
