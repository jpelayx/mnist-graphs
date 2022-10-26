#include "compute_features.h"

#include <iostream>
#include <cmath>
#include <unordered_set>
#include <set>
#include <utility>

cv::Mat grayscale_features(cv::Mat s, int n, cv::Mat img)
{
    cv::Mat s1 = cv::Mat::zeros(n, 1, CV_64F),
            s2 = cv::Mat::zeros(n, 1, CV_64F),
            posi1 = cv::Mat::zeros(n, 1, CV_64F),
            posj1 = cv::Mat::zeros(n, 1, CV_64F),
            posi2 = cv::Mat::zeros(n, 1, CV_64F),
            posj2 = cv::Mat::zeros(n, 1, CV_64F),
            num_pixels = cv::Mat::zeros(n, 1, CV_64F);
    int node;
    float color;
    for (int i=0; i<img.rows; i++)
        for (int j=0; j<img.cols; j++)
        {
            node = s.at<int32_t>(i, j);
            color = img.at<float>(i,j)/255;
            s1.at<double>(node,0) += color;
            s2.at<double>(node,0) += pow(color, 2);
            posi1.at<double>(node,0) += i;
            posj1.at<double>(node,0) += j;
            posi2.at<double>(node,0) += pow(i, 2);
            posj2.at<double>(node,0) += pow(j, 2);
            num_pixels.at<double>(node,0) += 1;
        }
    cv::Mat features(n, FEATURES_GRAYSCALE, CV_64F);
    // color features
    cv::divide(s1, num_pixels, s1);
    cv::divide(s2, num_pixels, s2);
    cv::Mat avg_color = s1;
    avg_color = avg_color;
    cv::Mat std_dev_color = cv::abs(s2 - s1.mul(s1));
    cv::sqrt(std_dev_color, std_dev_color);
    std_dev_color = std_dev_color;
    avg_color.copyTo(features.col(GRAY_AVG_COLOR));
    std_dev_color.copyTo(features.col(GRAY_STD_DEV_COLOR));
    // positional features
    cv::divide(posi1, num_pixels, posi1);
    cv::divide(posj1, num_pixels, posj1);
    cv::divide(posi2, num_pixels, posi2);
    cv::divide(posj2, num_pixels, posj2);
    cv::Mat centroid_i = posi1;
    cv::Mat centroid_j = posj1;
    cv::Mat std_dev_centroid_i = cv::abs(posi2 - posi1.mul(posi1));
    cv::sqrt(std_dev_centroid_i, std_dev_centroid_i);
    cv::Mat std_dev_centroid_j = cv::abs(posj2 - posj1.mul(posj1));
    cv::sqrt(std_dev_centroid_j, std_dev_centroid_j);
    centroid_i.copyTo(features.col(GRAY_CENTROID_I));
    centroid_j.copyTo(features.col(GRAY_CENTROID_J));
    std_dev_centroid_i.copyTo(features.col(GRAY_STD_DEV_CENTROID_I));
    std_dev_centroid_j.copyTo(features.col(GRAY_STD_DEV_CENTROID_J));

    num_pixels.copyTo(features.col(GRAY_NUM_PIXELS));

    return features;
}

PyArrayObject *get_edge_index(cv::Mat s)
{
    std::set<std::pair<int, int>> adj;
    int current, other;
    for (int i=0; i<s.rows; i++)
        for (int j=0; j<s.cols; j++)
        {
            current = s.at<int32_t>(i,j);
            if(i-1 >= 0 && current != s.at<int32_t>(i-1, j))
            {
                other = s.at<int32_t>(i-1, j);
                std::pair<int, int> edge = std::make_pair(std::min(current, other), std::max(current, other));
                adj.emplace(edge);
            }
            if(i+1 < s.rows && current != s.at<int32_t>(i+1, j))
            {
                other = s.at<int32_t>(i+1, j);
                std::pair<int, int> edge = std::make_pair(std::min(current, other), std::max(current, other));
                adj.emplace(edge);
            }
            if(j-1 >= 0 && current != s.at<int32_t>(i, j-1))
            {
                other = s.at<int32_t>(i, j-1);
                std::pair<int, int> edge = std::make_pair(std::min(current, other), std::max(current, other));
                adj.emplace(edge);
            }
            if(j+1 < s.cols && current != s.at<int32_t>(i, j+1))
            {
                other = s.at<int32_t>(i, j+1);
                std::pair<int, int> edge = std::make_pair(std::min(current, other), std::max(current, other));
                adj.emplace(edge);
            }
        }
    PyArrayObject *edge_index;
    int64_t dims[2];
    dims[0] = 2;
    dims[1] = adj.size();
    edge_index = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)dims, PyArray_LONG);
    if (edge_index == NULL)
        return NULL;
    int i = 0;
    for(auto edge : adj)
    {
        *((long *)PyArray_GETPTR2(edge_index, 0, i)) = edge.first;
        *((long *)PyArray_GETPTR2(edge_index, 1, i)) = edge.second;
        i++;
    }
    return edge_index;
}

cv::Mat from_numpy(PyArrayObject *a)
{
    int ndims = PyArray_NDIM(a);
    int rows = PyArray_DIM(a, 0);
    int cols = PyArray_DIM(a, 1);

    cv::Mat img;
    if(ndims == 2)
    {
        img.create(rows, cols, CV_32F);
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                img.at<float>(i, j) = *(float *)PyArray_GETPTR2(a, i, j) * 255;
    }
    else  // ndims == 3
    {
        img.create(rows, cols, CV_32FC3);
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
            {
                img.ptr<float>(i, j)[0] = *(float *)PyArray_GETPTR3(a, i, j, 0) * 255;
                img.ptr<float>(i, j)[1] = *(float *)PyArray_GETPTR3(a, i, j, 1) * 255;
                img.ptr<float>(i, j)[2] = *(float *)PyArray_GETPTR3(a, i, j, 2) * 255;
            }
    }
    return img;    
}

PyArrayObject *to_numpy_int32(cv::Mat a)
{
    PyArrayObject *x;
    int64_t x_dims[2];
    x_dims[0] = a.rows;
    x_dims[1] = a.cols;
    x = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)x_dims, PyArray_INT32);
    if(x == NULL)
        return NULL;
    for(int i=0; i<a.rows; i++)
        for(int j=0; j<a.cols; j++)
           *((int32_t *)PyArray_GETPTR2(x, i, j)) = a.at<int32_t>(i,j);
    return x;
}

PyArrayObject *to_numpy_float64(cv::Mat a)
{
    PyArrayObject *x;
    int64_t x_dims[a.rows];
    x_dims[0] = a.rows;
    x_dims[1] = a.cols;
    x = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)x_dims, PyArray_FLOAT64);
    if(x == NULL)
        return NULL;
    for(int i=0; i<a.rows; i++)
        for(int j=0; j<a.cols; j++)
           *((double *)PyArray_GETPTR2(x, i, j)) = a.at<double>(i,j);
    return x;
}

static PyObject* compute_features_color(PyObject *self, PyObject *args)
{
    PyArrayObject *img_np;
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &img_np))
        return NULL;
    
    cv::Mat img = from_numpy(img_np);

    PyArrayObject *x;
    int64_t x_dims[3];
    x_dims[0] = 10;
    x_dims[1] = 10;
    x_dims[2] = 10;
    x = (PyArrayObject *) PyArray_SimpleNew(3, (npy_intp *)x_dims, PyArray_DOUBLE);
    if (x == NULL)
        return NULL;
    
    return PyArray_Return(x);    
}

static PyObject* compute_features_gray(PyObject *self, PyObject *args)
{
    PyArrayObject *img_np;
    int n_segments;
    float compactness;
    if(!PyArg_ParseTuple(args, "O!if", &PyArray_Type, &img_np, &n_segments, &compactness))
        return NULL;

    cv::Mat img = from_numpy(img_np);
    int region_size = sqrt((img.rows*img.cols)/n_segments);
    if (region_size < 1)
        region_size = 1;
    cv::Mat s;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLIC, region_size, compactness);
    if (slic->getNumberOfSuperpixels() > 1)
    {
        slic->iterate();
        slic->enforceLabelConnectivity(50);
    }
    slic->getLabels(s);
    int n = slic->getNumberOfSuperpixels();

    cv::Mat features = grayscale_features(s, n, img);
    // std::cout << features << std::endl;

    PyArrayObject *s_np = to_numpy_int32(s); // segments 
    if (s_np == NULL)
        return NULL;
    
    PyArrayObject *features_np = to_numpy_float64(features);
    if (features_np == NULL)
        return NULL;

    PyArrayObject *edge_index = get_edge_index(s);
    if (edge_index == NULL)
        return NULL;

    return Py_BuildValue("OO", PyArray_Return(features_np), PyArray_Return(edge_index));    
}

static PyMethodDef compute_features_methods[] = {
    {"color_features", compute_features_color, METH_VARARGS, 
     "Computes features for RGB color datasets."},
    {"grayscale_features", compute_features_gray, METH_VARARGS, 
     "Computes features for grayscale datasets."}, 
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef compute_features_module = {
    PyModuleDef_HEAD_INIT, 
    "compute_features", 
    NULL, 
    -1,
    compute_features_methods
};

PyMODINIT_FUNC PyInit_compute_features(void)
{
    import_array();
    return PyModule_Create(&compute_features_module);
}