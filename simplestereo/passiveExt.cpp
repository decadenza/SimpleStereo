/* Custom implementation of
 * Adaptive Support Weight from "Locally adaptive support-weight approach
 * for visual correspondence search", K. Yoon, I. Kweon, 2005.
 * */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
 
#include <iostream>
#include <math.h>
#include "threadPool.h"

PyObject *computeASW(PyObject *self, PyObject *args)
{
    PyArrayObject *img1, *img2, *img1Lab, *img2Lab;
    int winSize, maxDisparity, minDisparity, gammaC, gammaP;
    
    // Parse input
    if (!PyArg_ParseTuple(args, "OOOOiiiii", &img1, &img2, &img1Lab, &img2Lab, 
                          &winSize, &maxDisparity, &minDisparity, &gammaC, &gammaP)){
        PyErr_SetString(PyExc_ValueError, "Invalid input format!");
        return NULL;
        }
    
    // Check input format
    if (!(PyArray_TYPE(img1) == NPY_UBYTE and PyArray_TYPE(img1) == NPY_UBYTE)){
        // Raising an exception in C is done by setting the exception object or string and then returning NULL from the function.
        // See https://docs.python.org/3/c-api/exceptions.html
        PyErr_SetString(PyExc_TypeError, "Wrong type input!");
        return NULL;
        }
    if (PyArray_NDIM(img1)!=3 or PyArray_NDIM(img1)!=PyArray_NDIM(img2) or 
        PyArray_DIM(img1,2)!=3  or PyArray_DIM(img2,2)!=3){
        PyErr_SetString(PyExc_ValueError, "Wrong image dimensions!");
        return NULL;
        }
    if (!(winSize>0 and winSize%2==1)) {
        PyErr_SetString(PyExc_ValueError, "winSize must be a positive odd number!");
        return NULL;
        }
    
    
    //Retrieve input
    int height = PyArray_DIM(img1,0);
    int width = PyArray_DIM(img1,1);
    
    // See https://numpy.org/devdocs/reference/c-api/dtype.html
    npy_ubyte *data1 = (npy_ubyte *)PyArray_DATA(img1);       // Pointer to first element (casted to right type!)
    npy_ubyte *data2 = (npy_ubyte *)PyArray_DATA(img2);       // These are 1D arrays, (f**k)!
    npy_float *dataLab1 = (npy_float *)PyArray_DATA(img1Lab); // No elegant way to see them as data[height][width][color]?
    npy_float *dataLab2 = (npy_float *)PyArray_DATA(img2Lab);
    
    // Initialize disparity map
    npy_intp disparityMapDims[2] = {height, width};
    PyArrayObject *disparityMapObj = (PyArrayObject*)PyArray_EMPTY(2, disparityMapDims, NPY_INT,0);
    npy_int *disparityMap = (npy_int *)PyArray_DATA(disparityMapObj); // Pointer to first element
    
    // Working variables
    int padding = winSize / 2;
    int i,j,x,y,d;
    float w1[winSize][winSize], w2[winSize][winSize]; // Weights
    int dBest;
    float cost, costBest, tot;
    int ii,jj,kk;
    
    // Build proximity weights matrix
    float proximityWeights[winSize][winSize];
    
    for(i = 0; i < winSize; ++i) {
        for(j = 0; j < winSize; ++j) {
            proximityWeights[i][j] = exp(-sqrt( pow(i-padding,2) + pow(j-padding,2))/gammaP);
        }
    }
    
    // Left image as reference - Main loop
    for(y=0; y < height; ++y) {         // For each row
        for(x=0; x < width; ++x) {      // For each column on left image
            
            // Pre-compute weights for left window
            for(i = 0; i < winSize; ++i) {
                for(j = 0; j < winSize; ++j) {
                    ii = std::min(std::max(y-padding+i,0),height-1); // Ensure to be within image
                    jj = std::min(std::max(x-padding+j,0),width-1);  // Replicate border value if not
                    
                    w1[i][j] = proximityWeights[i][j] * 
                               exp(-sqrt( fabs(dataLab1[3*(ii*width + jj)] - dataLab1[3*(y*width+x)]) +
                                          fabs(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+x)+1]) + 
                                          fabs(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+x)+2]) )/gammaC);
                }
            }
            
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            for(d = x-minDisparity+1; d > std::max(0,x-maxDisparity); --d) {  // For each allowed disparity (reverse order)
                
                cost = 0;   // Cost of current match
                tot = 0;    // Sum of weights
                for(i = 0; i < winSize; ++i) {
                    for(j = 0; j < winSize; ++j) {
                        ii = std::min(std::max(y-padding+i,0),height-1); // Ensure to be within image borders
                        jj = std::min(std::max(d-padding+j,0),width-1);  // Replicate border value if not
                        kk = std::min(std::max(x-padding+j,0),width-1);
                        
                        // Build weight
                        w2[i][j] = proximityWeights[i][j] * 
                                   exp(-sqrt( fabs(dataLab2[3*(ii*width + jj)] - dataLab2[3*(y*width+d)]) +
                                              fabs(dataLab2[3*(ii*width + jj)+1] - dataLab2[3*(y*width+d)+1]) + 
                                              fabs(dataLab2[3*(ii*width + jj)+2] - dataLab2[3*(y*width+d)+2]) )/gammaC);
                        
                        // Update cost
                        cost += w1[i][j]*w2[i][j]*( abs(data1[3*(ii*width + kk)] - data2[3*(ii*width + jj)]) + 
                                                    abs(data1[3*(ii*width + kk)+1] - data2[3*(ii*width + jj)+1]) + 
                                                    abs(data1[3*(ii*width + kk)+2] - data2[3*(ii*width + jj)+2]) );
                        // And denominator
                        tot += w1[i][j]*w2[i][j];
                        
                    }
                }
                
                // Weighted average
                cost = cost / tot;
                
                if(cost < costBest) {
                    costBest = cost;
                    dBest = d;
                    }
                
            }
            
            // Update disparity
            disparityMap[y*width + x] = x-dBest;
            
        }
    }
    
    
    // Cast to PyObject and return (apparently you cannot return a PyArrayObject)
    return (PyObject*)disparityMapObj;  
    
}


/*MODULE INITIALIZATION*/
static struct PyMethodDef module_methods[] = {
    {"computeASW", computeASW, METH_VARARGS, NULL},             //nameout,functionname, METH_VARARGS, NULL
    { NULL,NULL,0, NULL}
};



static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "passiveExt",
        NULL,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_passiveExt(void)
{
    PyObject *m;
    
    import_array(); //This function must be called to use Numpy C-API
    
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    
    

    return m;
}

