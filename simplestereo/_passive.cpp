/* C++ extension for stereo matching algorithms */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
 
#include <iostream>
#include <math.h>
#include <algorithm>

#include <thread>
#include "headers/safequeue.hpp"
#include "headers/colorconversion.hpp"


// ******************************** ASW ********************************
void workerASW(SafeQueue<int> &jobs, npy_ubyte *data1, npy_ubyte *data2, double *dataLab1, double *dataLab2,
              npy_int16 *disparityMap, double *proximityWeights, double gammaC,
              int width, int height, int winSize, int padding,
              int minDisparity, int maxDisparity)
{
    int dBest;
    double cost, costBest, tot;
    int ii,jj,kk;
    int i,j,y,x,d;
    double *w1 = new double[winSize*winSize]; // Weights
    double *w2 = new double[winSize*winSize];
    
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    for(x=0; x < width; ++x) {      // For each column on left image
            
            // Pre-compute weights for left window
            for(i = 0; i < winSize; ++i) {
                ii = y - padding + i;
                if( ii < 0) continue;       // Image top border
                if( ii >= height) break;    // Image bottom border
                
                for(j = 0; j < winSize; ++j) {
                    jj = x - padding + j;
                    if(jj<0) continue;
                    if(jj>=width) break;
                    
                    w1[i*winSize+j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( pow(dataLab1[3*(ii*width + jj)  ] - dataLab1[3*(y*width + x)  ],2) +
                                          pow(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width + x)+1],2) + 
                                          pow(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width + x)+2],2) )/gammaC);
                }
            }
            
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            for(d = x-minDisparity; d >= std::max(0,x-maxDisparity); --d) {  // For each allowed disparity (reverse order)
                cost = 0;   // Cost of current match
                tot = 0;    // Sum of weights
                for(i = 0; i < winSize; ++i) {
                    ii = y - padding + i;
                    if( ii < 0) continue;       // Image top border
                    if( ii >= height) break;    // Image bottom border
                    
                    for(j = 0; j < winSize; ++j) {
                        jj = d - padding + j;
                        kk = x - padding + j;
                        if(jj<0 or kk<0) continue;
                        if(jj>=width or kk>=width) break;
                        
                        // Build weight
                        w2[i*winSize+j] = proximityWeights[i*winSize + j] * 
                                   exp(-sqrt( pow(dataLab2[3*(ii*width + jj)  ] - dataLab2[3*(y*width + d)  ],2) +
                                              pow(dataLab2[3*(ii*width + jj)+1] - dataLab2[3*(y*width + d)+1],2) + 
                                              pow(dataLab2[3*(ii*width + jj)+2] - dataLab2[3*(y*width + d)+2],2) )/gammaC);
                        
                        // Update cost
                        cost += w1[i*winSize+j]*w2[i*winSize+j]*std::min( 40, abs(data1[3*(ii*width + kk)  ] - data2[3*(ii*width + jj)  ]) + 
                                                                              abs(data1[3*(ii*width + kk)+1] - data2[3*(ii*width + jj)+1]) + 
                                                                              abs(data1[3*(ii*width + kk)+2] - data2[3*(ii*width + jj)+2]) );
                        
                        // And denominator
                        tot += w1[i*winSize+j]*w2[i*winSize+j];
                        
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
    
    
    } // End of while
    
}


void workerASWconsistent(SafeQueue<int> &jobs, npy_ubyte *data1, npy_ubyte *data2, double *dataLab1, double *dataLab2,
            npy_int16 *disparityMap, double *proximityWeights, double gammaC,
            int width, int height, int winSize, int padding,
            int minDisparity, int maxDisparity)
{
    int dBest;
    double cost, costBest, tot;
    int ii,jj,kk;
    int i,j,y,x,d;
    double *w1 = new double[winSize*winSize]; // Weights
    double *w2 = new double[winSize*winSize];
    int left,right,k;
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    for(x=0; x < width; ++x) {      // For each column on left image
            
            // Pre-compute weights for left window
            for(i = 0; i < winSize; ++i) {
                ii = y-padding+i;
                if(ii<0) continue;
                if(ii>=height) break;
                for(j = 0; j < winSize; ++j) {
                    jj = x-padding+j;
                    if(jj<0) continue;
                    if(jj>=width) break;
                    w1[i*winSize+j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( pow(dataLab1[3*(ii*width + jj)  ] - dataLab1[3*(y*width+x)  ],2) +
                                          pow(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+x)+1],2) + 
                                          pow(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+x)+2],2) )/gammaC);
                }
            }
            
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            for(d = x-minDisparity; d >= std::max(0,x-maxDisparity); --d) {  // For each allowed disparity ON RIGHT (reverse order)
                
                cost = 0;   // Cost of current match
                tot = 0;    // Sum of weights
                for(i = 0; i < winSize; ++i) {
                    ii = y-padding+i;
                    if(ii<0) continue;
                    if(ii>=height) break;
                    for(j = 0; j < winSize; ++j) {
                        jj = d-padding+j;
                        kk = x-padding+j;
                        if(jj<0 or kk<0) continue;
                        if(jj>=width or kk>=width) break;
                        // Build weight
                        w2[i*winSize+j] = proximityWeights[i*winSize + j] * 
                                   exp(-sqrt( pow(dataLab2[3*(ii*width + jj)  ] - dataLab2[3*(y*width+d)  ],2) +
                                              pow(dataLab2[3*(ii*width + jj)+1] - dataLab2[3*(y*width+d)+1],2) + 
                                              pow(dataLab2[3*(ii*width + jj)+2] - dataLab2[3*(y*width+d)+2],2) )/gammaC);
                        
                        // Update cost
                        cost += w1[i*winSize+j]*w2[i*winSize+j]*std::min( 40, abs(data1[3*(ii*width + kk)] - data2[3*(ii*width + jj)]) + 
                                                    abs(data1[3*(ii*width + kk)+1] - data2[3*(ii*width + jj)+1]) + 
                                                    abs(data1[3*(ii*width + kk)+2] - data2[3*(ii*width + jj)+2]) );
                        // And denominator
                        tot += w1[i*winSize+j]*w2[i*winSize+j];
                        
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
        
    // Consistency check *****************
    for(x=0; x < width; ++x) {      // For each column on RIGHT image
            
        // Pre-compute weights for RIGHT window
        for(i = 0; i < winSize; ++i) {
            ii = y-padding+i;
            if(ii<0) continue;
            if(ii>=height) break;
            for(j = 0; j < winSize; ++j) {
                jj = x-padding+j;
                if(jj<0) continue;
                if(jj>=width) break;
                w2[i*winSize+j] = proximityWeights[i*winSize + j] * 
                           exp(-sqrt( pow(dataLab2[3*(ii*width + jj)  ] - dataLab2[3*(y*width+x)  ],2) +
                                      pow(dataLab2[3*(ii*width + jj)+1] - dataLab2[3*(y*width+x)+1],2) + 
                                      pow(dataLab2[3*(ii*width + jj)+2] - dataLab2[3*(y*width+x)+2],2) )/gammaC);
            }
        }
        
        dBest = 0;
        costBest = INFINITY; // Initialize cost to an high value
        for(d = x+minDisparity; d <= std::min(width-1,x+maxDisparity); ++d) {  // For each allowed disparity ON LEFT
            
            cost = 0;   // Cost of current match
            tot = 0;    // Sum of weights
            for(i = 0; i < winSize; ++i) {
                ii = y-padding+i;
                if(ii<0) continue;
                if(ii>=height) break;
                for(j = 0; j < winSize; ++j) {
                    jj = d-padding+j;
                    kk = x-padding+j;
                    if(jj<0 or kk<0) continue;
                    if(jj>=width or kk>=width) break;
                    // Build weight
                    w1[i*winSize+j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( pow(dataLab1[3*(ii*width + jj)  ] - dataLab1[3*(y*width+d)  ],2) +
                                          pow(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+d)+1],2) + 
                                          pow(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+d)+2],2) )/gammaC);
                    
                    // Update cost
                    cost += w1[i*winSize+j]*w2[i*winSize+j]*std::min( 40, abs(data2[3*(ii*width + kk)  ] - data1[3*(ii*width + jj)  ]) + 
                                                                          abs(data2[3*(ii*width + kk)+1] - data1[3*(ii*width + jj)+1]) + 
                                                                          abs(data2[3*(ii*width + kk)+2] - data1[3*(ii*width + jj)+2]) );
                    // And denominator
                    tot += w1[i*winSize+j]*w2[i*winSize+j];
                    
                }
            }
            
            // Weighted average
            cost = cost / tot;
            
            if(cost < costBest) {
                costBest = cost;
                dBest = d;
                }
            
        }
        
        // Update disparity map (dBest-x is the disparity, dBest is the best x coordinate on img1)
        if(disparityMap[y*width + dBest] != dBest-x) // Check if equal to first calculation
            disparityMap[y*width + dBest] = -1;       // Invalidated pixel!
        }
        
    
    // Left-Right consistency check
    // Disparity value == -1 means invalidated (occluded) pixel
    for(j=0; j < width; ++j) {
        if(disparityMap[y*width + j] == -1){
            // Find limits
            left = j-1;
            right = j+1;
            while(left>=0 and disparityMap[y*width + left] == -1){
                --left;
                }
            while(right<width and disparityMap[y*width + right] == -1){
                ++right;
                }
            // Left and right contain the first non occluded pixel in that direction
            // Ensure that we are within image limits
            // and assing valid value to occluded pixels
            if(left < 0){
                for(k=0;k<right;++k)
                    disparityMap[y*width + k] = disparityMap[y*width + right];
                }
            else if(right > width-1){
                for(k=left+1;k<width;++k)
                    disparityMap[y*width + k] = disparityMap[y*width + left];
                }
            else{
                for(k=left+1;k<right;++k)
                    disparityMap[y*width + k] = std::min(disparityMap[y*width + left],disparityMap[y*width + right]); // Set background disparity
                }
            }
        }
    
    
    }  // End of while
}



PyObject *computeASW(PyObject *self, PyObject *args)
{
    PyArrayObject *img1, *img2;
    int winSize, maxDisparity, minDisparity;
    double gammaC, gammaP;
    int consistent = 0; // Optional value
    
    // Parse input. See https://docs.python.org/3/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!O!iiidd|p", &PyArray_Type, &img1, &PyArray_Type, &img2,
                          &winSize, &maxDisparity, &minDisparity, &gammaC, &gammaP,
                          &consistent)){
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
        PyArray_DIM(img1,2)!=3 or PyArray_DIM(img2,2)!=3 or
        PyArray_DIM(img1,0)!=PyArray_DIM(img2,0) or
        PyArray_DIM(img1,1)!=PyArray_DIM(img2,1)){
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
    
    // Convert to CIELab
    ColorConversion cc;
    double *dataLab1 = new double[height*width*3];
    double *dataLab2 = new double[height*width*3];
    cc.ImageFromBGR2Lab(data1, dataLab1, width, height);
    cc.ImageFromBGR2Lab(data2, dataLab2, width, height);
    
    // Initialize disparity map
    npy_intp disparityMapDims[2] = {height, width};
    PyArrayObject *disparityMapObj = (PyArrayObject*)PyArray_EMPTY(2, disparityMapDims, NPY_INT16,0);
    npy_int16 *disparityMap = (npy_int16 *)PyArray_DATA(disparityMapObj); // Pointer to first element
    
    // Working variables
    int padding = winSize / 2;
    int i,j;
    SafeQueue<int> jobs; // Jobs queue
    int num_threads = std::thread::hardware_concurrency();
    
    
    std::thread workersArr[num_threads];
    
    // Build proximity weights matrix
    double *proximityWeights = new double[winSize*winSize];
    
    for(i = 0; i < winSize; ++i) {
        for(j = 0; j < winSize; ++j) {
            proximityWeights[i*winSize+j] = exp(-sqrt( pow(i-padding,2) + pow(j-padding,2))/gammaP);
        }
    }
    
    // TEMP
    //printf("BGR %d %d %d\n", data1[3*(90*width + 67)], data1[3*(90*width + 67)+1], data1[3*(90*width + 67)+2]);
    //printf("LAB %f %f %f\n", dataLab1[3*(90*width + 67)], dataLab1[3*(90*width + 67)+1], dataLab1[3*(90*width + 67)+2]);
            
            
    // Put each image row in queue
    for(i=0; i < height; ++i) {   
        jobs.push(i);
    }
    
    if(!consistent) {
    // Start workers
    for(i = 0; i < num_threads; ++i) {
        workersArr[i] = std::thread( workerASW, std::ref(jobs), data1, data2, dataLab1, dataLab2,
                                          disparityMap, proximityWeights, gammaC,             
                                          width, height, winSize, padding, minDisparity, maxDisparity);
        }
    } else { // If consistent mode is chosen
        
        // Start consistent workers
        for(i = 0; i < num_threads; ++i) {
            workersArr[i] = std::thread( workerASWconsistent, std::ref(jobs), data1, data2, dataLab1, dataLab2,
                                          disparityMap, proximityWeights, gammaC,             
                                          width, height, winSize, padding, minDisparity, maxDisparity);
        }
    }
    
    // Join threads
    for(i = 0; i < num_threads; ++i) {
        workersArr[i].join();
    }    
    
        
    return PyArray_Return(disparityMapObj);
}






// ******************************** GSW ********************************
void workerGSW(SafeQueue<int> &jobs, npy_ubyte *data1, npy_ubyte *data2,
            npy_int16 *disparityMap, int width, int height, int winSize, int padding,
            int minDisparity, int maxDisparity, int gamma, float fMax, int iterations, int bins)
{
    int dBest;
    float cost, costBest, temp, wBest;
    int ii,jj,kk;
    int i,j,k,y,x,d;
    int tot = winSize*winSize;
    float w[tot]; // Weights
    int center = (tot-1) / 2;
    int xx, yy;
    int left,right;
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    // USING LEFT IMAGE AS REFERENCE
    for(x=0; x < width; ++x) {      // For each column on left image
            
            /* Build geodesic map approximation*/
            /* Refer to "Distance Transformations in Digital Images", GUNILLA BORGEFORS*/
            // Weights initialization
            for(i=0;i<tot;i++){
                w[i]=INFINITY; // Set all weights to high value
                }
            w[center] = 0; // Except for the center one
            
            // Iterations
            for (d=0;d<iterations;++d){
                
                // Forward pass (row major order)
                for(i=0;i<tot;++i){                 // For every window pixel
                   yy = y-padding + i/winSize; // Whole image coordinates
                   xx = x-padding + i%winSize;
                   if(xx<0 or yy<0) continue;           //Image left border
                   if(xx>=width or yy>=height) break;  // Image right border
                   wBest = INFINITY;  
                   
                   for(k=0;k<=center;++k) // Find minimum in upper kernel
                   {
                       jj = y-padding + k/winSize; // Whole image coordinates (kernel)
                       kk = x-padding + k%winSize;
                       if(jj<0 or kk<0) continue;
                       if(jj>=height or kk>=width) break;
                       
                       // OVER THE UPPER KERNEL
                       temp = w[k] + sqrt( pow(data1[3*(yy*width + xx)  ] - data1[3*(jj*width + kk)  ],2)
                                         + pow(data1[3*(yy*width + xx)+1] - data1[3*(jj*width + kk)+1],2)
                                         + pow(data1[3*(yy*width + xx)+2] - data1[3*(jj*width + kk)+2],2) ); 
                               
                       if(temp<wBest) wBest=temp;
                       }
                   w[i] = wBest;
                   }
                   
                // Backward pass (reverse row major order)
                for(i=tot-1;i>=0;--i){                 // For every window pixel
                   yy = y-padding + i/winSize; // Whole image coordinates
                   xx = x-padding + i%winSize;
                   if(yy<0 or xx<0) continue;   
                   if(yy>=height or xx>=width) break; 
                   wBest = INFINITY;  
                   
                   for(k=center;k<tot;++k) // Find minimum in upper kernel
                   {
                       jj = y-padding + k/winSize; // Whole image coordinates (kernel)
                       kk = x-padding + k%winSize;
                       if(jj<0 or kk<0) continue;
                       if(jj>=height or kk>=width) break;
                       
                       // OVER THE LOWER KERNEL
                       temp = w[k] + sqrt( pow(data1[3*(yy*width + xx)  ] - data1[3*(jj*width + kk)  ],2)
                                         + pow(data1[3*(yy*width + xx)+1] - data1[3*(jj*width + kk)+1],2)
                                         + pow(data1[3*(yy*width + xx)+2] - data1[3*(jj*width + kk)+2],2) ); 
                       
                       if(temp<wBest) wBest=temp;
                       }
                   w[i] = wBest;
                   }
                    
                }
            
            // Convert to weights
            for(i=0;i<tot;i++){
                w[i]=exp(-w[i]/gamma);
                }
            
            // Calculate best disp
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            
            for(d = x-minDisparity; d >= std::max(0,x-maxDisparity); --d) {  // For each allowed x-coord on image 2 (reverse order)
                
                cost=0;    // Cost of current match
                
                // MUTUAL INFORMATION
                /* Needs to be implemented, but it's difficult to.
                 * OpenCV Stereo_SGBM replaced the matching cost as well. */
                 
                // SQUARED COLOR DIFFERENCES
                
                cost = 0;   // Cost of current match
                for(i = 0; i < winSize; ++i) {
                    ii = y - padding + i;
                    if( ii < 0) continue;       // Image top border
                    if( ii >= height) break;    // Image bottom border
                    
                    for(j = 0; j < winSize; ++j) {
                        kk = x - padding + j;
                        jj = d - padding + j;
                        
                        if( jj < 0 or kk < 0) continue;         // Image left border
                        if( jj >= width or kk >= width) break;  // Image right border
                        
                        
                        // Update cost
                        // Color difference is capped to fMax
                        cost += w[i*winSize + j] * std::min(fMax,
                                                            (float) sqrt(pow(data1[3*(ii*width + kk)]   - data2[3*(ii*width + jj)  ], 2)
                                                                       + pow(data1[3*(ii*width + kk)+1] - data2[3*(ii*width + jj)+1], 2)
                                                                       + pow(data1[3*(ii*width + kk)+2] - data2[3*(ii*width + jj)+2], 2)) );
                        
                        
                    }
                }
                
                
                if(cost < costBest) {
                    costBest = cost;
                    dBest = d;
                    }
                
            }
            
            // Update disparity
            disparityMap[y*width + x] = x-dBest;
            
        }
    
    // USING RIGHT IMAGE AS REFERENCE
    for(x=0; x < width; ++x) {      // For each column on left image
            
            /* Build geodesic map approximation, managing border areas too */
            // Weights initialization
            for(i=0;i<tot;i++){
                w[i]=INFINITY; // Set all weights to high value
                }
            w[center] = 0; // Except for the center one
            
            // Iterations
            for (d=0;d<iterations;++d){
                
                // Forward pass (row major order)
                for(i=0;i<tot;++i){                 // For every window pixel
                   yy = y-padding + i/winSize; // Whole image coordinates
                   if(yy<0 or yy>=height) continue;   //Image y border
                   xx = x-padding + i%winSize;
                   if(xx<0 or xx>=width) continue;  // Image x border
                   wBest = INFINITY;  
                   
                   for(k=0;k<=center;++k) // Find minimum in upper kernel
                   {
                       jj = y-padding + k/winSize; // Whole image coordinates (kernel)
                       if(jj<0 or jj>=height) continue;
                       kk = x-padding + k%winSize;
                       if(kk<0 or kk>=width) continue;
                       
                       // OVER THE UPPER KERNEL
                       temp = w[k] + sqrt( pow(data2[3*(yy*width + xx)  ] - data2[3*(jj*width + kk)  ],2)
                                         + pow(data2[3*(yy*width + xx)+1] - data2[3*(jj*width + kk)+1],2)
                                         + pow(data2[3*(yy*width + xx)+2] - data2[3*(jj*width + kk)+2],2) ); 
                               
                       if(temp<wBest) wBest=temp;
                       }
                   w[i] = wBest;
                   }
                   
                // Backward pass (reverse row major order)
                for(i=tot-1;i>=0;--i){                 // For every window pixel
                   yy = y-padding + i/winSize; // Whole image coordinates
                   if(yy<0 or yy>=height) continue;   //Image y border
                   xx = x-padding + i%winSize;
                   if(xx<0 or xx>=width) continue;  // Image x border
                   wBest = INFINITY;  
                   
                   for(k=center;k<tot;++k) // Find minimum in upper kernel
                   {
                       jj = y-padding + k/winSize; // Whole image coordinates (kernel)
                       if(jj<0 or jj>=height) continue;
                       kk = x-padding + k%winSize;
                       if(kk<0 or kk>=width) continue;
                       
                       // OVER THE LOWER KERNEL
                       temp = w[k] + sqrt( pow(data2[3*(yy*width + xx)  ] - data2[3*(jj*width + kk)  ],2)
                                         + pow(data2[3*(yy*width + xx)+1] - data2[3*(jj*width + kk)+1],2)
                                         + pow(data2[3*(yy*width + xx)+2] - data2[3*(jj*width + kk)+2],2) ); 
                       
                       if(temp<wBest) wBest=temp;
                       }
                   w[i] = wBest;
                   }
                    
                }
            
            // Convert to weights
            for(i=0;i<tot;i++){
                w[i]=exp(-w[i]/gamma);
                }
            
            // Calculate best disp
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            
            for(d = x+minDisparity; d <= std::min(width-1,x+maxDisparity); ++d) {  // For each allowed disparity ON LEFT    
                cost=0;    // Cost of current match
                
                // SQUARED COLOR DIFFERENCES
                cost = 0;   // Cost of current match
                for(i = 0; i < winSize; ++i) {
                    ii = y - padding + i;
                    if( ii < 0) continue;       // Image top border
                    if( ii >= height) break;    // Image bottom border
                    
                    for(j = 0; j < winSize; ++j) {
                        kk = x - padding + j;
                        jj = d - padding + j;
                        
                        if( jj < 0 or kk < 0) continue;         // Image left border
                        if( jj >= width or kk >= width) break;  // Image right border
                        
                        
                        // Update cost
                        // Color difference is capped to fMax
                        cost += w[i*winSize + j] * std::min(fMax,
                                                            (float) sqrt(pow(data2[3*(ii*width + kk)]   - data1[3*(ii*width + jj)  ], 2)
                                                               + pow(data2[3*(ii*width + kk)+1] - data1[3*(ii*width + jj)+1], 2)
                                                               + pow(data2[3*(ii*width + kk)+2] - data1[3*(ii*width + jj)+2], 2)) );
                        
                        
                    }
                }
                
                
                if(cost < costBest) {
                    costBest = cost;
                    dBest = d;
                    }
                
            }
            
        // Update disparity map (dBest-x is the disparity, dBest is the best x coordinate on img1)
        if(disparityMap[y*width + dBest] != dBest-x) // Check if equal to first calculation
            disparityMap[y*width + dBest] = -1;       // Invalidated pixel!
            
        }
    
    // Left-Right consistency check
    // Disparity value == -1 means invalidated (occluded) pixel
    for(j=0; j < width; ++j) {
        if(disparityMap[y*width + j] == -1){
            // Find limits
            left = j-1;
            right = j+1;
            while(left>=0 and disparityMap[y*width + left] == -1){
                --left;
                }
            while(right<width and disparityMap[y*width + right] == -1){
                ++right;
                }
            // Left and right contain the first non occluded pixel in that direction
            // Ensure that we are within image limits
            // and assing valid value to occluded pixels
            if(left < 0){
                for(k=0;k<right;++k)
                    disparityMap[y*width + k] = disparityMap[y*width + right];
                }
            else if(right > width-1){
                for(k=left+1;k<width;++k)
                    disparityMap[y*width + k] = disparityMap[y*width + left];
                }
            else{
                for(k=left+1;k<right;++k)
                    disparityMap[y*width + k] = std::min(disparityMap[y*width + left],disparityMap[y*width + right]);
                }
            }
        }
        
    } // End of while
    
}


PyObject *computeGSW(PyObject *self, PyObject *args)
{
    PyArrayObject *img1, *img2;
    int winSize, maxDisparity, minDisparity, gamma, iterations, bins;
    float fMax;
    // Parse input. See https://docs.python.org/3/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!O!iiiifii", &PyArray_Type, &img1, &PyArray_Type, &img2, 
                          &winSize, &maxDisparity, &minDisparity, &gamma,
                          &fMax, &iterations, &bins)){
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
        PyArray_DIM(img1,2)!=3 or PyArray_DIM(img2,2)!=3 or
        PyArray_DIM(img1,0)!=PyArray_DIM(img2,0) or
        PyArray_DIM(img1,1)!=PyArray_DIM(img2,1)){
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
    
    // Initialize disparity map
    npy_intp disparityMapDims[2] = {height, width};
    PyArrayObject *disparityMapObj = (PyArrayObject*)PyArray_EMPTY(2, disparityMapDims, NPY_INT16, 0);
    npy_int16 *disparityMap = (npy_int16 *)PyArray_DATA(disparityMapObj); // Pointer to first element
    
    // Working variables
    int padding = winSize / 2;
    int i;
    SafeQueue<int> jobs; // Jobs queue
    int num_threads = std::thread::hardware_concurrency();
    
    std::thread workersArr[num_threads];
    
    
    // Put each image row in queue
    for(i=0; i < height; ++i) {   
        jobs.push(i);
    }
    
    for(i = 0; i < num_threads; ++i) {
        workersArr[i] = std::thread( workerGSW, std::ref(jobs), data1, data2,
                                          disparityMap, width, height, winSize,
                                          padding, minDisparity, maxDisparity, gamma, fMax, iterations, bins);
    }
    
    // Join threads
    for(i = 0; i < num_threads; ++i) {
        workersArr[i].join();
    }    
    
    return PyArray_Return(disparityMapObj);
}







/*____________________PYTHON MODULE INITIALIZATION____________________*/
static struct PyMethodDef module_methods[] = {
    /* {name (external), function, calling, doc} */
    {"computeASW",  computeASW, METH_VARARGS, NULL},
    {"computeGSW",  computeGSW, METH_VARARGS, NULL},
    {NULL,NULL,0, NULL}
};



static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_passive",
        NULL,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__passive(void)
{
    PyObject *m;
    
    import_array(); //This function must be called to use Numpy C-API
    
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    
    

    return m;
}

