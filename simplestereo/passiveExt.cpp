/* Custom implementation of
 * Adaptive Support Weight from "Locally adaptive support-weight approach
 * for visual correspondence search", K. Yoon, I. Kweon, 2005.
 * */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
 
#include <iostream>
#include <math.h>
#include <algorithm>

#include <thread>
#include "headers/SafeQueue.hpp"


// ******************************** ASW ********************************
void workerASW(SafeQueue<int> &jobs, npy_ubyte *data1, npy_ubyte *data2, npy_float *dataLab1, npy_float *dataLab2,
            npy_int16 *disparityMap, float *proximityWeights, int gammaC,
            int width, int height, int winSize, int padding,
            int minDisparity, int maxDisparity)
{
    int dBest;
    float cost, costBest, tot;
    int ii,jj,kk;
    int i,j,y,x,d;
    float w1[winSize][winSize], w2[winSize][winSize]; // Weights
    
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    for(x=0; x < width; ++x) {      // For each column on left image
            
            // Pre-compute weights for left window
            for(i = 0; i < winSize; ++i) {
                ii = std::min(std::max(y-padding+i,0),height-1); // Ensure to be within image, replicate border if not
                for(j = 0; j < winSize; ++j) {
                    jj = std::min(std::max(x-padding+j,0),width-1);
                    
                    w1[i][j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( fabs(dataLab1[3*(ii*width + jj)] - dataLab1[3*(y*width+x)]) +
                                          fabs(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+x)+1]) + 
                                          fabs(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+x)+2]) )/gammaC);
                }
            }
            
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            for(d = x-minDisparity; d >= std::max(0,x-maxDisparity); --d) {  // For each allowed disparity (reverse order)
                
                cost = 0;   // Cost of current match
                tot = 0;    // Sum of weights
                for(i = 0; i < winSize; ++i) {
                    ii = std::min(std::max(y-padding+i,0),height-1);
                    for(j = 0; j < winSize; ++j) {
                        jj = std::min(std::max(d-padding+j,0),width-1);
                        kk = std::min(std::max(x-padding+j,0),width-1);
                        
                        // Build weight
                        w2[i][j] = proximityWeights[i*winSize + j] * 
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
    
    
    } // End of while
    
}


void workerASWconsistent(SafeQueue<int> &jobs, npy_ubyte *data1, npy_ubyte *data2, npy_float *dataLab1, npy_float *dataLab2,
            npy_int16 *disparityMap, float *proximityWeights, int gammaC,
            int width, int height, int winSize, int padding,
            int minDisparity, int maxDisparity)
{
    int dBest;
    float cost, costBest, tot;
    int ii,jj,kk;
    int i,j,y,x,d;
    float w1[winSize][winSize], w2[winSize][winSize]; // Weights
    int left,right,k;
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    // TEMP
    //printf("Working on %d\n", y);
    
    for(x=0; x < width; ++x) {      // For each column on left image
            
            // Pre-compute weights for left window
            for(i = 0; i < winSize; ++i) {
                ii = std::min(std::max(y-padding+i,0),height-1); // Ensure to be within image, replicate border if not
                for(j = 0; j < winSize; ++j) {
                    jj = std::min(std::max(x-padding+j,0),width-1);
                    
                    w1[i][j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( fabs(dataLab1[3*(ii*width + jj)] - dataLab1[3*(y*width+x)]) +
                                          fabs(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+x)+1]) + 
                                          fabs(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+x)+2]) )/gammaC);
                }
            }
            
            dBest = 0;
            costBest = INFINITY; // Initialize cost to an high value
            for(d = x-minDisparity; d >= std::max(0,x-maxDisparity); --d) {  // For each allowed disparity ON RIGHT (reverse order)
                
                cost = 0;   // Cost of current match
                tot = 0;    // Sum of weights
                for(i = 0; i < winSize; ++i) {
                    ii = std::min(std::max(y-padding+i,0),height-1);
                    for(j = 0; j < winSize; ++j) {
                        jj = std::min(std::max(d-padding+j,0),width-1);
                        kk = std::min(std::max(x-padding+j,0),width-1);
                        
                        // Build weight
                        w2[i][j] = proximityWeights[i*winSize + j] * 
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
        
    // Consistency check *****************
    for(x=0; x < width; ++x) {      // For each column on RIGHT image
            
        // Pre-compute weights for RIGHT window
        for(i = 0; i < winSize; ++i) {
            ii = std::min(std::max(y-padding+i,0),height-1); // Ensure to be within image, replicate border if not
            for(j = 0; j < winSize; ++j) {
                jj = std::min(std::max(x-padding+j,0),width-1);
                
                w2[i][j] = proximityWeights[i*winSize + j] * 
                           exp(-sqrt( fabs(dataLab2[3*(ii*width + jj)] - dataLab2[3*(y*width+x)]) +
                                      fabs(dataLab2[3*(ii*width + jj)+1] - dataLab2[3*(y*width+x)+1]) + 
                                      fabs(dataLab2[3*(ii*width + jj)+2] - dataLab2[3*(y*width+x)+2]) )/gammaC);
            }
        }
        
        dBest = 0;
        costBest = INFINITY; // Initialize cost to an high value
        for(d = x+minDisparity; d <= std::min(width-1,x+maxDisparity); ++d) {  // For each allowed disparity ON LEFT
            
            cost = 0;   // Cost of current match
            tot = 0;    // Sum of weights
            for(i = 0; i < winSize; ++i) {
                ii = std::min(std::max(y-padding+i,0),height-1);
                for(j = 0; j < winSize; ++j) {
                    jj = std::min(std::max(d-padding+j,0),width-1);
                    kk = std::min(std::max(x-padding+j,0),width-1);
                    
                    // Build weight
                    w1[i][j] = proximityWeights[i*winSize + j] * 
                               exp(-sqrt( fabs(dataLab1[3*(ii*width + jj)] - dataLab1[3*(y*width+d)]) +
                                          fabs(dataLab1[3*(ii*width + jj)+1] - dataLab1[3*(y*width+d)+1]) + 
                                          fabs(dataLab1[3*(ii*width + jj)+2] - dataLab1[3*(y*width+d)+2]) )/gammaC);
                    
                    // Update cost
                    cost += w1[i][j]*w2[i][j]*( abs(data2[3*(ii*width + kk)] - data1[3*(ii*width + jj)]) + 
                                                abs(data2[3*(ii*width + kk)+1] - data1[3*(ii*width + jj)+1]) + 
                                                abs(data2[3*(ii*width + kk)+2] - data1[3*(ii*width + jj)+2]) );
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
    
    
    }  // End of while
}


PyObject *computeASW(PyObject *self, PyObject *args)
{
    PyArrayObject *img1, *img2, *img1Lab, *img2Lab;
    int winSize, maxDisparity, minDisparity, gammaC, gammaP;
    int consistent = 0; // Optional value
    
    // Parse input. See https://docs.python.org/3/c-api/arg.html
    if (!PyArg_ParseTuple(args, "OOOOiiiii|p", &img1, &img2, &img1Lab, &img2Lab, 
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
    npy_float *dataLab1 = (npy_float *)PyArray_DATA(img1Lab); // No elegant way to see them as data[height][width][color]?
    npy_float *dataLab2 = (npy_float *)PyArray_DATA(img2Lab);
    
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
    float proximityWeights[winSize][winSize];
    
    for(i = 0; i < winSize; ++i) {
        for(j = 0; j < winSize; ++j) {
            proximityWeights[i][j] = exp(-sqrt( pow(i-padding,2) + pow(j-padding,2))/gammaP);
        }
    }
    
    
    // Put each image row in queue
    for(i=0; i < height; ++i) {   
        jobs.push(i);
    }
    
    
    if(!consistent) {
    // Start workers
    for(i = 0; i < num_threads; ++i) {
        workersArr[i] = std::thread( workerASW, std::ref(jobs), data1, data2, dataLab1, dataLab2,
                                          disparityMap, &proximityWeights[0][0], gammaC,             // Don't know why usign "proximityWeights" only does not work
                                          width, height, winSize, padding, minDisparity, maxDisparity);
        }
    } else { // If consistent mode is chosen
        
        // Start consistent workers
        for(i = 0; i < num_threads; ++i) {
            workersArr[i] = std::thread( workerASWconsistent, std::ref(jobs), data1, data2, dataLab1, dataLab2,
                                          disparityMap, &proximityWeights[0][0], gammaC,             // Don't know why usign "proximityWeights" only does not work
                                          width, height, winSize, padding, minDisparity, maxDisparity);
        }
    }
    
    // Join threads
    for(i = 0; i < num_threads; ++i) {
        workersArr[i].join();
    }    
    
        
    // Cast to PyObject and return (apparently you cannot return a PyArrayObject)
    return (PyObject*)disparityMapObj;  
    
}






// ******************************** GSW ********************************

// WORK IN PROGRESS
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
    float hist1[bins], hist2[bins],histJoint[bins*bins];
    float histStep = (float)256/bins;
    int a,b;
    
    while(!jobs.empty()) 
    {
        
    jobs.pop(y); // Get element, put it in y and remove from queue
    
    std::cout << "Working " << y << std::endl;
    
    // USING LEFT IMAGE AS REFERENCE
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
                /*
                 * Il peso w[i*winSize+j] misura l'importanza di QUEL pixel nel matching.
                 * La probabilità (calcolata come istogramma) ci dice quanti pixel (di tutta la finestra)
                 * cadono in quell'intervallo (bin). La mutual information ci dice quanto due finestre
                 * sono interdipendenti.
                 * */
                 /*
                //Reset histograms
                for(k=0;k<bins;k++){
                    hist1[k]=1e-10; // Avoid perfect zero...
                    hist2[k]=1e-10; 
                    }
                for(k=0;k<bins*bins;k++){
                    histJoint[k]=1e-10;
                    }
                
                for(i = 0; i < winSize; ++i) {
                    ii = y - padding + i;
                    if( ii < 0) continue;       // Image top border
                    if( ii >= height) break;    // Image bottom border
                    
                    for(j = 0; j < winSize; ++j) {
                        kk = x - padding + j;
                        jj = d - padding + j;
                        
                        if(jj < 0 or kk < 0) continue;         // Image left border
                        if(jj >= width or kk >= width) break;  // Image right border
                        
                        for(k=0;k<3;++k){ // For each color channel
                            a = data1[3*(ii*width + kk)+k]/histStep;
                            b = data2[3*(ii*width + jj)+k]/histStep;
                            
                            //hist1[ a ] += w[i*winSize + j];
                            //hist2[ b ] += w[i*winSize + j];
                            //histJoint[ a*bins + b ] += w[i*winSize + j];
                            
                            hist1[ a ]++;
                            hist2[ b ]++;
                            histJoint[ a*bins + b ]++;
                            }
                    }
                }
                
                
                // Convert to probabilities
                for(k=0;k<bins;++k){
                    hist1[k]=hist1[k]/tot;
                    hist2[k]=hist2[k]/tot;
                    }
                for(k=0;k<bins*bins;++k){
                    histJoint[k]=histJoint[k]/tot;
                    }
                
                
                // Compute cost. How to compute cost???
                
                */
                
                // NORMAL COLOR DIFFERENCES
                
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
                        // SQUARED DIFFERENCE TO BE REPLACED WITH MUTUAL INFORMATION. BUT HOW?
                        // MUTUAL INFORMATION CANNOT BE ITERATED OVER PIXEL
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
    
    
    } // End of while
    
}


PyObject *computeGSW(PyObject *self, PyObject *args)
{
    PyArrayObject *img1, *img2;
    int winSize, maxDisparity, minDisparity, gamma, iterations, bins;
    float fMax;
    // Parse input. See https://docs.python.org/3/c-api/arg.html
    if (!PyArg_ParseTuple(args, "OOiiiifii", &img1, &img2, 
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
    PyArrayObject *disparityMapObj = (PyArrayObject*)PyArray_EMPTY(2, disparityMapDims, NPY_INT16,0);
    npy_int16 *disparityMap = (npy_int16 *)PyArray_DATA(disparityMapObj); // Pointer to first element
    
    // Working variables
    int padding = winSize / 2;
    int i,j;
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
    
    // Cast to PyObject and return (apparently you cannot return a PyArrayObject)
    return (PyObject*)disparityMapObj;
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

