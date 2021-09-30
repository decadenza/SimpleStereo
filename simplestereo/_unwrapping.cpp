/* C++ extension for phase unwrapping algorithms */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
 
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

#include <thread>


struct Coord
{
int x;
int y;  
};

// Wrap in [-pi,pi)
double W(double angle)
{
    double a = fmod(angle + M_PI, 2 * M_PI);
    return a >= 0 ? (a - M_PI) : (a + M_PI);
}

// Get neighbours as 3x3 tiles (which include central one)
// managing border limits
std::vector<Coord> neighbours(int y, int x, npy_intp *dims)
{
    std::vector<Coord> neigh;
    int top = std::max(0,y-1);
    int right = std::min(x+2,(int)dims[1]);
    int bottom = std::min(y+2,(int)dims[0]);
    int left = std::max(x-1,0);

    for(int i = top; i < bottom; ++i) {
            for(int j = left; j < right; ++j) {
                Coord n;
                n.y = i;
                n.x = j;
                neigh.push_back(n);
            }
        }

    return neigh;
}


//Main IIR function
PyObject *infiniteImpulseResponse(PyObject *self, PyObject *args)
{
    PyArrayObject *phase;
    double tau;
    
    // Parse input. See https://docs.python.org/3/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &phase, &tau)){
        PyErr_SetString(PyExc_ValueError, "Invalid input format!");
        return NULL;
        }
    // Check input format
    if (PyArray_NDIM(phase)!=2){
        PyErr_SetString(PyExc_ValueError, "Wrong phase dimensions!");
        return NULL;
        }    
    if (tau<0 || tau>1){
        PyErr_SetString(PyExc_ValueError, "Wrong tau value!");
        return NULL;
        }
    
    //Retrieve dims
    int h = PyArray_DIM(phase,0);
    int w = PyArray_DIM(phase,1);
    
    //Initialisation
    npy_intp dims[2] = {h, w};
    PyArrayObject *unwrapped = (PyArrayObject *) PyArray_Zeros(2, dims, PyArray_DTYPE(phase), 0);
    
    bool s[h][w];
    int y;
    int x;
    int S;
    double temp;
    double cur;
    std::vector<Coord> neigh;
    
    // Initialize s with zeros
    std::fill_n(&s[0][0], h * w, 0);
    
    // Remove transient response (first row unwrapped forth and back)
    // First row forward...
    for(x=0;x<w;++x)
        {
        S = 0;
        temp = 0;
        cur = *((double *)PyArray_GETPTR2(phase,y,x));
        neigh = neighbours(y,x,dims);
        for(auto const& c : neigh)
            {
            if (s[c.y][c.x]) // Get only neighbours already processed
                {
                S+=1;
                temp += (double) ( ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0] + tau*W(cur - ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0]) );
                }
            }
        s[y][x] = 1;
        PyArray_SETITEM(unwrapped, PyArray_GETPTR2(unwrapped,y,x), Py_BuildValue("f", (S>0)? temp/S : cur) );
        }
    
    // ...and backward
    for(x=w-1;x>0;--x)
        {
        S = 0;
        temp = 0;
        cur = *((double *)PyArray_GETPTR2(phase,y,x));
        neigh = neighbours(y,x,dims);
        for(auto const& c : neigh)
            {
            if (s[c.y][c.x])
                {
                S+=1;
                temp += (double) ( ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0] + tau*W(cur - ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0]) );
                }
            }
        s[y][x] = 1;
        PyArray_SETITEM(unwrapped, PyArray_GETPTR2(unwrapped,y,x), Py_BuildValue("f", (S>0)? temp/S : cur) );
        }    
    
    // Main loop over the whole image 
    for(y=0;y<h;++y)
    {
        for(x=0;x<w;++x)
            {
            S = 0;
            temp = 0;
            cur = *((double *)PyArray_GETPTR2(phase,y,x));
            neigh = neighbours(y,x,dims);
            for(auto const& c : neigh)
                {
                if (s[c.y][c.x])
                    {
                    S+=1;
                    temp += (double) ( ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0] + tau*W(cur - ((double *)PyArray_GETPTR2(unwrapped,c.y,c.x))[0]) );
                    }
                }
            PyArray_SETITEM(unwrapped, PyArray_GETPTR2(unwrapped,y,x), Py_BuildValue("f", (S>0)? temp/S : cur) );
            s[y][x] = 1;
            }
    }  
    
    //It should be used whenever 0-dimensional arrays could be returned to Python.
    return PyArray_Return(unwrapped);
}



/*____________________PYTHON MODULE INITIALIZATION____________________*/
static struct PyMethodDef module_methods[] = {
    /* {name (external), function, calling, doc} */
    {"infiniteImpulseResponse",  infiniteImpulseResponse, METH_VARARGS, NULL},
    {NULL,NULL,0, NULL}
};



static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_unwrapping",
        "Unwrapping interface",
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__unwrapping(void)
{
    PyObject *m;
    import_array(); //This function must be called to use Numpy C-API
    
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    
    return m;
}

