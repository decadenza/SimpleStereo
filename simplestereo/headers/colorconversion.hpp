#include <Python.h>
#include <math.h>

// Convert from BGR to CIELab
// Adapted from original code by Mohamed Shahawy (MIT Licence).
class ColorConversion{
    private:
    void RGBtoXYZ(npy_ubyte R, npy_ubyte G, npy_ubyte B, double &X, double &Y, double &Z);
    void XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b);
    void RGBtoLab(npy_ubyte R, npy_ubyte G, npy_ubyte B, double &L, double &a, double &b);

    public:
    void ImageFromBGR2Lab(npy_ubyte *pBGR, double *pLab, int width, int height);
    void ImageFromRGB2Lab(npy_ubyte *pRGB, double *pLab, int width, int height);
    };


void ColorConversion::RGBtoXYZ(npy_ubyte R, npy_ubyte G, npy_ubyte B, double &X, double &Y, double &Z){
    float r, g, b;
    
    r = R / 255.0;
    g = G / 255.0;
    b = B / 255.0;
    
    if (r > 0.04045)
        r = powf(( (r + 0.055) / 1.055 ), 2.4);
    else r /= 12.92;
    
    if (g > 0.04045)
        g = powf(( (g + 0.055) / 1.055 ), 2.4);
    else g /= 12.92;
    
    if (b > 0.04045)
        b = powf(( (b + 0.055) / 1.055 ), 2.4);
    else b /= 12.92;
    
    r *= 100; g *= 100; b *= 100;
    
    // Calibration for observer @2° with illumination = D65
    X = r * 0.4124 + g * 0.3576 + b * 0.1805;
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505;
}

void ColorConversion::XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b)
{
    double x, y, z;
    const float refX = 95.047, refY = 100.0, refZ = 108.883;
    
    // References set at calibration for observer @2° with illumination = D65
    x = X / refX;
    y = Y / refY;
    z = Z / refZ;
    
    if (x > 0.008856)
        x = powf(x, 1 / 3.0);
    else x = (7.787 * x) + (16.0 / 116.0);
    
    if (y > 0.008856)
        y = powf(y, 1 / 3.0);
    else y = (7.787 * y) + (16.0 / 116.0);
    
    if (z > 0.008856)
        z = powf(z, 1 / 3.0);
    else z = (7.787 * z) + (16.0 / 116.0);
    
    L = 116 * y - 16;
    a = 500 * (x - y);
    b = 200 * (y - z);
}


void ColorConversion::RGBtoLab(npy_ubyte R, npy_ubyte G, npy_ubyte B, double &L, double &a, double &b)
{
    double X, Y, Z;
    RGBtoXYZ(R, G, B, X, Y, Z);
    XYZtoLab(X, Y, Z, L, a, b);
}

// From BGR (OpenCV standard)
void ColorConversion::ImageFromBGR2Lab(npy_ubyte *pBGR, double *pLab, int width, int height){
    int i1, i2, i3, index;
    index=width*height*3;
    for(i1=0, i2=1, i3=2; i3<index; i1+=3, i2+=3, i3+=3)
        RGBtoLab(pBGR[i3], pBGR[i2], pBGR[i1], pLab[i1], pLab[i2], pLab[i3]); //BGR inverted
    }

// From RGB
void ColorConversion::ImageFromRGB2Lab(npy_ubyte *pRGB, double *pLab, int width, int height){
    int i1, i2, i3, index;
    index=width*height*3;
    for(i1=0, i2=1, i3=2; i3<index; i1+=3, i2+=3, i3+=3)
        RGBtoLab(pRGB[i1], pRGB[i2], pRGB[i3], pLab[i1], pLab[i2], pLab[i3]);
    }
