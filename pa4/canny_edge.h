


#define CANNY_EDGE_H

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#define PI 3.14



void _suppression(float * direction, float * magnitude, float * suppression, int width, int height){

   
 
    for(int i =0 ; i< width*height; i++){
        int y = i%width; // column
        int x = i/width; // row 
        float angle = direction[i];
        if (angle < 0 ) angle += PI;
        angle = angle*180/ PI ;
        if(angle <= 22.5 && angle > 157.5){
            (suppression)[i] = magnitude[i];
            if ( x>0 ) if ( magnitude[i] < magnitude[ (x-1)*width + y]) (suppression)[i] = 0;
            if (x < (height -1 )) if (magnitude[i] < magnitude[(x+1)*width + y ]) (suppression)[i]= 0;

        }
        if(angle > 22.5 && angle <= 67.5 ){
            (suppression)[i] = magnitude[i];
            if ( x>0 && y>0 ) if ( magnitude[i] < magnitude[ (x-1)*width + y-1]) (suppression)[i] = 0;
            if (x < (height -1) && y<(width-1) ) if (magnitude[i] < magnitude[(x+1)*width + y+1 ]) (suppression)[i]= 0;
        }
        if(angle > 67.5 && angle <= 112.5){
             (suppression)[i] = magnitude[i];
            if ( y>0 ) if ( magnitude[i] < magnitude[ (x)*width + y-1]) (suppression)[i] = 0;
            if (y<width-1 ) if (magnitude[i] < magnitude[(x)*width + y+1 ]) (suppression)[i]= 0;
        }
        if(angle > 112.5 && angle <= 157.5){
             (suppression)[i] = magnitude[i];
            if ( x>0 && y<width-1 ) if ( magnitude[i] < magnitude[ (x-1)*width + y+1]) (suppression)[i] = 0;
            if (x < height-1 && y > 0 ) if (magnitude[i] < magnitude[(x+1)*width + y-1 ]) (suppression)[i]= 0;

        }
    }

}

int comp (const void * elem1, const void * elem2) 
{
    float f = *((float*)elem1);
    float s = *((float*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

void _edge_linking(float * suppression, float * edge_linking, int width, int height){
  
    qsort (suppression, sizeof(suppression)/sizeof(*suppression), sizeof(*suppression), comp);
    float t_high = suppression[(int)((width*height)*0.9) - 1];
    float t_low = t_high/5;
    float  * hysteresis = (float * ) malloc(sizeof(float)*width*height);

 
    for(int i  =0 ;i <width*height;i++){
        if(suppression[i] > t_high) hysteresis[i] = 255;
        else if(suppression[i ]< t_low ) hysteresis[i] = 0;
        else hysteresis[i] = 125;
    }

 
    for(int i  =0 ;i <width*height;i++){
        if (hysteresis[i]==125){
        (edge_linking)[i] = 0;

        int x = i/width;
        int y = i%width;

        if(x>0) if(hysteresis[(x-1)*width+y] ==255) (edge_linking)[i] = 255;
        if(x>0 && y>0) if(hysteresis[(x-1)*width+y-1] ==255) (edge_linking)[i] = 255;
        if(x>0 && y<width-1) if(hysteresis[(x-1)*width+y+1] ==255) (edge_linking)[i] = 255;

        if(y>0) if(hysteresis[(x)*width+y-1] ==255) (edge_linking)[i] = 255;
        if(y<width-1) if(hysteresis[(x)*width+y+1] ==255) (edge_linking)[i] = 255;

        if(x<height-1) if(hysteresis[(x+1)*width+y] ==255) (edge_linking)[i] = 255;
        if(x<height-1 && y>0) if(hysteresis[(x+1)*width+y-1] ==255) (edge_linking)[i] = 255;
        if(x<height-1 && y<width-1) if(hysteresis[(x+1)*width+y+1] ==255) (edge_linking)[i] = 255;
        
        }

    }



}


void _magnitude(float * vertical, float * horizontal, float * magnitude, int width, int height){
 
    for(int i = 0; i< width*height; i++){
        (magnitude)[i] = sqrt(vertical[i]*vertical[i] + horizontal[i]*horizontal[i]);

    }
}


void _direction(float * vertical, float * horizontal, float * direction, int width, int height){
 
    for(int i = 0; i< width*height; i++){
        (direction)[i] = atan2(horizontal[i], vertical[i]);

    }
}



void swap( float * a, float * b){
    float temp = *a;
    *a = *b;
    *b = temp;

}

void kernelflipping(float * kernel, int width){

for(int i=0; i<(floor(width/2)); i++){
     swap(&(kernel)[i] ,  &(kernel)[width-1-i]);
}
}

void guassianderivativekernel(float * kernel, int *width, float sigma){
    int a =  round(2.5*sigma - 0.5);
    *width = 2*a+1;
    float sum = 0 ;
       (kernel) = (float* ) malloc(sizeof(float)*(*width));
    for(int i=0; i<(*width); i++){
        (kernel)[i]= -1*(i-a)*exp((-1*(i-a)*(i-a))/(2*sigma*sigma));
       
        sum = sum - i* (kernel)[i];
   }
     for(int i=0; i<(*width); i++){
        (kernel)[i] = (kernel)[i]/sum;
     
    }
}

void guassiankernel(float * kernel, int * width, float sigma){
    int a =  round(2.5*sigma - 0.5);
    *width = 2*a+1;
    float sum = 0 ;
    (kernel) = (float* ) malloc(sizeof(float)*(*width));
    
    for(int i=0; i<(*width); i++){
        (kernel)[i]= exp((-1*(i-a)*(i-a))/(2*sigma*sigma));
      
        sum = sum + (kernel)[i];
     
    }
   
    for(int i=0; i<(*width); i++){
        (kernel)[i] = (kernel)[i]/sum;
       
    }
}


void convolution_3(float ** pimage, int * pwidth, int * pheight, float ** pkernel, int * pkwidth, int * pkheight, float ** poutput){
// getting parameters value
float * image = *pimage;
int width = *pwidth;
int height = *pheight;
float * kernel = *pkernel;
int kwidth = *pkwidth;
int kheight = *pkheight;

// temp variable

 
for ( int x = 0; x < width*height ; x++){
    float sum = 0;
    float temp = 0.0;

    for( int y = 0; y < kwidth*kheight; y++){

        // offset of y in kernel matrix
        int offseti = -1*floor(kheight/2) + floor(y/kwidth);
        int offsetj = -1*floor(kwidth/2) + floor(y%kwidth);

        // adding offset to x in image matrix
        int y_pos = (offseti+(int)floor(x/width) )*width + (offsetj+(int)floor(x%width));

        // mulitply respective kernel cell and image cell and add to sum
        if ( y_pos < width*height 
            && y_pos  >= 0 )
                temp = (float)image[  y_pos  ]*kernel[y];

                sum = sum + temp;        
    }
   
    //
    (*poutput)[x] =  (float)sum;
}
}

void convolution_2(float *image,int height,int width,float *mask, int ker_h,int ker_w,float *output) {

        int i,j,k,m;
        int offseti,offsetj;
        float sum=0;

        for(i=0;i<height;i++){
                for(j=0;j<width;j++) {
                   sum=0;
                   for(k=0;k<ker_h;k++)
                     for(m=0;m<ker_w;m++) {
                        offseti = -1*(ker_h/2) + k;
                        offsetj=-1*(ker_w/2)+m;

                        if(i+offseti>=0 && i+offseti<height &&j+offsetj>=0 && j+offsetj<width) {
                                sum+=image[(i+offseti)*width+j+offsetj]*mask[k*ker_w+m];
                        }
                      }
                output[i*width+j] = sum;
                }
        }
}

void convolution(float * image, int  width, int  height, float * kernel, int  kwidth, int kheight, float * output){


// temp variable

 
for ( int x = 0; x < width*height ; x++){
    float sum = 0;
    float temp = 0.0;

    for( int y = 0; y < kwidth*kheight; y++){

        // offset of y in kernel matrix
        int offseti = -1*floor(kheight/2) + floor(y/kwidth);
        int offsetj = -1*floor(kwidth/2) + floor(y%kwidth);

        // adding offset to x in image matrix
        int y_pos = (offseti+(int)floor(x/width) )*width + (offsetj+(int)floor(x%width));

        // mulitply respective kernel cell and image cell and add to sum
        if ( y_pos < width*height 
            && y_pos  >= 0 )
                temp = (float)image[  y_pos  ]*kernel[y];

                sum = sum + temp;        
    }
   
    //
    output[x] =  (float)sum;
    }
}
