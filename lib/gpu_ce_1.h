
#ifndef CANNY_EDGE_H

#define CANNY_EDGE_H

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "cuda.h"

#define PI 3.14

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
        swap(&(kernel[i]) ,  &(kernel[width-1-i]));
    }
}
//need to use float ** kernel since we assign a new memory block to kernel within the function (malloc)

void guassianderivativekernel(float ** kernel, int *width, float sigma){
    int a =  round(2.5*sigma - 0.5);
    *width = 2*a+1;
    float sum = 0 ;
       (*kernel) = (float* ) malloc(sizeof(float)*(*width));
    for(int i=0; i<(*width); i++){
        (*kernel)[i]= -1*(i-a)*exp((-1*(i-a)*(i-a))/(2*sigma*sigma));
       
        sum = sum - i* (*kernel)[i];
   }
     for(int i=0; i<(*width); i++){
        (*kernel)[i] = (*kernel)[i]/sum;
     }
}

//need to use float ** kernel since we assign a new memory block to kernel within the function (malloc)
void guassiankernel(float ** kernel, int *width, float sigma){
    int a =  round(2.5*sigma - 0.5);
    *width = 2*a+1;
    float sum = 0 ;
    (*kernel) = (float* ) malloc(sizeof(float)*(*width));
    
    for(int i=0; i<(*width); i++){
        (*kernel)[i] = exp((-1*(i-a)*(i-a))/(2*sigma*sigma));
        sum = sum + (*kernel)[i];
     
    }
   
    for(int i=0; i<(*width); i++){
        (*kernel)[i] = (*kernel)[i]/sum;
       
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



__global__ void gpu_convolution(float * image, int  width, int  height, float * kernel, int  kwidth, int  kheight, float * output){
        int x= threadIdx.x+blockIdx.x*blockDim.x;
            if (x<height*width){
                    float sum = 0;
                    float temp = 0.0;

                    for( int y = 0; y < kwidth*kheight; y++){

                        // offset of y in kernel matrix
                        int offseti = -1*(kheight/2) + (y/kwidth);
                        int offsetj = -1*(kwidth/2) + (y%kwidth);

                        // adding offset to x in image matrix
                        int y_pos = (offseti+(int)(x/width) )*width + (offsetj+(int)(x%width));

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

__global__ void gpu_magnitude(float * vertical, float * horizontal, float * magnitude, int width, int height){
        int i= threadIdx.x+blockIdx.x*blockDim.x;
        if (i<height*width){
                  (magnitude)[i] = sqrt(vertical[i]*vertical[i] + horizontal[i]*horizontal[i]);
        }
}

__global__  void gpu_direction(float * vertical, float * horizontal, float * direction, int width, int height){
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    if (i<height*width){
             (direction)[i] = atan2(horizontal[i], vertical[i]);

    }
}

__global__ void gpu_suppression(float * direction, float * magnitude, float * suppression, int width, int height){
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    if (i<height*width){
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



__global__ void gpu_hysteresis(float * suppression, float * sorted_suppression, float * hysteresis, int width, int height){
    
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    if (i<height*width){
        float t_high = sorted_suppression[(int)((width*height)*0.9)];
        float t_low = t_high/5; 
        if( suppression[i] > t_high) hysteresis[i] = 255;
        else if(suppression[i ] < t_low ) hysteresis[i] = 0;
        else hysteresis[i] = 125;
    }

}


__global__ void gpu_edge_linking(float * hysteresis, float * edge_linking, int width, int height){
  
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    if (i<height*width){
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
__global__ void gpu_copy(float * original, float * copy, int length){
    
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    if (i<length){
        copy[i] = original[i];
    }
}

#endif