#include <time.h>
#include <sys/time.h>

#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <thrust/sort.h>

#include "image_template.h"
#include "gpu_ce_1.h"
#include "canny_edge.h"
#include "cuda.h"

#define GPU_NO 3

long clock(struct timeval t, struct timeval s){
  return ((s.tv_sec * 1000000 + s.tv_usec) - (t.tv_sec * 1000000 + t.tv_usec));
}

int main(int argc, char **argv)
{

  cudaSetDevice(GPU_NO);
    
  if (argc<3)
    {
      printf("USAGE: %s <image-file> <sigma value> \n", argv[0]);
      return 1;
    }


  struct timeval start_end2end, end_end2end, start_comp, end_comp; //pa2 times
  struct timeval start_pa4, end_pa4, start_pa5, end_pa5; //pa4, pa5 times
  struct timeval start_conv, end_conv, start_magdir, end_magdir, start_supp, end_supp,
                  start_sort, end_sort, start_hyst, end_hyst, start_edge, end_edge,
                  start_d2h, end_d2h, start_h2d, end_h2d;
  
  gettimeofday(&start_end2end, NULL);

  float sigma;
  char * imgname;
  int height, width;

  imgname = argv[1];
  sigma   = atof(argv[2]);

  float * imggray;
  read_image_template(imgname, &imggray, &width, &height);

  gettimeofday(&start_comp, NULL);

  gettimeofday(&start_pa4, NULL);

    int threads = 1000;
    int blocks = width*height/1000;

   

    float * kernel,  * kerneld;
    int kwidth, kwidthd;
    float  * temphorizontal = (float*) malloc(sizeof(float)*width*height);
    float  * tempvertical = (float*) malloc(sizeof(float)*width*height);
    float  * horizontal = (float*) malloc(sizeof(float)*width*height);
    float  * vertical = (float*) malloc(sizeof(float)*width*height);
    float  * magnitude = (float*) malloc(sizeof(float)*width*height);
    float  * direction = (float*) malloc(sizeof(float)*width*height);
    float  * suppression = (float*) malloc(sizeof(float)*width*height);
    float  * sorted_suppression = (float*) malloc(sizeof(float)*width*height);
    float  * hysteresis = (float*) malloc(sizeof(float)*width*height);
    float  * edge_linking = (float*) malloc(sizeof(float)*width*height);


    float * dimage, *dhorizontal, *dtemphorizontal, *dvertical, *dtempvertical, *dkernel, *dkerneld;
    float *ddirection, *dmagnitude, *dedge_linking , *dsuppression, *dhysteresis, *dsorted_suppression ;
    cudaMalloc((void **) &dimage, sizeof(float)*height*width);
    cudaMalloc((void **) &dtemphorizontal, sizeof(float)*height*width);
    cudaMalloc((void **) &dhorizontal, sizeof(float)*height*width);
    cudaMalloc((void **) &dtempvertical, sizeof(float)*height*width);
    cudaMalloc((void **) &dvertical, sizeof(float)*height*width);
    cudaMalloc((void **) &ddirection, sizeof(float)*height*width);
    cudaMalloc((void **) &dmagnitude, sizeof(float)*height*width);
    cudaMalloc((void **) &dsuppression, sizeof(float)*height*width);
    cudaMalloc((void **) &dsorted_suppression, sizeof(float)*height*width);
    cudaMalloc((void **) &dhysteresis, sizeof(float)*height*width);
    cudaMalloc((void **) &dedge_linking, sizeof(float)*height*width);
    
 
    guassiankernel(kernel, kwidth, sigma);
    guassianderivativekernel(kerneld, kwidthd, sigma);
    kernelflipping(kerneld, kwidthd);


    gettimeofday(&start_h2d, NULL);
    cudaMalloc((void **) &dkernel, sizeof(float)*kwidth);
    cudaMemcpy(dkernel, kernel, sizeof(float)*kwidth, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &dkerneld, sizeof(float)*kwidthd);
    cudaMemcpy(dkerneld, kerneld, sizeof(float)*kwidthd, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &dimage, sizeof(float)*height*width);
    cudaMemcpy(dimage, imggray, sizeof(float)*height*width, cudaMemcpyHostToDevice);
    gettimeofday(&end_h2d, NULL);


    gettimeofday(&start_conv, NULL);
    //horizontal
    gpu_convolution<<<blocks,threads>>>(dimage, width, height, dkernel, 1, kwidth , dtemphorizontal);
    gpu_convolution<<<blocks,threads>>>(dtemphorizontal, width, height, dkerneld, kwidth, 1 , dhorizontal);

    //vertical
    gpu_convolution<<<blocks,threads>>>(dimage, width, height, dkernel, kwidth, 1 , dtempvertical);
    gpu_convolution<<<blocks,threads>>>(dtempvertical, width, height, dkerneld, 1, kwidth , dvertical);
    gettimeofday(&end_conv, NULL);


    gettimeofday(&start_magdir, NULL);
    //magnitude
    gpu_magnitude<<<blocks, threads>>>(dvertical, dhorizontal, dmagnitude, width, height);
    //direction
    gpu_direction<<<blocks, threads>>>(dvertical, dhorizontal, ddirection, width, height);
    gettimeofday(&end_magdir, NULL);

  
    gettimeofday(&end_pa4, NULL);
    gettimeofday(&start_pa5, NULL);

    //suppression
    gettimeofday(&start_supp, NULL);
    gpu_suppression<<<blocks,threads>>>(ddirection, dmagnitude, dsuppression, width, height);
    gettimeofday(&end_supp, NULL);

    //copy
    gpu_copy<<<blocks,threads>>>(dsuppression, dsorted_suppression,  width*height);
   

    //sort
    gettimeofday(&start_sort, NULL);
    //gpu_sort<<<blocks, threads>>>(dsorted_suppression, 0, width*height, threads);
    thrust::sort(dsorted_suppression, dsorted_suppression + width*height);
    gettimeofday(&end_sort, NULL);

    //hysteresis
    gettimeofday(&start_hyst, NULL);
    gpu_hysteresis<<<blocks,threads>>>(dsuppression, dsorted_suppression, dhysteresis, width, height);
    gettimeofday(&end_hyst, NULL);

    //edge_linking
    gettimeofday(&start_edge, NULL);
    gpu_edge_linking<<<blocks,threads>>>(dhysteresis,dedge_linking, width, height);
    gettimeofday(&end_edge, NULL);

    
    // d2h
    gettimeofday(&start_d2h, NULL);
    cudaMemcpy(temphorizontal, dtemphorizontal, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(horizontal, dhorizontal, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(tempvertical, dtempvertical, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(vertical, dvertical, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(magnitude, dmagnitude, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(direction, ddirection, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(suppression, dsuppression, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(hysteresis, dhysteresis, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    cudaMemcpy(edge_linking, dedge_linking, sizeof(float)*height*width,cudaMemcpyDeviceToHost);
    gettimeofday(&end_d2h, NULL);

    gettimeofday(&end_pa5, NULL);

    gettimeofday(&end_comp, NULL);


    write_image_template("temphorizontal.pgm", temphorizontal, width, height);
    write_image_template("tempvertical.pgm", tempvertical, width, height);
    write_image_template("horizontal.pgm", horizontal, width, height);
    write_image_template("vertical.pgm", vertical, width, height);
    write_image_template("magnitude.pgm", magnitude, width, height);
    write_image_template("direction.pgm", direction, width, height);
    write_image_template("suppression.pgm", suppression, width, height);
    write_image_template("hysteresis.pgm", suppression, width, height);
    write_image_template("edge.pgm", edge_linking, width, height);
    
    free(temphorizontal);
    free(tempvertical);
    free(horizontal);
    free(vertical);
    free(magnitude);
    free(direction);
    free(suppression);
    free(dhysteresis);
    free(edge_linking);

    gettimeofday(&end_end2end, NULL); 


    int size = width;
    printf("%d, ", size);
    printf("%f, ", sigma);

    printf("%ld, ", clock(start_comp, end_comp));
    printf("%ld, ", clock(start_end2end, end_end2end));
    printf("%ld, ", clock(start_pa4, end_pa4));
    printf("%ld, ", clock(start_pa5, end_pa5));
    printf("%ld, ", clock(start_h2d, end_h2d));
    printf("%ld, ", clock(start_conv, end_conv));
    printf("%ld, ", clock(start_magdir, end_magdir));
    printf("%ld, ", clock(start_supp, end_supp));
    printf("%ld, ", clock(start_sort, end_sort));
    printf("%ld, ", clock(start_hyst, end_hyst));
    printf("%ld, ", clock(start_d2h, end_d2h));

    printf("\n");

    return 0;
}

