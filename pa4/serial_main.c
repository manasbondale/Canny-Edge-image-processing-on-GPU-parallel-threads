#include <time.h>
#include <sys/time.h>

#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "image_template.h"
#include "canny_edge.h"

int main(int argc, char **argv)
{

    
  if (argc<3)
    {
      printf("USAGE: %s <image-file> <sigma value> \n", argv[0]);
      return 1;
    }


  struct timeval start_end2end, end_end2end, start_comp, end_comp;
  
  gettimeofday(&start_end2end, NULL);

   float sigma;
   char * imgname;
   int height, width;

    imgname = argv[1];
    sigma   = atof(argv[2]);


   
   
    

    
    float * imggray;
    read_image_template(imgname, &imggray, &width, &height);

    gettimeofday(&start_comp, NULL);

    
    float * kernel,  * kerneld;
    int kwidth, kwidthd;
    guassiankernel(kernel, kwidth, sigma);
     
    guassianderivativekernel(kerneld, kwidthd, sigma);

    for(int i=0; i< kwidth; i++){
      printf("%f ", kernel[i]);
    }
    printf("\n");

    for(int i=0; i< kwidthd; i++){
      printf("%f ", kerneld[i]);
    }
    printf("\n");
    
    kernelflipping(kerneld, kwidthd);


    for(int i=0; i< kwidth; i++){
      printf("%f ", kernel[i]);
    }
    printf("\n");

    for(int i=0; i< kwidthd; i++){
      printf("%f ", kerneld[i]);
    }
    printf("\n");
    // horizontal
    float  * temphorizontal= (float*) malloc(sizeof(float)*width*height);
    
    int kw = 1;
    int kh = kwidth;
    //convolution(&imggray, &width,&height, &kernel, &kw, &kh , &temphorizontal);
    convolution(imggray, width, height, kernel, kw, kh , temphorizontal);

       
    float  * horizontal= (float*) malloc(sizeof(float)*width*height);
    kw=kwidth;
    kh=1;
   // convolution(&temphorizontal, &width, &height, &kerneld, &kw, &kh , &horizontal);
    convolution( temphorizontal,  width, height, kerneld, kw, kh , horizontal);

    
   // vertical
    float  * tempvertical= (float*) malloc(sizeof(float)*width*height);
    
     kw = kwidth;
     kh = 1;
 //   convolution(&imggray, &width,&height, &kernel, &kw, &kh , &tempvertical);
        convolution(imggray, width, height, kernel, kw, kh, tempvertical);
    free(imggray);
     
    float  * vertical= (float*) malloc(sizeof(float)*width*height);
    kw=1;
    kh=kwidth;
    //convolution(&tempvertical, &width, &height, &kerneld, &kw, &kh , &vertical);
    convolution( tempvertical,  width, height, kerneld, kw, kh , vertical);
  
    
    
    //magnitude
    float  * magnitude= (float*) malloc(sizeof(float)*width*height);
    _magnitude(vertical, horizontal, magnitude, width, height);
    
    
    
    //direction
    float  * direction= (float*) malloc(sizeof(float)*width*height);
    _direction(vertical, horizontal, direction, width, height);

  
   
    
    //suppression
    float  * suppression= (float*) malloc(sizeof(float)*width*height);
    _suppression(direction, magnitude, suppression, width, height);

   

    //edge_linking
     float  * edge_linking= (float*) malloc(sizeof(float)*width*height);
    _edge_linking(suppression,edge_linking, width, height);

    gettimeofday(&end_comp, NULL);

   char htname[256];
    strcat(strtok(strcpy(htname, imgname), "."), "_h_t.pgm");
    strtok(htname,"/");
    strtok(NULL,"/"); 
    write_image_template(strtok(NULL,"/"), temphorizontal, width, height);
    free(temphorizontal);

    char vtname[256];
    strcat(strtok(strcpy(vtname, imgname), "."), "_v_t.pgm");
    strtok(vtname,"/");
    strtok(NULL,"/"); 
    write_image_template(strtok(NULL,"/"), tempvertical, width, height);
    free(tempvertical);

    char hname[256];
    strcat(strtok(strcpy(hname, imgname), "."), "_h.pgm");
    strtok(hname,"/");
    strtok(NULL,"/"); 
    write_image_template("horizontal.pgm", horizontal, width, height);
    free(horizontal);

char vname[256];
    strcat(strtok(strcpy(vname, imgname), "."), "_v.pgm");
    strtok(vname,"/");
    strtok(NULL,"/"); 
    write_image_template("vertical.pgm", vertical, width, height);
    free(vertical);
    
    
     char mname[256];
    strcat(strtok(strcpy(mname, imgname), "."), "_m.pgm");
    strtok(mname,"/");
    strtok(NULL,"/"); 
    write_image_template("magnitude.pgm", magnitude, width, height);
    free(magnitude);

char dname[256];
    strcat(strtok(strcpy(dname, imgname), "."), "_d.pgm");
    strtok(dname,"/");
    strtok(NULL,"/"); 
    write_image_template("direction.pgm", direction, width, height);
    free(direction);
    
    
    char sname[256];
    strcat(strtok(strcpy(sname, imgname), "."), "_s.pgm");
    strtok(sname,"/");
    strtok(NULL,"/"); 
    write_image_template(strtok(NULL,"/"), suppression, width, height);
    free(suppression);

    char ename[256];
    strcat(strtok(strcpy(ename, imgname), "."), "_e.pgm");
    strtok(ename,"/");
    strtok(NULL,"/"); 
    //write_image_template(strtok(NULL,"/"), edge_linking, width, height);
    write_image_template("edge.pgm", edge_linking, width, height);
    free(edge_linking);
    
    gettimeofday(&end_end2end, NULL); 


    int size = width*height;

    printf("%d, ", size);
    printf("%f, ", sigma);

    printf("%ld, ", ((end_comp.tv_sec * 1000000 + end_comp.tv_usec)
		  - (start_comp.tv_sec * 1000000 + start_comp.tv_usec)));
    printf("%ld\n", ((end_end2end.tv_sec * 1000000 + end_end2end.tv_usec)
		  - (start_end2end.tv_sec * 1000000 + start_end2end.tv_usec)));

  return 0;
}