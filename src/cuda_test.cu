#include <cuda.h>
#include<stdio.h>

__global__ void dd(int *d_a, int *d_b, int *d_c, int vec_size){
    int tid= threadIdx.x+blockIdx.x*blockDim.x;
    if (tid<vec_size) d_c[tid]= d_a[tid] + d_b[tid];
}

int main(int argc, char ** argv){
    cudaSetDevice(3);
    int i, vec_size;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    vec_size=1000000;
    h_a=(int *)malloc(sizeof(int)*vec_size);

    h_b=(int *)malloc(sizeof(int)*vec_size);
    
    h_c=(int *)malloc(sizeof(int)*vec_size);

    cudaMalloc((void **) &d_a, sizeof(int)*vec_size);
    cudaMalloc((void **) &d_b, sizeof(int)*vec_size);
    cudaMalloc((void **) &d_c, sizeof(int)*vec_size);

    for(i=0;i<vec_size;i++){
        h_a[i]=i;
        h_b[i]=10;

    }

    cudaMemcpy(d_a, h_a, sizeof(int)*vec_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*vec_size,cudaMemcpyHostToDevice);
   
    dd<<<1000,1000>>>(d_a,d_b,d_c,vec_size); //dim3 ?
    cudaMemcpy(h_c, d_c, sizeof(int)*vec_size,cudaMemcpyDeviceToHost);
    for(i =0; i<vec_size;i++) printf("\n C %d == %d", i, h_c[i]);

}



