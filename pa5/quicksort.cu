#include <cuda.h>
#include<stdio.h>
#include <stdlib.h>

__global__ void quicksort(int *w, int *p, int *q, int w_size, int p_size){

    int i= threadIdx.x+blockIdx.x*blockDim.x;

    if (i<p_size){

        int start;
        int end = p[i];
        if (i==0) start = 0;
        else start = p[i-1];
        int pivot = end-1;
        int temp;
        int j = start - 1;
        for ( int k=start; k<(end-1); k++){
            if( w[k] < w[pivot]){
                j++;
                temp = w[k];
                w[k] = w[j];
                w[j] = temp;
            }
        }

        j+=1;
        temp        = w[pivot];
        w[pivot]    = w[j];
        w[j]        = temp;

        int a1 = i/2 + 1;
        int a2 = a1*3;
        int b1 = a2-3;
        int b2 = a2-2;
        int b3 = a2-1;

        q[b1]=start;
        q[b2]=j;
        q[b3]=end;


    }
}

__device__ void quicks(int * w, int s, int e){
    int size = e-s;
    int pivotpt = s;
    int pivot = w[s];



}

int main(int argc, char ** argv){
    cudaSetDevice(3);
    int i, vec_size;
    int * v; //HOST
    int * w; //DEVICE
    int * p, * dp; // partitioner
    int * q, * dq; // new partition

    vec_size=1000;
    v=(int *)malloc(sizeof(int)*vec_size);
    cudaMalloc((void **) &w, sizeof(int)*vec_size);

    for(i=0;i<vec_size;i++){
        v[i] = rand()%10000;
    }

    cudaMemcpy(w, v, sizeof(int)*vec_size,cudaMemcpyHostToDevice);
   
    quicksort<<<1000,1000>>>(w,vec_size); //dim3 ?

    cudaMemcpy(v, w, sizeof(int)*vec_size,cudaMemcpyDeviceToHost);
    for(i =0; i<vec_size;i++) printf("\n C %d == %d", i, v[i]);

}



