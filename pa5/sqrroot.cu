#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

    thrust::host_vector<float> H;
    float r;
    for(int i=0; i<100; i++ ){
        r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        H.push_back(r);
    }

    thrust::device_vector<float> D=H;
    
    thrust::sort(D.begin(),D.end());

    H=D;

    for(int i=0; i<100; i++ ){
        printf("%f ", H[i]);
    }

    return 0;
}
