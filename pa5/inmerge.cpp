#include<iostream>
#include<vector>

void inplace_merge(std::vector<int>& v, int start, int mid, int end){
    while (start < end){
        if (v[mid] < v[start] && start < mid){
            std::swap(v[start], v[mid]);
            if (mid+1<end && v[mid+1] < v[mid]) mid++;            
           } 
        start++; 
    }
}

int main(){
    std::vector<int> v{  2, 5, 6, 10, 11, 0, 1, 3, 9, 20};
    int mid = 5; // start of the second segment
    inplace_merge(v, 0, mid, v.size()); //inplace merge to a single sorted segment.
    for(auto a: v){
        std::cout<<" "<<a;
    }
    
    return 0;
    
}