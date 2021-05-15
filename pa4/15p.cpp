#include<iostream>
#include<vector>
using namespace std;

vector< vector<int> > v;

int main(){

    freopen("i.txt", "r", stdin);

    v.resize(4);

    for ( int i=0; i<v.size();i++)
        v[i].resize(4);

    for(int i=0; i<4; i++) 
        for(int j=0; j<4; j++) 
            cin>>v[i][j];


}