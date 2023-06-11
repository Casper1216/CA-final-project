//one dimension DIT FFT
#include "Complex_host.cpp"
#include <iostream>
#include<vector>
#include <string>
#include <stdlib.h>
#include<stdio.h>
#include <time.h>
using namespace std;
//#include <Windows.h>

int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

int br(int i, int bits) {   //bit reverse
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}




vector<Complex> FFT(vector<Complex>& x) {
    if(x.size()==1)
        return x;
    
    int N = x.size();
    vector<Complex> Y(N);
    Complex W_n;
    W_n = W_n.W(N);


    Complex w(1,0);

    vector<Complex> x_even(N/2);
    vector<Complex> x_odd(N/2);

    for(int i=0;i<N/2;i++){
        x_even[i] = x[2*i];
        x_odd[i] = x[1+2*i];
    }
    vector<Complex> y_even = FFT(x_even);
    vector<Complex> y_odd = FFT(x_odd);
    
    for(int i=0;i<N/2;i++){
        Y[i] = y_even[i] + w*y_odd[i];
        Y[i+N/2] = y_even[i] - w*y_odd[i];
        w = w*W_n;
    }

    return Y;
}

void printSequence(vector<Complex>& nums, const int N) {
    
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real, imag = nums[i].imag;
        if (imag == 0) printf("%.4f", real);
        else {
            if (imag > 0) printf("%.4f+%.4fi", real, imag);
            else printf("%.4f%.4fi", real, imag);
        }
        if (i != N - 1) printf(" ");
    }
    
}

int main() {
    srand(time(0));
    
    const int N = 1024*32;        //FFT points

    vector<Complex> nums(N);
    vector<Complex> res(N);
    for (int i = 0; i < N; ++i) {
        nums[i] = nums[i].GetRandomReal();
    }
    // printf("Length of Sequence: %d\n", N);
    // cout<<"Before FFT"<<endl;
    // printSequence(nums, N);
	// cout<<endl;

    // Start Record the time
    time_t  start = clock();
    //float s = GetTickCount();
    //***************************************************************************
    
    res = FFT(nums);

    //***************************************************************************

    // Record the end time
    time_t end = clock();
    double diff = end - start; // ms
    printf(" %f  sec\n", diff / CLOCKS_PER_SEC );



    // cout<<"After FFT"<<endl;
    // printSequence(res, N);
    // cout<<endl;
    // printf("END \n");

    
}
