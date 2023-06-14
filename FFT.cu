//one dimension FFT
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Complex.cu"
#include <iostream>
#include <string>
#include <stdlib.h>
#include<stdio.h>
#include <time.h>


using namespace std;


int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

__device__ int br(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}

__device__ void Bufferfly(Complex* a, Complex* b, Complex factor) { 
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k) k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[br(tid + j, bits)], &nums[br(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[br(tid, bits)];
}

void printSequence(Complex nums[], const int N) {
    
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real, imag = nums[i].imag;
        if (imag == 0) 
            printf("%.4f", real);
        else {
            if (imag > 0) 
                printf("%.4f+%.4fi", real, imag);
            else 
                printf("%.4f%.4fi", real, imag);
        }
        
        printf("\n");
    }
    
}

int main() {
    srand(time(0));

    const int TPB = 128;
    const int N = 128 * 32;        //FFT point
    
    const int bits = GetBits(N);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    Complex* nums = (Complex*)malloc(sizeof(Complex) * N), * dNums, * dResult;
    for (int i = 0; i < N; ++i) {
        nums[i] = Complex::GetRandomReal();
    }
    printf("Length of Sequence: %d\n", N);
    cout<<"Before FFT"<<endl;
    printSequence(nums, N);

    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(TPB);
    dim3 blockNum(N / TPB);
    

    //----------------------Start Record the time---------------------
    cudaEventRecord(start, 0); //keep start time

    FFT << <threadPerBlock, blockNum >> > (dNums, dResult, N, bits);


    //----------------------END Record the time---------------------
    cudaEventRecord(stop, 0); //keep stop time
    cudaEventSynchronize(start); //wait stop event
    cudaEventSynchronize(stop); //wait stop event
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time (GPU) : %f ms \n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //---------------------------------------------------------------
    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    cout<<"After FFT"<<endl;
    printSequence(nums, N);
    free(nums);
    cudaFree(dNums);
    cudaFree(dResult);

}