#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

const int BlockSize = 2;
const int ThreadNum = 5;
const int N = 10;

__global__ void increment_gpu(int a[],const int b, const int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
         a[idx]=a[idx]+b;
}

int main()
{
    int a[N];
	int *d_a = 0;
	int i = 0;
	const int b = 10;
	
	cout << "a[N] array before scaling: [";
	for(i=0;i<N;i++)
	{
		a[i] = i;
		cout << a[i] << " ";
	}
	cout << "]" << endl;
	
	cudaMalloc((void**) &d_a,sizeof(int)*N);
	cudaMemcpy(d_a,a,sizeof(int)*N,cudaMemcpyHostToDevice);
	
    dim3 dimBlock(BlockSize);
    dim3 dimGrid(ThreadNum);
    increment_gpu<<<dimGrid,dimBlock>>>(d_a,b,N);
	cudaDeviceSynchronize();
	
	cudaMemcpy(a,d_a,sizeof(int)*N,cudaMemcpyDeviceToHost);
	
	cout << "a[N] array after scaling: [";
	for(i=0;i<N;i++)
	{
		cout << a[i] << " ";
	}
	cout << "]" << endl;
	
	return 0;
}
