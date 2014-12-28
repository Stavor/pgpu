#include <cuda_runtime.h>

typedef unsigned int uint;

uint max_threads = 1024;
uint max_blocks = 65535;

__global__ 
void bitonicSortStep(float *cudaArr, uint i, uint j)
{
	uint tid = threadIdx.x + blockDim.x * blockIdx.x;
	uint mate = tid ^ j;
	if (tid < mate) 
	{
		if((tid & i) == 0)
		{
			if(cudaArr[tid] > cudaArr[mate])
			{
				float temp = cudaArr[tid];
				cudaArr[tid] = cudaArr[mate];
				cudaArr[mate] = temp;
			}
		}
		else
		{
			if(cudaArr[tid] < cudaArr[mate])
			{
				float temp = cudaArr[tid];
				cudaArr[tid] = cudaArr[mate];
				cudaArr[mate] = temp;
			}
		}
	}
}


//len должно быть степенью 2
extern "C" void bitonicSort(float *cudaArr, uint len)
{
	uint threads = max_threads;
	uint blocks = len / threads;
	if(len % threads != 0)
		blocks++;
	
	if(blocks > max_blocks)
		throw 1;

	for(uint i = 2; i <= len; i <<= 1)
	{
		for(uint j = i>>1; j > 0; j >>= 1)
		{
			bitonicSortStep<<<blocks, threads>>>(cudaArr, i, j);
			cudaThreadSynchronize();
		}
	}
}
