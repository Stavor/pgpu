#include <stdio.h>
#include <cuda_runtime.h>

static int threadsPerBlock = 256;

__global__ void reductionByAdd(int *a, int *res, int elCnt) {
	int firstThreadBlockId = blockIdx.x * blockDim.x;	
	int threadId = firstThreadBlockId + threadIdx.x;

	for(int i = 1; i < blockDim.x; i <<= 1) {
		if(threadId % (2 * i) == 0 && threadId < elCnt && threadId + i < elCnt) {
			a[threadId] += a[threadId + i];
		}
		__syncthreads();
	}
	if(threadId == firstThreadBlockId)
		res[blockIdx.x] = a[firstThreadBlockId];
}

void checkCudaError(cudaError error) {
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Failed! (error cdoe %s)\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

__host__ void reductionAdd(int *a, int *res, int elCnt) {
	int blocksPerGrid = (elCnt + threadsPerBlock - 1) / threadsPerBlock;
	size_t sizeRes = blocksPerGrid * sizeof(int);
	
	int *result = NULL;
	cudaError_t error = cudaMalloc((void**)&result, sizeRes);
	checkCudaError(error);

	reductionByAdd<<<blocksPerGrid, threadsPerBlock>>>(a, result, elCnt);
	if(elCnt > threadsPerBlock) {
		reductionAdd(result, res, blocksPerGrid);
		checkCudaError(error);
	} else {
		error = cudaMemcpy(res, result, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaError(error);
	}

	error =  cudaFree(result);
	checkCudaError(error);
}

int main() {
	int elementCnt = 7000000;

	size_t size1 = elementCnt * sizeof(int);
	
	int *hosta = (int *)malloc(size1);
	int *ans = (int *)malloc(sizeof(int));

	if(hosta == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < elementCnt; ++i) {
		hosta[i] = rand() % 100;
	}

	int *deva1 = NULL;
	cudaError_t error = cudaMalloc((void**)&deva1, size1);
	checkCudaError(error);

	error = cudaMemcpy(deva1, hosta, size1, cudaMemcpyHostToDevice);
	checkCudaError(error);

    reductionAdd(deva1, ans, elementCnt);

    error = cudaGetLastError();
    checkCudaError(error);

    //Check
    int res = 0;
    for (int i = 0; i < elementCnt; ++i) {
    	res += hosta[i];
    }
    printf("%d\n", res);
    printf("%d\n", *ans);
    printf("%d\n", abs(res - *ans));

    if (abs(res - *ans) != 0) {
            fprintf(stderr, "Result verification failed!\n");
            exit(EXIT_FAILURE);
    }

    error = cudaFree(deva1);
	checkCudaError(error);

    free(hosta);
    free(ans);

    error = cudaDeviceReset();
	checkCudaError(error);

    printf("Done\n");
    return 0;
}