#include <stdio.h>
#include <cuda_runtime.h>
#include <functional>

using std::tr1::function;

static const int threadsPerBlock = 256;

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

__global__ void reductionByAddWithShared(int *a, int *res, int elCnt) {
	extern __shared__ int sdata[];

	int threadId = threadIdx.x;
	int glThreadId = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[threadId] = glThreadId < elCnt ? a[glThreadId] : 0;
	__syncthreads();

	for (int i = 1; i < blockDim.x; i <<= 1) {
		if(threadId % (2 * i) == 0 && threadId + i < blockDim.x) {
			sdata[threadId] += sdata[threadId + i];
		}
		__syncthreads();
	}
	
	if(threadId == 0)
		res[blockIdx.x] = sdata[0];
}

void checkCudaError(cudaError error) {
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Failed! (error code %s)\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

__host__ void recursiveAddReduction(int *a, int *res, int elCnt, function<void (int, int, int *, int *n, int)> reductionImpl) {
	int blocksPerGrid = (elCnt + threadsPerBlock - 1) / threadsPerBlock;
	size_t sizeRes = blocksPerGrid * sizeof(int);
	
	//printf("%d\n", blocksPerGrid);

	int *result = NULL;
	cudaError_t error = cudaMalloc((void**)&result, sizeRes);
	checkCudaError(error);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

	cudaEventRecord(start, 0);

	reductionImpl(blocksPerGrid, threadsPerBlock, a, result, elCnt);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time is : %f\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if(elCnt > threadsPerBlock) {
		recursiveAddReduction(result, res, blocksPerGrid, reductionImpl);
	} else {
		error = cudaMemcpy(res, result, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaError(error);
	}

	error =  cudaFree(result);
	checkCudaError(error);

    error = cudaGetLastError();
    checkCudaError(error);
}

void resultVerification(int redRes, int *m, int cnt) {
	int rightRes = 0;
    for (int i = 0; i < cnt; ++i) {
    	rightRes += m[i];
    }

    printf("%d\n", rightRes);
    printf("%d\n", redRes);
    int abSub = abs(rightRes - redRes);
    printf("%d\n", abSub);

    if (abSub != 0) {
            fprintf(stderr, "Result verification failed!\n");
            exit(EXIT_FAILURE);
    }
}

int main() {
	int elementCnt = 400000; //4M

	size_t size = elementCnt * sizeof(int);
	
	int *hosta = (int *)malloc(size);
	int *ans = (int *)malloc(sizeof(int));

	if(hosta == NULL || ans == NULL) {
		fprintf(stderr, "Failed to allocate host data!\n");
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < elementCnt; ++i) {
		hosta[i] = rand() % 100;
	}

	int *deva = NULL;
	cudaError_t error = cudaMalloc((void**)&deva, size);
	checkCudaError(error);

	error = cudaMemcpy(deva, hosta, size, cudaMemcpyHostToDevice);
	checkCudaError(error);

	printf("With Shared mem:\n");
	recursiveAddReduction(deva, ans, elementCnt, [] (int blocksPerGrid, int threadsPerBlock, int *a, int *res, int elCnt) -> void
		{
			reductionByAddWithShared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(a, res, elCnt);
		});
    resultVerification(*ans, hosta, elementCnt);
    
    *ans = -1;

    printf("\nOnly globalname:\n");
    recursiveAddReduction(deva, ans, elementCnt, [] (int blocksPerGrid, int threadsPerBlock, int *a, int *res, int elCnt) -> void
		{
			reductionByAdd<<<blocksPerGrid, threadsPerBlock>>>(a, res, elCnt);
		});
    resultVerification(*ans, hosta, elementCnt);

    error = cudaFree(deva);
	checkCudaError(error);

    free(hosta);
    free(ans);

    error = cudaDeviceReset();
	checkCudaError(error);

    printf("Done\n");
    return 0;
}