#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "sort.h"

int getLenFloatsArrByMBSize(int mbSize)
{
	return (mbSize * 1024 * 1024) / sizeof(float); 
}

float get_rand_float()
{
	return (float)rand()/RAND_MAX;
}

void init_array(float* theArr, int len)
{
	srand(2014);
	for(int i = 0; i < len; i++)
	{
		theArr[i] = get_rand_float();
	}
}

bool check_result(float *result, int len)
{	
	//for(int i = len - 1; i >= len - 16; i--)
	//{
	//	printf("%.3f ", result[i]);
	//}
	//printf("\n");

	int failsCnt = 0;
	for(int i = 1; i < len; i++)
	{
		if(result[i] < result[i-1])
			failsCnt++;
	}
	printf("fails: %d\n", failsCnt);
	return failsCnt == 0;
}

void runTest(float* theArr, int len, int mbSize)
{
	float *cudaArr;
	size_t size = len * sizeof(float);

	cudaMalloc((void**) &cudaArr, size);
	cudaMemcpy(cudaArr, theArr, size, cudaMemcpyHostToDevice);
	float *result = (float*) malloc(size);
 
	check_result(theArr, len);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time = 0;

	cudaEventRecord(start, 0);

	bitonicSort(cudaArr, len);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(result, cudaArr, size, cudaMemcpyDeviceToHost);

	bool isSuccess = check_result(result, len);
	printf("%dMB. %d Elements. %s. ElapsedTime is %.6f\n", mbSize, len , isSuccess ? "Success" : "Fail", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(cudaArr);
	free(result);
}

void runTests(int *ascendingSizes, int len)
{
	int maxSize = getLenFloatsArrByMBSize(ascendingSizes[len - 1]);
	float* theArr = (float*) malloc( maxSize * sizeof(float));
	init_array(theArr, maxSize);

	for(int i = 0; i < len; i++)
	{
		runTest(theArr, getLenFloatsArrByMBSize(ascendingSizes[i]), ascendingSizes[i]);
	}
}

int main()
{
	int ascendingSizes[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
	runTests(ascendingSizes, 8);
}