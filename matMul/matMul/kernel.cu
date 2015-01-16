#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 16

cudaError_t multWithCuda(
	double* c, 
	const double* a, 
	const double* b, 
	int hc, int wc,
	int ha, int wa,
	int hb, int wb);

void init(double* data, int size);

__global__ void
matrixMul(
	double* c,
	double* a,
	double* b, 
	int hc, int wc,
	int ha, int wa, 
	int hb, int wb)
{
   int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
 
   double value = 0;
   for (int i = 0; i < wa; ++i)
   {
      double elementA = a[ty * wa + i];
      double elementB = b[i * wb + tx];
      value += elementA * elementB;
   }
 
   c[ty * wa + tx] = value;
}

void debug_print(double* data, int size, int rowSize)
{
	printf("\n==================================================\n");
    for(int i = 0; i < size; i++)
    {
       printf("%f ", data[i]);
       if(((i + 1) % rowSize) == 0)
          printf("\n");
    }
}

bool check_result(double *c, double *a, double *b, int hc, int wc, int ha, int wa, int hb, int wb)
{
	bool isCorrect = true;
	double eps = 1e-6;

	for(int i = 0; i < ha; i++) //ha = wb
	{
		for(int j = 0; j < wb; j++) 
		{
			double cur = 0;

			for(int k = 0; k < wa; k++) //wa = hb
				cur += a[i * wa + k] *  b[k * wb + j];
			if(fabs(cur - c[i * wc + j]) > eps)
				isCorrect = false;
		}
	}
		
	return isCorrect;
}

int main(int argc, char *argv[])
{
	//int n = argc == 1 ? 2048 : atoi(argv[1]);
	int n = 1024;
	int ha = n;
	int wa = n;
	int hb = n;
	int wb = n;
	int hc = n;
	int wc = n;

 
    int sizeA = ha * wa;
    int sizeB = hb * wb;
    int sizeC = hc * wc;

    double* a = (double*) malloc(sizeof(double) * sizeA);
    double* b = (double*) malloc(sizeof(double) * sizeB);
    double* c = (double*) malloc(sizeof(double) * sizeC);
 
	srand(146);
    init(a, sizeA);
    init(b, sizeB);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time = 0;

	cudaEventRecord(start, 0);

	cudaError_t cudaStatus = multWithCuda(c, a, b,
		hc, wc,
		ha, wa,
		hb, wb);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	bool isSuccess = check_result(c, a, b, hc, wc, ha, wa, hb, wb);
	printf("%dKB. %s. ElapsedTime is %.6f\n", sizeof(double) * sizeC / 1024, isSuccess ? "Success" : "Fail", time);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	//debug_print(c, sizeC, wc);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	free(a);
    free(b);
    free(c);

    return 0;
}

void init(double* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (double)RAND_MAX;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multWithCuda(
	double *c, 
	const double *a, 
	const double *b, 
	int hc, int wc,
	int ha, int wa,
	int hb, int wb)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    cudaError_t cudaStatus;

	int sizeC = hc * wc;
	int sizeA = ha * wa;
	int sizeB = hb * wb;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, sizeC * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeA * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeB * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeB * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(wc / threads.x, hc / threads.y);
 
    // Launch a kernel on the GPU with one thread for each element.
    matrixMul<<< grid, threads >>>(dev_c, dev_a, dev_b, 
		hc, wc,
		ha, wa,
		hb, wb);
 
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, sizeC * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

	return cudaStatus;
}