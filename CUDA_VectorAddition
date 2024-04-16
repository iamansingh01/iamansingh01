
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include<iostream>
using namespace std;
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

// Function to calculate the sum of the vector and print the result
double sumVector(double *vector, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += vector[i];
    return sum;
}

int main(int argc, char *argv[])
{
    // Size of vectors
    int n = 0;
    cout<<"Enter the number of element:";
	cin>>n;
    

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    // Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++)
    {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Measure CPU execution time
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Execute the kernel on the GPU
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Synchronize the GPU
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    double timeTakenGPU = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Measure CPU execution time
    gettimeofday(&start, NULL);

    // Perform the vector addition on the CPU
    for (i = 0; i < n; i++)
    {
        h_c[i] = h_a[i] + h_b[i];
    }

    gettimeofday(&end, NULL);
    double timeTakenCPU = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // Calculate the speedup
    double speedup = timeTakenGPU / timeTakenCPU;

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sumGPU = sumVector(h_c, n);
    printf("GPU Result: %f\n", sumGPU / n);
    printf("GPU Time: %f seconds\n", timeTakenGPU);

    double sumCPU = sumVector(h_c, n);
    printf("CPU Result: %f\n", sumCPU / n);
    printf("CPU Time: %f seconds\n", timeTakenCPU);

    printf("Speedup: %f\n", speedup);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
