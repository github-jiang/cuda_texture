#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include "cuda_texture_types.h"
#include<math.h>

#include "cuda.h"
//#include "cpu_anim.h" //调用texture的时候必须加上这个头文件
#define size 256

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void transformKernel(float* input, float* output, int width, int height, float theta)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float u = x / (float)width;
	float v = y / (float)height;
	// 坐标转换
	u -= 0.5f;
	v -= 0.5f;
	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
	int col = tu*width;
	int row = tv*height;
	//output[y*width + x] = input[0];
	output[y*width + x] = tex2D(texRef, tu, tv);
}
void main()
{
	int width = 3840, height = 1920;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray*cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);
	float*h_data = (float*)malloc(width*height*sizeof(float));
	for (int i = 0; i<height; ++i)
	{
		for (int j = 0; j<width; ++j)
		{
			h_data[i*width + j] = i*width + j;
		}
	}
	cudaMemcpyToArray(cuArray, 0, 0, h_data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.addressMode[1] = cudaAddressModeWrap;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = true;
	cudaBindTextureToArray(texRef, cuArray, channelDesc);
	float*output;
	cudaMalloc(&output, width*height*sizeof(float));
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
	float angle = 30;

	float *input = NULL;
	cudaMalloc(&input, width*height*sizeof(float));
	cudaMemcpy(input, h_data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	transformKernel << <dimGrid, dimBlock >> >(input, output, width, height, angle);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, NULL);

	for (int i = 0; i < 1000; i++)
	{
		transformKernel << <dimGrid, dimBlock >> >(input, output, width, height, angle);
		cudaGetLastError();
	}
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float costtime;
	cudaEventElapsedTime(&costtime, start, stop);
	printf("kernel run time: %f ms\n", costtime);

	float*hostPtr = (float*)malloc(sizeof(float)*width*height);
	cudaMemcpy(hostPtr, output, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	/*for (int i = 0; i<height; ++i)
	{
		for (int j = 0; j<width; ++j)
		{
			printf("%f\n", hostPtr[i*width + j]);
		}
		printf("\n");
	}*/
	free(hostPtr);
	cudaFreeArray(cuArray);
	cudaFree(output);
	system("pause");
}