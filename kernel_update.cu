
// includes, cuda
#include <cuda_runtime.h>

__global__ void update_kernel(unsigned char *dst, int step_dst, int row, int col, int margin, unsigned char* src, int step_src, int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char* bgr_dst = dst + ((row + 1) * margin + row * height + y) * step_dst + ((col + 1) * margin + col * width + x) * 3;
	unsigned char* bgr_src = src + y * step_src + x * 3;

	bgr_dst[0] = bgr_src[0];
	bgr_dst[1] = bgr_src[1];
	bgr_dst[2] = bgr_src[2];
}

cudaError_t cuda_update(void *dst, int step_dst, int row, int col, int margin, void* src, int step_src, int width, int height)
{
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	update_kernel << < grid, block >> > ((unsigned char*)dst, step_dst, row, col, margin, (unsigned char*)src, step_src, width, height);
	return cudaPeekAtLastError();
}

__global__ void update_horizontal_margin_kernel(unsigned char *dst, int step, int row, int texture_width, int height, int margin, unsigned char b, unsigned char g, unsigned char r)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < texture_width && y < margin)
	{
		unsigned char* bgr_dst = dst + (row * (margin + height) + y) * step + x * 3;

		bgr_dst[0] = b;
		bgr_dst[1] = g;
		bgr_dst[2] = r;
	}
}

cudaError_t cuda_update_horizontal_margin(void *dst, int step, int row, int texture_width, int height, int margin, unsigned char b, unsigned char g, unsigned char r)
{
	// execute the kernel
	dim3 block(4, 4, 1);
	dim3 grid((texture_width + 3) / block.x, (margin + 3) / block.y, 1);
	update_horizontal_margin_kernel << < grid, block >> > ((unsigned char*)dst, step, row, texture_width, height, margin, b, g, r);
	return cudaPeekAtLastError();
}

__global__ void update_vertical_margin_kernel(unsigned char *dst, int step, int col, int width, int texture_height, int margin, unsigned char b, unsigned char g, unsigned char r)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < margin && y < texture_height)
	{
		unsigned char* bgr_dst = dst + y * step + (col * (margin + width) + x) * 3;

		bgr_dst[0] = b;
		bgr_dst[1] = g;
		bgr_dst[2] = r;
	}
}

cudaError_t cuda_update_vertical_margin(void *dst, int step, int col, int width, int texture_height, int margin, unsigned char b, unsigned char g, unsigned char r)
{
	// execute the kernel
	dim3 block(4, 4, 1);
	dim3 grid((margin + 3) / block.x, (texture_height + 3) / block.y, 1);
	update_vertical_margin_kernel << < grid, block >> > ((unsigned char*)dst, step, col, width, texture_height, margin, b, g, r);
	return cudaPeekAtLastError();
}

