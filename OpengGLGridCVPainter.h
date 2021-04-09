#ifndef _OPENGL_GRID_PAINTER_HEADER_H_
#define _OPENGL_GRID_PAINTER_HEADER_H_

#include <glad/glad.h>

#include <opencv2/opencv.hpp>

#include <driver_types.h>

#include <vector>

class OpengGLGridCVPainter
{
private:
	GLuint _vertex_array, _vertex_object, _element_object;

	GLuint _pixel_buffer;

	GLuint _texture;

	GLuint _program;

private:
	cudaGraphicsResource_t _cuda_resource;

private:
	int _rows;
	int _cols;
	int _width;
	int _height;
	int _margin;

private:
	int _texture_width;
	int _texture_height;

private:
	int _step;
	void* _data;

public:
	OpengGLGridCVPainter(int rows, int cols, int width, int height, int margin = 0, int device = 0);

	~OpengGLGridCVPainter();

	void BeginUpdate();
	void UpdateMargin(const cv::Scalar& margin_color);
	void Update(int row, int col, const cv::cuda::GpuMat& image);
	void EndUpdate();

	void Paint();

private:
	OpengGLGridCVPainter() = delete;
	OpengGLGridCVPainter(const OpengGLGridCVPainter&) = delete;
	OpengGLGridCVPainter(OpengGLGridCVPainter&&) = delete;
	OpengGLGridCVPainter& operator=(const OpengGLGridCVPainter&) = delete;
	OpengGLGridCVPainter& operator=(OpengGLGridCVPainter&&) = delete;
};
#endif

