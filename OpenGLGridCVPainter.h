#ifndef _OPENGL_GRID_PAINTER_HEADER_H_
#define _OPENGL_GRID_PAINTER_HEADER_H_

#include <opencv2/opencv.hpp>

#include <driver_types.h>

#include <glad/glad.h>

#include <vector>

class OpenGLGridCVPainter
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
	OpenGLGridCVPainter(int rows, int cols, int width, int height, int margin = 0, int device = 0);

	~OpenGLGridCVPainter();

	int TextureWidth() { return _texture_width; }
	int TextureHeight() { return _texture_height; }

	void BeginUpdate();
	void UpdateMargin(const cv::Scalar& margin_color);
	void Update(int row, int col, const cv::cuda::GpuMat& image);
	void EndUpdate();

	void Paint();

private:
	OpenGLGridCVPainter() = delete;
	OpenGLGridCVPainter(const OpenGLGridCVPainter&) = delete;
	OpenGLGridCVPainter(OpenGLGridCVPainter&&) = delete;
	OpenGLGridCVPainter& operator=(const OpenGLGridCVPainter&) = delete;
	OpenGLGridCVPainter& operator=(OpenGLGridCVPainter&&) = delete;
};
#endif

