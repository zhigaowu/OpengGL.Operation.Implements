#ifndef _OPENGL_SHAPE_PAINTER_HEADER_H_
#define _OPENGL_SHAPE_PAINTER_HEADER_H_

#include <glad/glad.h>

class OpenGLShapePainter
{
private:
	GLuint _vertex_array, _vertex_object, _element_object;
	GLuint _program;
	
private:
	int _texture_width;
	int _texture_height;
	
public:
	OpenGLShapePainter(int texture_width, int texture_height);

	~OpenGLShapePainter();

	void Paint();

private:
	OpenGLShapePainter() = delete;
	OpenGLShapePainter(const OpenGLShapePainter&) = delete;
	OpenGLShapePainter(OpenGLShapePainter&&) = delete;
	OpenGLShapePainter& operator=(const OpenGLShapePainter&) = delete;
	OpenGLShapePainter& operator=(OpenGLShapePainter&&) = delete;
};
#endif

