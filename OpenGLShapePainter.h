#ifndef _OPENGL_SHAPE_PAINTER_HEADER_H_
#define _OPENGL_SHAPE_PAINTER_HEADER_H_

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>

class OpenGLShapePainter
{
private:
	struct Shape 
	{
        int indexs;
		glm::vec4 color;
		GLuint vertex_array = 0;
		GLuint vertex_object = 0;
		GLuint element_object = 0;
	};
	typedef std::vector<Shape> Shapes;

private:
	GLuint _program;
	
private:
	int _half_texture_width;
	int _half_texture_height;

private:
	Shapes _shapes;
	
public:
	OpenGLShapePainter(int texture_width, int texture_height);

	~OpenGLShapePainter();

	void Reset(int texture_width, int texture_height);

	void Parse(const std::vector<glm::vec2>& points, const glm::vec4& color);

	void Paint();

private:
	OpenGLShapePainter() = delete;
	OpenGLShapePainter(const OpenGLShapePainter&) = delete;
	OpenGLShapePainter(OpenGLShapePainter&&) = delete;
	OpenGLShapePainter& operator=(const OpenGLShapePainter&) = delete;
	OpenGLShapePainter& operator=(OpenGLShapePainter&&) = delete;
};
#endif

