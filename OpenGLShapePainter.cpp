#include "OpenGLShapePainter.h"


#include <stdio.h>
#include <stdlib.h>
// utility function for checking shader compilation/linking errors.
// ------------------------------------------------------------------------
static void checkShaderCompileResult(GLuint object, const char *const file, int const line)
{
	GLint code;
	GLchar desc[256];
	glGetShaderiv(object, GL_COMPILE_STATUS, &code);
	if (!code)
	{
		glGetShaderInfoLog(object, 256, NULL, desc);
		fprintf(stderr, "Shader error at %s:%d code=%d(%s)\n", file, line,
			static_cast<int>(code), desc);
	}
}

#define checkCompileErrors(val) checkShaderCompileResult(val, __FILE__, __LINE__)

OpenGLShapePainter::OpenGLShapePainter(int texture_width, int texture_height)
	: _program(0)
	, _half_texture_width(texture_width >> 1), _half_texture_height(texture_height >> 1)
	, _shapes()
{

	// 1. compile shaders
	// ------------------------------------------------------------------
	// 1.1. prepare shader source
	static const char* vertex_shader_src =
		"#version 330 core                   \n"
		"layout (location = 0) in vec3 aPos; \n"
		"                                    \n"
		"void main()                         \n"
		"{                                   \n"
		"	gl_Position = vec4(aPos, 1.0);   \n"
		"}                                   \n";

	static const char* fragment_shader_src =
		"#version 330 core          \n"
		"out vec4 FragColor;        \n"
		"uniform vec4 shapeColor;   \n"
		"                           \n"
		"void main()                \n"
		"{                          \n"
		"	FragColor = shapeColor; \n"
		"}                          \n";

	// 1.2. compile shaders
	// vertex shader
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_src, NULL);
	glCompileShader(vertex_shader);
	checkCompileErrors(vertex_shader);

	// fragment Shader
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_src, NULL);
	glCompileShader(fragment_shader);
	checkCompileErrors(fragment_shader);

	// shader Program
	_program = glCreateProgram();
	glAttachShader(_program, vertex_shader);
	glAttachShader(_program, fragment_shader);
	glLinkProgram(_program);
	checkCompileErrors(_program);

	// 1.3. delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
}

OpenGLShapePainter::~OpenGLShapePainter()
{
	for (const Shape& shape : _shapes)
	{
		if (shape.vertex_array)
		{
			glDeleteVertexArrays(1, &shape.vertex_array);
		}

		if (shape.vertex_object)
		{
			glDeleteBuffers(1, &shape.vertex_object);
		}
		if (shape.element_object)
		{
			glDeleteBuffers(1, &shape.element_object);
		}
	}

	if (_program)
	{
		glDeleteProgram(_program);
	}
}

void OpenGLShapePainter::Parse(const std::vector<glm::vec2>& points, const glm::vec4& color)
{
	Shape shape;
	shape.color = color;

	shape.points.resize(points.size() * 3);
	for (size_t p = 0; p < points.size(); ++p)
	{
		shape.points[p * 3 + 0] = (points[p].x - _half_texture_width) / _half_texture_width;
		shape.points[p * 3 + 1] = (_half_texture_height - points[p].y) / _half_texture_height;
	}

	std::vector<unsigned int> seqs(points.size());
	unsigned int s = 0, e = (unsigned int)(points.size()) - 1;
	int index = 0;
	do 
	{
		seqs[index++] = s++;
		seqs[index++] = e--;
	} while (s < e);

	shape.indexs.resize((points.size() - 2) * 3);
	for (size_t i = 0; i < points.size(); ++i)
	{
		shape.indexs[i * 3 + 0] = seqs[i + 0];
		shape.indexs[i * 3 + 1] = seqs[i + 1];
		shape.indexs[i * 3 + 2] = seqs[i + 2];
	}

	// vertex data and element data
	// ------------------------------------------------------------------
	// 2.1. configure vertex attributes
	glGenVertexArrays(1, &shape.vertex_array);
	glGenBuffers(1, &shape.vertex_object);
	glGenBuffers(1, &shape.element_object);

	glBindVertexArray(shape.vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, shape.vertex_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * shape.points.size(), shape.points.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.element_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * shape.indexs.size(), shape.indexs.data(), GL_STATIC_DRAW);

	// position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);

	_shapes.push_back(shape);
}

void OpenGLShapePainter::Paint()
{
	// render container
	glUseProgram(_program);
	
	for (const Shape& shape : _shapes)
	{
		glUniform4f(glGetUniformLocation(_program, "shapeColor"), shape.color.z / 255.0f, shape.color.y / 255.0f, shape.color.x / 255.0f, shape.color.w);

		glBindVertexArray(shape.vertex_array);
		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)shape.indexs.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
}
