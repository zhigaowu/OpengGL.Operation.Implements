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
	: _vertex_array(0), _vertex_object(0), _element_object(), _program(0)
	, _texture_width(texture_width), _texture_height(texture_height)
{

	// 1. compile shaders
	// ------------------------------------------------------------------
	// 1.1. prepare shader source
	static const char* vertex_shader_src =
		"#version 330 core                            \n"
		"layout (location = 0) in vec3 aPos;          \n"
		"                                             \n"
		"void main()                                  \n"
		"{                                            \n"
		"	gl_Position = vec4(aPos, 1.0);            \n"
		"}                                            \n";

	static const char* fragment_shader_src =
		"#version 330 core                           \n"
		"out vec4 FragColor;                         \n"
		"                                            \n"
		"void main()                                 \n"
		"{                                           \n"
		"	FragColor = vec4(1.0f, 0.5f, 0.2f, 0.5f);\n"
		"}                                           \n";

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

	// 2. vertex data and element data
	// ------------------------------------------------------------------
	static float vertice_data[] = {
		// positions
		-0.5f,  0.5f, 0.0f, // top left
		 0.5f,  0.5f, 0.0f, // top right
		-0.5f, -0.5f, 0.0f, // bottom left
		 0.5f, -0.5f, 0.0f, // bottom right
	};
	static unsigned int indice_triangle_data[] = {
		0, 1, 2, // first triangle
		1, 2, 3  // second triangle
	};

	// 2.1. configure vertex attributes
	glGenVertexArrays(1, &_vertex_array);
	glGenBuffers(1, &_vertex_object);
	glGenBuffers(1, &_element_object);

	glBindVertexArray(_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, _vertex_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertice_data), vertice_data, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _element_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indice_triangle_data), indice_triangle_data, GL_STATIC_DRAW);

	// position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);
}

OpenGLShapePainter::~OpenGLShapePainter()
{
	if (_vertex_array)
	{
		glDeleteVertexArrays(1, &_vertex_array);
	}

	if (_vertex_object)
	{
		glDeleteBuffers(1, &_vertex_object);
	}
	if (_element_object)
	{
		glDeleteBuffers(1, &_element_object);
	}

	if (_program)
	{
		glDeleteProgram(_program);
	}
}

void OpenGLShapePainter::Paint()
{
	// render container
	glUseProgram(_program);
	glBindVertexArray(_vertex_array);

	glDrawElements(GL_TRIANGLE_STRIP, 6, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}
