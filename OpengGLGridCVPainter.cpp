#include "OpengGLGridCVPainter.h"

#include <cuda_gl_interop.h>

#ifdef _MSC_VER
#pragma comment(lib, "opencv_core450.lib")
#pragma comment(lib, "cudart.lib")
#endif

template <typename T>
static void checkCudaCallResult(T result, char const *const func, const char *const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), cudaGetErrorString(result), func);
		exit(EXIT_FAILURE);
	}
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) checkCudaCallResult((val), #val, __FILE__, __LINE__)
#endif

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
		exit(EXIT_FAILURE);
	}
}

#define checkCompileErrors(val) checkShaderCompileResult(val, __FILE__, __LINE__)

OpengGLGridCVPainter::OpengGLGridCVPainter(int rows, int cols, int width, int height, int margin/* = 0*/, int device/* = 0*/)
	: _vertex_array(0), _vertex_object(0), _element_object(), _texture(0), _program(0)
	, _cuda_resource(nullptr)
	, _rows(rows), _cols(cols), _width(width), _height(height), _margin(margin)
	, _texture_width(_width* cols + _margin * (cols + 1)), _texture_height(_height* rows + _margin * (rows + 1))
	, _step(_texture_width * 3), _data(nullptr)
{

	// 1. compile shaders
	// ------------------------------------------------------------------
	// 1.1. prepare shader source
	static const char* vertex_shader_src =
		"#version 330 core                            \n"
		"layout (location = 0) in vec3 aPos;          \n"
		"layout (location = 1) in vec2 aTexCoord;     \n"
		"                                             \n"
		"out vec2 TexCoord;                           \n"
		"                                             \n"
		"void main()                                  \n"
		"{                                            \n"
		"	gl_Position = vec4(aPos, 1.0);            \n"
		"	TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
		"}                                            \n";

	static const char* fragment_shader_src =
		"#version 330 core                          \n"
		"out vec4 FragColor;                        \n"
		"                                           \n"
		"in vec2 TexCoord;                          \n"
		"                                           \n"
		"uniform sampler2D texture1;                \n"
		"                                           \n"
		"void main()                                \n"
		"{                                          \n"
		"	FragColor = texture(texture1, TexCoord);\n"
		"}                                          \n";

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
		// positions          // texture coordinates
		-1.0f,  1.0f, 0.0f,   0.0f, 0.0f, // top left,     bottom left
		 1.0f,  1.0f, 0.0f,   1.0f, 0.0f, // top right,    bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 1.0,  // bottom left,  top left 
		 1.0f, -1.0f, 0.0f,   1.0f, 1.0f  // bottom right, top right 
	};
	static unsigned int indice_data[] = {
		0, 1, 3, // first triangle
		0, 2, 3  // second triangle
	};

	// 2.1. configure vertex attributes
	glGenVertexArrays(1, &_vertex_array);
	glGenBuffers(1, &_vertex_object);
	glGenBuffers(1, &_element_object);

	glBindVertexArray(_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, _vertex_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertice_data), vertice_data, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _element_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indice_data), indice_data, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texture coordinates attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// 3. configure pixel buffer
	glGenBuffers(1, &_pixel_buffer);
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixel_buffer);
	// Allocate data for the buffer. 3-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, _step * _texture_height, NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// 4. configure texture
	glEnable(GL_TEXTURE_2D);
	// Generate a texture ID
	glGenTextures(1, &_texture);
	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_2D, _texture);

	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, _texture_width, _texture_height, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// 5. register pixel buffer for INTER_OP
	cudaSetDevice(device);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cuda_resource, _pixel_buffer, cudaGraphicsMapFlagsWriteDiscard));
}

OpengGLGridCVPainter::~OpengGLGridCVPainter()
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
	if (_pixel_buffer)
	{
		glDeleteBuffers(1, &_pixel_buffer);
	}

	if (_texture)
	{
		glDeleteTextures(1, &_texture);
	}

	if (_program)
	{
		glDeleteProgram(_program);
	}

	if (_cuda_resource)
	{
		cudaGraphicsUnregisterResource(_cuda_resource);
	}
}

void OpengGLGridCVPainter::BeginUpdate()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &_cuda_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&_data, &num_bytes, _cuda_resource));
}

extern cudaError_t cuda_update_horizontal_margin(void *dst, int step, int row, int texture_width, int height, int margin, unsigned char b, unsigned char g, unsigned char r);
extern cudaError_t cuda_update_vertical_margin(void *dst, int step, int col, int width, int texture_height, int margin, unsigned char b, unsigned char g, unsigned char r);
void OpengGLGridCVPainter::UpdateMargin(const cv::Scalar& margin_color)
{
	if (_margin > 0)
	{
		for (int r = 0; r <= _rows; ++r)
		{
			checkCudaErrors(cuda_update_horizontal_margin(_data, _step, r, _texture_width, _height, _margin, (unsigned char)margin_color[0], (unsigned char)margin_color[1], (unsigned char)margin_color[2]));
		}

		for (int c = 0; c <= _cols; ++c)
		{
			checkCudaErrors(cuda_update_vertical_margin(_data, _step, c, _width, _texture_height, _margin, (unsigned char)margin_color[0], (unsigned char)margin_color[1], (unsigned char)margin_color[2]));
		}
	}
}

extern cudaError_t cuda_update(void *dst, int step_dst, int row, int col, int margin, void* src, int step_src, int width, int height);
void OpengGLGridCVPainter::Update(int row, int col, const cv::cuda::GpuMat& image)
{
	checkCudaErrors(cuda_update(_data, _step, row, col, _margin, image.data, (int)(image.step), _width, _height));
}

void OpengGLGridCVPainter::EndUpdate()
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &_cuda_resource, 0));

	// Select the appropriate buffer 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixel_buffer);
	// Select the appropriate texture
	glBindTexture(GL_TEXTURE_2D, _texture);
	// Make a texture from the buffer
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _texture_width, _texture_height, GL_BGR, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void OpengGLGridCVPainter::Paint()
{
	glBindTexture(GL_TEXTURE_2D, _texture);
	// render container
	glUseProgram(_program);
	glBindVertexArray(_vertex_array);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
