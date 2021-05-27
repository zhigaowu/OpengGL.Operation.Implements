#include "OpenGLTextPainter.h"

#include <ft2build.h>
#include FT_FREETYPE_H

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
		fprintf(stderr, "shader error at %s:%d code=%d(%s)\n", file, line,
			static_cast<int>(code), desc);
	}
}

#define checkCompileErrors(val) checkShaderCompileResult(val, __FILE__, __LINE__)

OpenGLTextPainter::OpenGLTextPainter(int texture_width, int texture_height)
	: _font_vertex_array(0), _font_vertex_object(0), _font_program(0)
	, _indicator_vertex_array(0), _indicator_vertex_object(0), _indicator_program(0)
	, _characters()
	, _half_texture_width(texture_width >> 1), _half_texture_height(texture_height >> 1)
	, _font_height(48)
{

	// 1. compile shaders
	// ------------------------------------------------------------------
	// 1.1. prepare font shader source
	static const char* font_vertex_shader_src =
		"#version 330 core                            \n"
		"layout (location = 0) in vec4 vertex;        \n"
		"out vec2 TexCoords;                          \n"
		"                                             \n"
		"void main()                                  \n"
		"{                                            \n"
		"	gl_Position = vec4(vertex.xy, 0.0, 1.0);  \n"
		"	TexCoords = vertex.zw;                    \n"
		"}                                            \n";

	static const char* font_fragment_shader_src =
		"#version 330 core                                                  \n"
		"in vec2 TexCoords;                                                 \n"
		"out vec4 color;                                                    \n"
		"                                                                   \n"
		"uniform sampler2D text;                                            \n"
		"uniform vec3 textColor;                                            \n"
		"                                                                   \n"
		"void main()                                                        \n"
		"{                                                                  \n"
		"	vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r); \n"
		"	color = vec4(textColor, 1.0) * sampled;                         \n"
		"}                                                                  \n";

	// 1.2. compile font shaders
	// vertex shader
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &font_vertex_shader_src, NULL);
	glCompileShader(vertex_shader);
	checkCompileErrors(vertex_shader);

	// fragment Shader
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &font_fragment_shader_src, NULL);
	glCompileShader(fragment_shader);
	checkCompileErrors(fragment_shader);

	// shader Program
	_font_program = glCreateProgram();
	glAttachShader(_font_program, vertex_shader);
	glAttachShader(_font_program, fragment_shader);
	glLinkProgram(_font_program);
	checkCompileErrors(_font_program);

	// 1.3. delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	// 1.4. prepare indicator shader source
	static const char* indicator_vertex_shader_src =
		"#version 330 core                      \n"
		"layout (location = 0) in vec3 aPos;    \n"
		"                                       \n"
		"void main()                            \n"
		"{                                      \n"
		"	gl_Position = vec4(aPos.xyz, 1.0);  \n"
		"}                                      \n";

	static const char* indicator_fragment_shader_src =
		"#version 330 core                \n"
		"out vec4 color;                  \n"
		"                                 \n"
		"uniform vec3 lineColor;          \n"
		"                                 \n"
		"void main()                      \n"
		"{                                \n"
		"	color = vec4(lineColor, 1.0); \n"
		"}                                \n";

	// 1.5. compile indicator shaders
	// vertex shader
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &indicator_vertex_shader_src, NULL);
	glCompileShader(vertex_shader);
	checkCompileErrors(vertex_shader);

	// fragment Shader
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &indicator_fragment_shader_src, NULL);
	glCompileShader(fragment_shader);
	checkCompileErrors(fragment_shader);

	// shader Program
	_indicator_program = glCreateProgram();
	glAttachShader(_indicator_program, vertex_shader);
	glAttachShader(_indicator_program, fragment_shader);
	glLinkProgram(_indicator_program);
	checkCompileErrors(_indicator_program);

	// 1.6. delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	// 2.1. configure font vertex attributes
	glGenVertexArrays(1, &_font_vertex_array);
	glGenBuffers(1, &_font_vertex_object);

	glBindVertexArray(_font_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, _font_vertex_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

	// position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

	glBindVertexArray(0);

	// 2.3. configure indicator vertex attributes
	unsigned int indicator_elements[6] =
	{0, 1, 1, 2, 1, 3};

	glGenVertexArrays(1, &_indicator_vertex_array);
	glGenBuffers(1, &_indicator_vertex_object);
	glGenBuffers(1, &_indicator_element_object);

	glBindVertexArray(_indicator_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, _indicator_vertex_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * 3, NULL, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indicator_element_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indicator_elements), indicator_elements, GL_STATIC_DRAW);

	// position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);
}

OpenGLTextPainter::~OpenGLTextPainter()
{
	if (_font_vertex_array)
	{
		glDeleteVertexArrays(1, &_font_vertex_array);
	}

	if (_font_vertex_object)
	{
		glDeleteBuffers(1, &_font_vertex_object);
	}

	if (_font_program)
	{
		glDeleteProgram(_font_program);
	}

	if (_indicator_vertex_array)
	{
		glDeleteVertexArrays(1, &_indicator_vertex_array);
	}

	if (_indicator_vertex_object)
	{
		glDeleteBuffers(1, &_indicator_vertex_object);
	}

	if (_indicator_program)
	{
		glDeleteProgram(_indicator_program);
	}

	for (const std::pair<wchar_t, Character>& ch : _characters)
	{
		glDeleteTextures(1, &(ch.second.texture));
	}
}

void OpenGLTextPainter::Parse(const std::vector<std::wstring>& texts, const char* font_name)
{
	do
	{
		FT_Library ft;
		// All functions return a value different than 0 whenever an error occurred
		if (FT_Init_FreeType(&ft))
		{
			fprintf(stderr, "initialize FreeType library failed\n");
			break;
		}

		char font_file[128] = { 0 };
		sprintf(font_file, "fonts/%s.ttf", font_name);

		FT_Face face;
		if (FT_New_Face(ft, font_file, 0, &face))
		{
			fprintf(stderr, "initialize font face: %s failed\n", font_file);

			FT_Done_FreeType(ft);
			break;
		}

		FT_Select_Charmap(face, FT_ENCODING_UNICODE);
		// set size to load glyphs as
		FT_Set_Pixel_Sizes(face, 0, _font_height);

		// disable byte-alignment restriction
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for (const std::wstring& text : texts)
		{
			for (const wchar_t c : text)
			{
				if (_characters.find(c) == _characters.end())
				{
					// Load character glyph 
					if (FT_Load_Char(face, c, FT_LOAD_RENDER))
					{
						fprintf(stderr, "load character(%c) glyph failed\n", c);
						continue;
					}

					// generate texture
					GLuint texture;
					glGenTextures(1, &texture);
					glBindTexture(GL_TEXTURE_2D, texture);
					glTexImage2D(
						GL_TEXTURE_2D,
						0,
						GL_RED,
						face->glyph->bitmap.width,
						face->glyph->bitmap.rows,
						0,
						GL_RED,
						GL_UNSIGNED_BYTE,
						face->glyph->bitmap.buffer
					);
					// set texture options
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
					glBindTexture(GL_TEXTURE_2D, 0);

					// now store character for later use
					Character character = {
						texture,
						glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
						glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
						static_cast<unsigned int>(face->glyph->advance.x)
					};
					_characters.insert(std::pair<wchar_t, Character>(c, character));
				}
			}
		}

		// destroy FreeType once we're finished
		FT_Done_Face(face);
		FT_Done_FreeType(ft);
	} while (false);
}

#define FONT_INDICATOR_MARGIN 10
#define INDICATOR_TARGET_MARGIN 20
#define FONT_ORIGINAL_WIDTH 96

void OpenGLTextPainter::Paint(const std::wstring& text, float x, float y, float scale, unsigned char b, unsigned char g, unsigned char r, float link_x, float link_y)
{
	// render font
	glUseProgram(_font_program);
	glUniform3f(glGetUniformLocation(_font_program, "textColor"), (float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);

	glBindVertexArray(_font_vertex_array);

	float start_x = x;
	for (const wchar_t c : text)
	{
		const Character& ch = _characters[c];

		float xpos = x + ch.bearing.x * scale;
		float ypos = y + (ch.size.y - ch.bearing.y) * scale;

		float w = ch.size.x * scale;
		float h = ch.size.y * scale;
		// update VBO for each character
		float vertices[6][4] = {
			{ (xpos - _half_texture_width) / _half_texture_width,     (_half_texture_height - (ypos - h)) / _half_texture_height,   0.0f, 0.0f },
			{ (xpos - _half_texture_width) / _half_texture_width,     (_half_texture_height - ypos) / _half_texture_height,         0.0f, 1.0f },
			{ (xpos + w - _half_texture_width) / _half_texture_width, (_half_texture_height - ypos) / _half_texture_height,         1.0f, 1.0f },

			{ (xpos - _half_texture_width) / _half_texture_width,     (_half_texture_height - (ypos - h)) / _half_texture_height,   0.0f, 0.0f },
			{ (xpos + w - _half_texture_width) / _half_texture_width, (_half_texture_height - ypos) / _half_texture_height,         1.0f, 1.0f },
			{ (xpos + w - _half_texture_width) / _half_texture_width, (_half_texture_height - (ypos - h)) / _half_texture_height,   1.0f, 0.0f }
		};
		// render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.texture);
		// update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, _font_vertex_object);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // be sure to use glBufferSubData and not glBufferData

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		x += (ch.advance >> 6) * scale; // bit shift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
	}

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// render indicator
	glUseProgram(_indicator_program);
	glUniform3f(glGetUniformLocation(_indicator_program, "lineColor"), (float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);

	glBindVertexArray(_indicator_vertex_array);

	// update VBO for each character
	float vertices[4][3] = {
		{ 0.0, 0.0f, 0.0f },
		{ 0.0, 0.0f, 0.0f },
		{ 0.0, 0.0f, 0.0f },
		{ 0.0, 0.0f, 0.0f }
	};

	if (x + INDICATOR_TARGET_MARGIN < link_x)
	{
		vertices[0][0] = (start_x - _half_texture_width) / _half_texture_width;
		vertices[0][1] = (_half_texture_height - (y + FONT_INDICATOR_MARGIN)) / _half_texture_height;

		vertices[1][0] = ((x + FONT_INDICATOR_MARGIN) - _half_texture_width) / _half_texture_width;
		vertices[1][1] = vertices[0][1];

		vertices[2][0] = vertices[1][0];
		vertices[2][1] = (_half_texture_height - (y - (_font_height >> 1))) / _half_texture_height;

		vertices[3][0] = ((link_x) - _half_texture_width) / _half_texture_width;
		vertices[3][1] = (_half_texture_height - (link_y)) / _half_texture_height;
	}
	else if (start_x - INDICATOR_TARGET_MARGIN < link_x)
	{
		vertices[0][0] = ((start_x) - _half_texture_width) / _half_texture_width;
		if (y < link_y)
		{
			vertices[0][1] = (_half_texture_height - (y + FONT_INDICATOR_MARGIN * scale)) / _half_texture_height;
		} 
		else
		{
			vertices[0][1] = (_half_texture_height - (y - (FONT_ORIGINAL_WIDTH >> 1) * scale)) / _half_texture_height;
		}

		vertices[1][0] = (((x + start_x) / 2.0f) - _half_texture_width) / _half_texture_width;
		vertices[1][1] = vertices[0][1];

		vertices[2][0] = ((x) - _half_texture_width) / _half_texture_width;
		vertices[2][1] = vertices[0][1];

		vertices[3][0] = ((link_x) - _half_texture_width) / _half_texture_width;
		vertices[3][1] = (_half_texture_height - (link_y)) / _half_texture_height;
	}
	else
	{
		vertices[0][0] = (x - _half_texture_width) / _half_texture_width;
		vertices[0][1] = (_half_texture_height - (y + FONT_INDICATOR_MARGIN)) / _half_texture_height;

		vertices[1][0] = ((start_x - FONT_INDICATOR_MARGIN) - _half_texture_width) / _half_texture_width;
		vertices[1][1] = vertices[0][1];

		vertices[2][0] = vertices[1][0];
		vertices[2][1] = (_half_texture_height - (y - (_font_height >> 1))) / _half_texture_height;

		vertices[3][0] = ((link_x) - _half_texture_width) / _half_texture_width;
		vertices[3][1] = (_half_texture_height - (link_y)) / _half_texture_height;
	}

	// update content of VBO memory
	glBindBuffer(GL_ARRAY_BUFFER, _indicator_vertex_object);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // be sure to use glBufferSubData and not glBufferData
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);
}

