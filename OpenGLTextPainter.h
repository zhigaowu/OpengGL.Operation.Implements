#ifndef _OPENGL_TEXT_PAINTER_HEADER_H_
#define _OPENGL_TEXT_PAINTER_HEADER_H_

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <unordered_map>

class OpenGLTextPainter
{
private:
	struct Character {
		GLuint texture;  // ID handle of the glyph texture
		glm::ivec2   size;       // Size of glyph
		glm::ivec2   bearing;    // Offset from baseline to left/top of glyph
		unsigned int advance;    // Offset to advance to next glyph
	};

	typedef std::unordered_map<wchar_t, Character> Characters;

private:
	GLuint _font_vertex_array, _font_vertex_object;
	GLuint _font_program;

private:
	GLuint _indicator_vertex_array, _indicator_vertex_object, _indicator_element_object;;
	GLuint _indicator_program;

private:
	Characters _characters;

private:
	int _half_texture_width;
	int _half_texture_height;

private:
	int _font_height;
	
public:
	OpenGLTextPainter(int texture_width, int texture_height);

	~OpenGLTextPainter();

	void Parse(const std::vector<std::wstring>& texts, const char* font_name = "arial");
	void Paint(const std::wstring& text, float x, float y, float scale, unsigned char b, unsigned char g, unsigned char r, float link_x, float link_y);

private:
	OpenGLTextPainter() = delete;
	OpenGLTextPainter(const OpenGLTextPainter&) = delete;
	OpenGLTextPainter(OpenGLTextPainter&&) = delete;
	OpenGLTextPainter& operator=(const OpenGLTextPainter&) = delete;
	OpenGLTextPainter& operator=(OpenGLTextPainter&&) = delete;
};
#endif

