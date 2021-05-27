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

	struct TextInfo
	{
		std::wstring text;

		float scale = 1.0f;
		glm::vec2 font_position;
		glm::vec3 font_color = { 0.0f, 0.0f, 0.0f };

		glm::vec2 link_position = {-1.0f, -1.0f};
		glm::vec3 link_color = { 0.0f, 0.0f, 0.0f };
	};
	typedef std::vector<TextInfo> TextInfos;

private:
	GLuint _font_vertex_array, _font_vertex_object;
	GLuint _font_program;

private:
	GLuint _indicator_vertex_array, _indicator_vertex_object, _indicator_element_object;;
	GLuint _indicator_program;

private:
	std::string _font_name;
	Characters _characters;

private:
	int _half_texture_width;
	int _half_texture_height;

private:
	int _font_height;

private:
	TextInfos _text_infos;
	
public:
	OpenGLTextPainter(int texture_width, int texture_height, int font_height = 48, const char* font_name = "arial");

	~OpenGLTextPainter();

	void Parse(const std::vector<std::wstring>& texts);
	void Paint(const std::wstring& text, const glm::vec2& font_position, float scale, const glm::vec3& font_color, const glm::vec2& link_position = { -1.0f, -1.0f }, const glm::vec3& link_color = { 0.0f, 0.0f, 0.0f });

	void Parse(const std::wstring& text, const glm::vec2& font_position, float scale, const glm::vec3& font_color, const glm::vec2& link_position = { -1.0f, -1.0f }, const glm::vec3& link_color = { 0.0f, 0.0f, 0.0f });
	void Paint();

private:
	OpenGLTextPainter() = delete;
	OpenGLTextPainter(const OpenGLTextPainter&) = delete;
	OpenGLTextPainter(OpenGLTextPainter&&) = delete;
	OpenGLTextPainter& operator=(const OpenGLTextPainter&) = delete;
	OpenGLTextPainter& operator=(OpenGLTextPainter&&) = delete;
};
#endif

