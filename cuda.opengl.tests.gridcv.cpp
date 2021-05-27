// cuda.opengl.tests.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//



#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "OpenGLGridCVPainter.h"

#include "OpenGLShapePainter.h"
#include "OpenGLTextPainter.h"

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <iostream>
#include <thread>
#include <chrono>

#pragma comment(lib, "glfw3.lib")

#pragma comment(lib, "opencv_imgproc450.lib")
#pragma comment(lib, "opencv_imgcodecs450.lib")
#pragma comment(lib, "opencv_highgui450.lib")

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
static void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

const int Width = 1920;
const int Height = 1080;

int test_gridcv_buffer(int argc, char** argv)
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	bool isFullScreen = false;
	GLFWmonitor* pMonitor = isFullScreen ? glfwGetPrimaryMonitor() : NULL;
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", pMonitor, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	std::vector<cv::cuda::GpuMat> images;
	// read image(BGR)
	std::vector<std::string> urls;
	urls.emplace_back("E:/Media/images/d56a29001c1ffe50d9f4501525971fa9.jpg");
	urls.emplace_back("E:/Media/images/1261981340895sbzeca5ic3.jpg");
	urls.emplace_back("E:/Media/images/233106f4jfwpfsbsajjjrd.jpg");
	urls.emplace_back("E:/Media/images/d56a29001c1ffe50d9f4501525971fa9.jpg");
	urls.emplace_back("E:/Media/images/d56a29001c1ffe50d9f4501525971fa9.jpg");
	urls.emplace_back("E:/Media/images/1261981340895sbzeca5ic3.jpg");
	urls.emplace_back("E:/Media/images/233106f4jfwpfsbsajjjrd.jpg");
	urls.emplace_back("E:/Media/images/d56a29001c1ffe50d9f4501525971fa9.jpg");
	urls.emplace_back("E:/Media/images/d56a29001c1ffe50d9f4501525971fa9.jpg");
	for (const std::string& url : urls)
	{
		images.emplace_back(cv::imread(url));
	}

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int rows = 1, cols = 1;
	OpenGLGridCVPainter texture_painter(rows, cols, Width, Height, 0);

	OpenGLShapePainter shape_painter(texture_painter.TextureWidth(), texture_painter.TextureHeight());

	std::vector<glm::vec2> points{
		{ 500.0f, 400.0f}, { 700.0f, 250.0f }, { 900.0f, 400.0f },
		{ 900.0f, 800.0f }, { 750.0f, 900.0f }, { 600.0f, 800.0f }
	};
	std::reverse(points.begin(), points.end());
	shape_painter.Parse(points, {0.0f, 255.0f, 0.0f, 0.1f});

	OpenGLTextPainter text_painter(texture_painter.TextureWidth(), texture_painter.TextureHeight());

	text_painter.Parse(L"09 Hello World", glm::vec2{ 20.0f, 400.0f }, 0.5f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 50.0f, 450.0f }, glm::vec3{ 0, 255, 0 });
	text_painter.Parse(L"My Name", glm::vec2{ 1400.0f, 400.0f }, 1.0f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 1000.0f, 200.0f }, glm::vec3{ 0, 0, 255 });
	text_painter.Parse(L"Up test", glm::vec2{ 1000.0f, 600.0f }, 1.5f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 1010.0f, 400.0f }, glm::vec3{ 255, 0, 0 });
	text_painter.Parse(L"(C) LearnOpenGL.com", glm::vec2{ 20.0f, 700.0f }, 2.0f, glm::vec3{ 0, 0, 255 }, glm::vec2{ 1380.0f, 800.0f }, glm::vec3{ 0, 255, 0 });

	//text_painter.Parse(std::vector<std::wstring>{L"09Hello World", L"(C) LearnOpenGL.com", L"My Name", L"Up test"});
	//text_painter.Parse(std::vector<std::wstring>{L"娃哈", L"中国"});
	// render loop
	// -----------
	int b = 0, g = 125, r = 255;
	double lastTime = glfwGetTime() - 2.0;
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		int width = SCR_WIDTH, height = SCR_HEIGHT;
		glfwGetWindowSize(window, &width, &height);

		double nowTime = glfwGetTime();
		if (lastTime + 1.0 < nowTime)
		{
			lastTime = nowTime;

			texture_painter.BeginUpdate();
			for (int r = 0; r < rows; ++r)
			{
				for (int c = 0; c < cols; ++c)
				{
					texture_painter.Update(r, c, images[r * cols + c]);
				}
			}

			texture_painter.UpdateMargin(cv::Scalar(b, g, r));

			b += 5;
			if (b > 255)
			{
				b = 0;
			}
			g += 5;
			if (g > 255)
			{
				g = 0;
			}
			r += 5;
			if (r > 255)
			{
				r = 0;
			}

			texture_painter.EndUpdate();
		}

		texture_painter.Paint();

		shape_painter.Paint();

		text_painter.Paint();

		//text_painter.Paint(L"09 Hello World", glm::vec2{ 20.0f, 400.0f }, 0.5f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 50.0f, 450.0f }, glm::vec3{ 0, 255, 0 });
		//text_painter.Paint(L"My Name", glm::vec2{ 1400.0f, 400.0f }, 1.0f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 1000.0f, 200.0f }, glm::vec3{ 0, 255, 0 });
		//text_painter.Paint(L"Up test", glm::vec2{ 1000.0f, 600.0f }, 1.5f, glm::vec3{ 255, 0, 0 }, glm::vec2{ 1010.0f, 400.0f }, glm::vec3{ 0, 255, 0 });
		//text_painter.Paint(L"(C) LearnOpenGL.com", glm::vec2{ 20.0f, 700.0f }, 2.0f, glm::vec3{ 0, 0, 255 }, glm::vec2{ 1380.0f, 800.0f }, glm::vec3{ 0, 255, 0 });

		//text_painter.Paint(L"娃哈", 400.0f, 900.0f, 1.0f, 255, 0, 0);
		//text_painter.Paint(L"中国", 400.0f, 1000.0f, 1.0f, 0, 0, 255);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
