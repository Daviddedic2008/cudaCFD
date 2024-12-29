
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <time.h>
#include <glad/glad.h>
#include <glfw3.h>

// init of structs and methods as well as global vars and respective functions and macros
//*****************************************************************************************************************************************************************************************

// sizes for cfd
#define grid_l 512
#define grid_h 512


// vector for 2d cfd
struct vec2 {
    float x, y;

    __host__ __device__ vec2() : x(0), y(0) {}

    __host__ __device__ vec2(float X, float Y) : x(X), y(Y) {}

    inline __host__ __device__ vec2 operator+(const vec2& f) const {
        return vec2(x + f.x, y + f.y);
    }

    inline __host__ __device__ vec2 operator-(const vec2& f) const {
        return vec2(x - f.x, y - f.y);
    }

    inline __host__ __device__ vec2 operator*(const float scalar) const {
        return vec2(x * scalar, y * scalar);
    }
};

// global array for storing vecs
__device__ char vectors[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2)];
__device__ char vectorBuffer[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2)]; // used for divergence change and advection storage

__device__ bool barrier[grid_l * grid_h];

// macros for accessing array
// * "top" of a cell is lower in index than "bottom"
// * "top" of a cell is positive(if positive, then contribution to divergence is positive)
// * "right" of cell is higher in index than "left"
// * "right" of a cell is positive(if positive, then contribution to divergence is positive)
// * "left" and "down" negatively contribute to divergence
#define numHorizontal ((grid_l+1) * grid_h)
#define numVertical ((grid_h+1)*grid_l)

#define horizontalVectors ((vec2*)vectors)
#define verticalVectors ((vec2*)(vectors + numHorizontal * sizeof(vec2)))

#define rightVecIndex(cellX, cellY) (cellX + 1 + cellY * (grid_l + 1)) 
#define leftVecIndex(cellX, cellY) (cellX + cellY * (grid_l + 1))
#define upVecIndex(cellX, cellY) (cellX + cellY * grid_l)
#define downVecIndex(cellX, cellY) (cellX + (cellY+1) * grid_l)

// init grid
inline __device__ void init_vec() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_h;

    // set all vals to 0
    horizontalVectors[rightVecIndex(x, y)] = vec2();
    horizontalVectors[leftVecIndex(x, y)] = vec2();
    verticalVectors[upVecIndex(x, y)] = vec2();
    verticalVectors[downVecIndex(x, y)] = vec2();
}

// sets both left and right vecs of cell to v
inline __device__ void set_horizontal_vec_cell(const vec2 v, const int x, const int y) {
    horizontalVectors[rightVecIndex(x, y)] = v;
    horizontalVectors[leftVecIndex(x, y)] = v;
}

// sets both up and down vecs of cell to v
inline __device__ void set_vertical_vec_cell(const vec2 v, const int x, const int y) {
    verticalVectors[upVecIndex(x, y)] = v;
    verticalVectors[downVecIndex(x, y)] = v;
}

// init barrier
inline __device__ void init_barrier() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    barrier[id] = false;
}

// set val for a single barrier
inline __device__ void set_barrier(const int x, const int y) {
    barrier[x + y * grid_l] = true;
}

// divergence functions and kernel
//*****************************************************************************************************************************************************************************************

// calcs divergence for a single cell
inline __device__ float calc_divergence(const int x, const int y) {
    return verticalVectors[upVecIndex(x, y)].y - verticalVectors[downVecIndex(x, y)].y + horizontalVectors[rightVecIndex(x, y)].x - horizontalVectors[leftVecIndex(x, y)].x;
}

inline __device__ void apply_divergence(const int x, const int y) {
    float divergence = calc_divergence(x, y);
    unsigned char num_affected = 0;
    bool affected_cells[4];
    for (int i = 0; i < 4; i++) {
        affected_cells[i] = false;
    }

    // get rights
    #pragma unroll
    for (int xo = -1; xo < 1; xo += 2) {
        bool tmp = barrier[x + xo + y * grid_l];
        num_affected += tmp;
        affected_cells[(xo + 1) / 2] = tmp;
    }

    // get bottoms
    #pragma unroll
    for (int yo = -1; yo < 1; yo += 2) {
        num_affected += barrier[x + (y+yo) * grid_l];
    }

    divergence /= num_affected;

}

//*****************************************************************************************************************************************************************************************
// opengl stuff
// draws 2 triangles at z=0 and textures them with the pixel colors outputted by the cuda program
// no interop, data transfers from GPU to CPU and back to GPU each frame
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
unsigned int SCR_WIDTH = grid_l;
unsigned int SCR_HEIGHT = grid_h;


char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aColor;\n"
"layout(location = 2) in vec2 aTexCoord;\n"
"out vec3 ourColor;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   ourColor = aColor;\n"
"   TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
"}\0";

char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D texture1;\n"
"void main()\n"
"{\n"
"   FragColor = texture(texture1, TexCoord);\n"
"}\n\0";

float truncate(float f) {
    return fabs(1.0 / (1.0 + exp(-1.0 * f))-0.5) * 2.0f;
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Cuda-openGL Interop", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        return -1;
    }
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::FRAGMENT::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED %s\n", infoLog);
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

    unsigned int texture1;
    uint8_t pixels[grid_h * grid_l * 3];
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_l, grid_h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);
    glUseProgram(shaderProgram); 
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        int ind = 0;
        // **
        // dodaj boje tu u pixels
        // **
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_l, grid_h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		glGenerateMipmap(GL_TEXTURE_2D);
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);

        // render container
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}