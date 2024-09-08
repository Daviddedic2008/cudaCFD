
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

#define grid_l 512
#define grid_h 512
#define xVal val
#define yVal val
#define totalNumVecs grid_l * (grid_h + 1) + (grid_l + 1) * grid_h
#define totalNumCells grid_l * grid_h
#define getCellIndex(xargs, yargs) xargs + yargs * grid_l
#define getVerticalVecIndex(xargs, yargs) getHorizontalVecIndex(xargs, yargs) - grid_l
#define getHorizontalVecIndex(xargs, yargs) xargs + grid_l * (yargs+1) + (grid_l+1) * yargs
#define rightVecIndexOfCell(xargs, yargs) getHorizontalVecIndex((xargs+1), yargs)
#define leftVecIndexOfCell(xargs, yargs) getHorizontalVecIndex(xargs, yargs)
#define upVecIndexOfCell(xargs, yargs) getVerticalVecIndex(xargs, (yargs+1))
#define downVecIndexOfCell(xargs, yargs) getVerticalVecIndex(xargs, yargs)
#define rightValCell(xargs, yargs) vectorGrid[rightVecIndexOfCell(xargs, yargs)].val
#define leftValCell(xargs, yargs) vectorGrid[leftVecIndexOfCell(xargs, yargs)].val
#define upValCell(xargs, yargs) vectorGrid[upVecIndexOfCell(xargs, yargs)].val
#define downValCell(xargs, yargs) vectorGrid[downVecIndexOfCell(xargs, yargs)].val
#define inHorBounds(xargs, yargs) (xargs <= grid_l && xargs >= 0 && yargs < grid_h && yargs >= 0)
#define inVertBounds(xargs, yargs) (xargs < grid_l && xargs >= 0 && yargs <= grid_h && yargs >= 0)
#define inCellBounds(xargs, yargs) (xargs < grid_l && xargs >= 0 && yargs < grid_h && yargs >= 0)
#define inObstacleBounds(xargs, yargs) !cellGrid[getCellIndex(xargs, yargs)].obstacle
#define threads_divergence 512
#define blocks_divergence totalNumCells/threads_divergence


#define signf(f) f > 0.0f ? 1.0f : -1.0f

typedef struct{
    float val;
}vec;

typedef struct {
    vec upChange;
    vec downChange;
    vec rightChange;
    vec leftChange;
}cellChange;

typedef struct {
    unsigned char obstacle;
}cell;

__device__ vec vectorGrid[totalNumVecs];
vec vectorGridCPU[totalNumVecs];
__device__ cellChange vectorGridChanges[totalNumCells];
cellChange vectorGridChangesCPU[totalNumCells];
__device__ cell cellGrid[totalNumCells];
cell cellGridCPU[totalNumCells];
__device__ vec vectorGridBuffer[totalNumVecs];


// 0 : R 
// 1 : L 
// 2 : U 
// 3 : D

// 0 : R 
// 1 : L 
// 2 : U 
// 3 : D

__global__ void addAdvection() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int x = id % grid_l;
    int y = id / grid_l;
    if (x == 0) { vectorGrid[leftVecIndexOfCell(x, y)] = vectorGridBuffer[leftVecIndexOfCell(x, y)]; }
    if (y == 0) { vectorGrid[downVecIndexOfCell(x, y)] = vectorGridBuffer[downVecIndexOfCell(x, y)]; }
    vectorGrid[rightVecIndexOfCell(x, y)] = vectorGridBuffer[rightVecIndexOfCell(x, y)];
    vectorGrid[upVecIndexOfCell(x, y)] = vectorGridBuffer[upVecIndexOfCell(x, y)];
}

void clearVecBuffer() {
    cudaMemset(vectorGridBuffer, sizeof(vectorGridBuffer), 0);
}

__global__ void divergence_kernel(unsigned char r) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if ((id % 2 == 0) ^ r) { return; }
    if (cellGrid[id].obstacle) { return; }
    if (id >= totalNumCells) { return; }
    int x = id % grid_l;
    int y = id / grid_l;
    float divergence = upValCell(x, y) - upValCell(x, y-1) + rightValCell(x, y) - leftValCell(x, y);
    unsigned char stat[4];
    stat[0] = x < grid_l-1;
    if (x+1 < grid_l) {
        stat[0] = cellGrid[id + 1].obstacle ? 0 : stat[0];
    }
    stat[1] = x > 0;
    if (x-1 >= 0) {
        stat[1] = cellGrid[id - 1].obstacle ? 0 : stat[1];
    }
    stat[2] = y < 320;
    if (y+1 < grid_h) {
        stat[2] = cellGrid[id + grid_l].obstacle ? 0 : stat[2];
    }
    stat[3] = y > 0;
    if (y-1 >= 0) {
        stat[3] = cellGrid[id - grid_l].obstacle ? 0 : stat[3];
    }

    divergence = divergence / (float)(stat[0] + stat[1] + stat[2] + stat[3]);
    vectorGridChanges[id].rightChange.val = stat[0] ? -1.0f * divergence : 0.0f;
    vectorGridChanges[id].leftChange.val = stat[1] ? divergence : 0.0f;
    vectorGridChanges[id].upChange.val = stat[2] ? -1.0f * divergence : 0.0f;
    vectorGridChanges[id].downChange.val = stat[3] ? divergence : 0.0f;
}

__global__ void zeroVecChanges() {
    int id = threadIdx.x + blockIdx.x * grid_l;
    vectorGridChanges[id].rightChange.val = 0.0f;
    vectorGridChanges[id].leftChange.val = 0.0f;
    vectorGridChanges[id].upChange.val = 0.0f;
    vectorGridChanges[id].downChange.val = 0.0f;
}

__global__ void addChangesUpRight() {
    int id = threadIdx.x + blockIdx.x * grid_l;
    int x = id % grid_l;
    int y = id / grid_l;
    vectorGrid[rightVecIndexOfCell(x, y)].val += vectorGridChanges[id].rightChange.val;
    vectorGrid[upVecIndexOfCell(x, y)].val += vectorGridChanges[id].upChange.val;
}

__global__ void addChangesDownLeft() {
    int id = threadIdx.x + blockIdx.x * grid_l;
    int x = id % grid_l;
    int y = id / grid_l;
    vectorGrid[leftVecIndexOfCell(x, y)].val += vectorGridChanges[id].leftChange.val;
    vectorGrid[downVecIndexOfCell(x, y)].val += vectorGridChanges[id].downChange.val;
}

void addChanges() {
    addChangesDownLeft << <grid_l, grid_h >> > ();
    
    addChangesUpRight << <grid_l, grid_h >> > ();
    /*cudaMemcpyFromSymbol(vectorGridCPU, vectorGrid, sizeof(vec) * totalNumVecs);
    cudaMemcpyFromSymbol(vectorGridChangesCPU, vectorGridChanges, sizeof(cellChange) * totalNumCells);
    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_l; x++) {
            vectorGridCPU[rightVecIndexOfCell(x, y)].val += vectorGridChangesCPU[getCellIndex(x, y)].rightChange.val;
            vectorGridCPU[leftVecIndexOfCell(x, y)].val += vectorGridChangesCPU[getCellIndex(x, y)].leftChange.val;
            vectorGridCPU[upVecIndexOfCell(x, y)].val += vectorGridChangesCPU[getCellIndex(x, y)].upChange.val;
            vectorGridCPU[downVecIndexOfCell(x, y)].val += vectorGridChangesCPU[getCellIndex(x, y)].downChange.val;
        }
    }
    cudaMemcpyToSymbol(vectorGrid, vectorGridCPU, sizeof(vec) * totalNumVecs);*/
}

void printChange(cellChange c) {
    printf("right %f, left %f, up %f, down %f\n", c.rightChange.val, c.leftChange.val, c.upChange.val, c.downChange.val);
}

void solveDivergence(int num_reps) {
    zeroVecChanges << <grid_l, grid_h >> > (); // provereno
    divergence_kernel << <grid_l, grid_h >> > (0);
    addChanges();
    zeroVecChanges << <grid_l, grid_h >> > ();
    divergence_kernel << <grid_l, grid_h >> > (1);
    //printf("kernel err: %s\n", cudaGetErrorString(err));
    //cudaMemcpyFromSymbol(vectorGridChangesCPU, vectorGridChanges, sizeof(cellChange) * totalNumCells);
    addChanges();
}

__global__ void advectionKernelHorizontal() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int x = id % (grid_l + 1);
    int y = id / (grid_l + 1);
    if (inCellBounds(x, y)) {
        if (cellGrid[getCellIndex(x, y)].obstacle) { goto skpRight; }
    }
    if (inCellBounds(x-1, y)) {
        if (cellGrid[getCellIndex(x - 1, y)].obstacle) { goto skpRight; }
    }
    if (x == 0 || x == grid_l) { goto skpRight; }
    float yv = 0;
    int num;
    for (int yo = 0; yo <= 1; yo++) {
        for (int xo = -1; xo <= 0; xo++) {
            if (inVertBounds(x + xo, y + yo)) {
                num++; yv += vectorGrid[getVerticalVecIndex(x + xo, y + yo)].val; 
            }
        }   
    }
    yv /= num * -1;
    float xv = -1 * vectorGrid[getHorizontalVecIndex(x, y)].val;
    int xChange = ((int)xv) - ((xv < 0.0f) ? 1 : 0);
    int yChange = (int)(yv + (yv > 0.0f ? 0.5f : -0.5f));
    if (!inCellBounds(x + xChange, y + yChange)) { goto skpRight; }
    else { if (!inObstacleBounds(x + xChange, y + yChange)) { goto skpRight; } }
    float leftPercent = (xv > 0.0f ? fabs((fmodf(xv, 1.0f))) : (1.0f - fabs((fmodf(xv, 1.0f)))));
    float rightPercent = 1.0f - leftPercent;
    vectorGridBuffer[getHorizontalVecIndex(x, y)].val = rightValCell(x + xChange, y + yChange) * rightPercent + leftValCell(x + xChange, y + yChange) * leftPercent;
    return;
skpRight:;
    vectorGridBuffer[getHorizontalVecIndex(x, y)] = vectorGrid[getHorizontalVecIndex(x, y)];
}

__global__ void advectionKernelVertical() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int x = id % grid_l;
    int y = id / grid_l;
    if (cellGrid[getCellIndex(x, y)].obstacle) { goto skpUp; }
    if (inCellBounds(x-1, y)) { if (cellGrid[getCellIndex(x-1, y)].obstacle) { goto skpUp; } }
    float xv = 0.0f;
    int num = 0;
    for (int xo = 0; xo <= 1; xo++) {
        for (int yo = -1; yo <= 0; yo++) {
            if (inHorBounds(x + xo, y + yo)) {
                num++; xv += vectorGrid[getHorizontalVecIndex(x + xo, y + yo)].val;
            }
        }
    }
    xv /= -1 * num;
    float yv = vectorGrid[getVerticalVecIndex(x, y)].val * -1;
    int xChange, yChange;
    xChange = (int)(xv + signf(xv) * 0.5f);
    yChange = (int)yv - ((yv < 0.0f) ? 1 : 0);
    if (!inCellBounds(x + xChange, y + yChange)) { goto skpUp; }
    if(!inObstacleBounds(x + xChange, y + yChange)) { goto skpUp; }
    float upPercent = (yv < 0.0f) ? fabs(fmodf(yv, 1.0f)) : 1.0f - fabs(fmodf(yv, 1.0f));
    float downPercent = 1.0f - fabs(upPercent);
    vectorGridBuffer[getVerticalVecIndex(x, y)].val = upValCell(x + xChange, y + yChange) * upPercent + upValCell(x + xChange, y + yChange-1) * downPercent;
    //if (yChange != 0) { printf("%d\n", yChange); }
skpUp:;
}

void solveAdvection() {
    clearVecBuffer();
    advectionKernelHorizontal << <grid_l, grid_h + 1 >> > ();
    //advectionKernelVertical << <grid_l, grid_h + 1 >> > ();
    addAdvection << <grid_l, grid_h >> > ();
}

void navierStokes(int divergence_reps) {
    for (int i = 0; i < 10; i++) {
        solveDivergence(divergence_reps);
    }
    solveAdvection();
    cudaMemcpyFromSymbol(vectorGridCPU, vectorGrid, sizeof(vec) * totalNumVecs);
}

__global__ void setup(unsigned char x, unsigned char barrier) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int v = 0; v < x ? grid_h : grid_h; v++) {
        int v1 = x ? id : v;
        int v2 = x ? v : id;
        vectorGrid[getHorizontalVecIndex(v1, v2)].val = 1.0f;
    }
}

int dist(int x1, int y1, int x2, int y2) {
    return sqrt((float)(x1 - x2) * (float)(x1 - x2) + (float)(y1 - y2) * (float)(y1 - y2));
}

void setupCPU() {
    memset(vectorGridCPU, 0, sizeof(vectorGridCPU));
    for (int y = 0; y < grid_h; y++) {
        for (int x = grid_l-1; x > grid_l-2; x--) {
            vectorGridCPU[rightVecIndexOfCell(x, y)].val = -0.9f;
            vectorGridCPU[leftVecIndexOfCell(x, y)].val = -0.9f;
        }
        for (int x = 0; x < 1; x++) {
            vectorGridCPU[rightVecIndexOfCell(x, y)].val = -0.9f;
            vectorGridCPU[leftVecIndexOfCell(x, y)].val = -0.9f;
        }
    }
    memset(cellGridCPU, 0, sizeof(cellGridCPU));
    for (int y = 0; y < 320; y++) {
        for (int x = 0; x < grid_l; x++) {
            if (dist(x, y, 200, 200) < 30) {
                cellGridCPU[getCellIndex(x, y)].obstacle = 1;
            }
        }
    }
    cudaMemcpyToSymbol(vectorGrid, vectorGridCPU, sizeof(vec) * totalNumVecs);
    cudaMemcpyToSymbol(cellGrid, cellGridCPU, sizeof(cell) * totalNumCells);
}

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
    setupCPU();
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
    for (int jebi = 0; jebi < 2; jebi++) {
        //navierStokes(1);
    }

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
		// update pixel array here
        navierStokes(1);
        int ind = 0;
        for (int y = 0; y < grid_h; y++) {
            for (int x = 0; x < grid_l; x++) {
                float v1 = truncate(vectorGridCPU[upVecIndexOfCell(x, y)].val + vectorGridCPU[rightVecIndexOfCell(x, y)].val * 2.0f);
                float v2 = truncate(vectorGridCPU[downVecIndexOfCell(x, y)].val + vectorGridCPU[leftVecIndexOfCell(x, y)].val * 2.0f);
                pixels[ind] = v2 * 255;
                pixels[ind + 1] = v1 * 255;
                pixels[ind + 2] = 0;
                if (cellGridCPU[x + y * grid_l].obstacle) {
                    pixels[ind] = 255;
                    pixels[ind + 1] = 255;
                    pixels[ind + 2] = 255;
                }
                ind += 3;
            }
        }
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