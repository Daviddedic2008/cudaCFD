
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
#include <string.h>

#define DEBUG //if defined, extra stuff is printed(shouldn't slow down program noticeably)

// init of structs and methods as well as global vars and respective functions and macros
//*****************************************************************************************************************************************************************************************

// sizes for cfd
#define grid_l 512
#define grid_h 512


// vector for 2d cfd
#pragma pack(push, 4) // seems optimal to me
struct vec2 {
    float x, y;

    __host__ __device__ vec2() : x(0.0f), y(0.0f) {}

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
#pragma pop

// global pointers for storing vecs
//__device__ char vectors[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2)];
//__device__ char vectorBuffer[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2)]; // used for divergence change and advection storage
__device__ char* vectors;
__device__ char* vectorBuffer;

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

#define horizontalVectorsBuffer ((vec2*)vectorBuffer)
#define verticalVectorsBuffer ((vec2*)(vectorBuffer + numHorizontal * sizeof(vec2)))

#define horizontalVectorsCPU ((vec2*)cpuVecs)
#define verticalVectorsCPU ((vec2*)(cpuVecs+ numHorizontal * sizeof(vec2)))

#define rightVecIndex(cellX, cellY) (cellX + 1 + cellY * (grid_l + 1)) 
#define leftVecIndex(cellX, cellY) (cellX + cellY * (grid_l + 1))
#define upVecIndex(cellX, cellY) (cellX + cellY * grid_l)
#define downVecIndex(cellX, cellY) (cellX + (cellY+1) * grid_l)

#define verticalVecIndex(x, y) (x + y * (grid_l))
#define horizontalVecIndex(x, y) (x + y * (grid_l+1))

#define inVerticalBounds(x, y) (x >= 0 && x < grid_l && y >= 0 && y <= grid_h)
#define inHorizontalBounds(x, y) (x >= 0 && x <= grid_l && y >= 0 && y < grid_h)

#define inCellBounds(x, y) (x >= 0 && x < grid_l && y >= 0 && y < grid_h)

#define cellXFromPos(p) (int)p.x
#define cellYFromPos(p) (int)p.y

// init grid
inline __device__ void init_vec() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    // set all vals to 0
    horizontalVectors[rightVecIndex(x, y)] = vec2(1.0f, 1.0f);
    verticalVectors[upVecIndex(x, y)] = vec2(1.0f, 1.0f);
    //verticalVectors[downVecIndex(x, y)] = vec2();
    //horizontalVectors[leftVecIndex(x, y)] = vec2();
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

__global__ void setBarrier(const int x, const int y) {
    set_barrier(x, y);
}

void setBar(const int x, const int y) {
    setBarrier << <1, 1 >> > (x, y);
}

// reset kernels
__global__ void resetVectors() {
    init_vec();
}

__global__ void resetBarriers() {
    init_barrier();
}

void resetVecs() {
    resetVectors << <512, grid_l * grid_h / 512 >> > ();
}

void resetBars() {
    resetBarriers << <512, grid_l * grid_h / 512 >> > ();
}

// copy kernels
__global__ void swapBuffer() {
    char* tmp = vectors;
    vectors = vectorBuffer;
    vectorBuffer = tmp;
}

// divergence functions and kernel
//*****************************************************************************************************************************************************************************************

// calcs divergence for a single cell
inline __device__ float calc_divergence(const int x, const int y) {
    return verticalVectors[upVecIndex(x, y)].y - verticalVectors[downVecIndex(x, y)].y + horizontalVectors[rightVecIndex(x, y)].x - horizontalVectors[leftVecIndex(x, y)].x;
}

#define L 0
#define R 1
#define T 2
#define B 3
inline __device__ void apply_divergence(const int x, const int y) {
    float divergence = calc_divergence(x, y);
    unsigned char num_affected = 0;
    bool affected_cells[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        affected_cells[i] = false;
    }

    // get rights
    #pragma unroll
    for (int xo = -1; xo < 1; xo += 2) {
        const bool tmp = barrier[x + xo + y * grid_l];
        num_affected += tmp;
        affected_cells[(xo + 1) / 2] = tmp;
    }

    // get bottoms
    #pragma unroll
    for (int yo = -1; yo < 1; yo += 2) {
        const bool tmp = barrier[x + (y + yo) * grid_l];
        num_affected += tmp;
        affected_cells[(yo + 5) / 2] = tmp;
    }

    divergence /= num_affected;

    // subtract the divergence equally from each affected vector(not blocked by a barrier)
    verticalVectors[upVecIndex(x, y)].y -= divergence * affected_cells[T]; // up
    verticalVectors[downVecIndex(x, y)].y += divergence * affected_cells[B]; // down
    horizontalVectors[rightVecIndex(x, y)].y -= divergence * affected_cells[R]; // right
    horizontalVectors[leftVecIndex(x, y)].y -= divergence * affected_cells[L]; // left
}

// divergence equations are solved(variables eliminated) using gaussian elimination, and each iteration is done in 2 passes
// each pass is either "white" or "black", and these colors represent the squares on a checkerboard

// threads per block divergence
#define threads_divergence 512
#define blocks_divergence (grid_l * grid_h / 2) / threads_divergence + 1

// divergence kernel "white"
__global__ void divergenceGaussianW() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellId = id * 2; // account for checkerboard spacing
    apply_divergence(cellId % grid_l, cellId / grid_l); // may remove this func due to overhead
}

// divergence kernel "black"
__global__ void divergenceGaussianB() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellId = id * 2 + 1; // account for checkerboard spacing
    if (cellId >= grid_l * grid_h) { return; }
    apply_divergence(cellId % grid_l, cellId / grid_l); // may remove this func due to overhead
}

// cpu function to call kernels
void gaussianDivergenceSolver(const int passes) {
    for (int p = 0; p < passes; p++) {
        divergenceGaussianW << <threads_divergence, blocks_divergence >> > ();
        divergenceGaussianB << <threads_divergence, blocks_divergence >> > ();
    }
}

// advection functions and kernel
//*****************************************************************************************************************************************************************************************

inline __device__ vec2 sample_surrounding_vecs_H(const int x, const int y) {
    // gets the avg vertical component around a horizontal vector at (x, y). returns both the horizontal vector and the vertical component in a vec2
    const unsigned char num_verts = inVerticalBounds(x, y) + inVerticalBounds(x + 1, y) + inVerticalBounds(x, y + 1) + inVerticalBounds(x + 1, y + 1);// num of vertical vecs sampled

    const float vert_comp = (verticalVectors[verticalVecIndex(x, y)].y * inVerticalBounds(x, y) + verticalVectors[verticalVecIndex(x+1, y)].y * inVerticalBounds(x+1, y) + verticalVectors[verticalVecIndex(x, y+1)].y * inVerticalBounds(x, y+1) + verticalVectors[verticalVecIndex(x+1, y+1)].y * inVerticalBounds(x+1, y+1)) / num_verts;// bounds calcs redone to avoid branching(may be slower on some systems)
    
    return vec2(horizontalVectors[horizontalVecIndex(x, y)].x, vert_comp);
}

inline __device__ vec2 sample_surrounding_vecs_V(const int x, const int y) {
    // gets the avg horizontal component around a vertical vector at (x, y). returns both the horizontal vector and the vertical component in a vec2
    const unsigned char num_hors = inHorizontalBounds(x, y) + inHorizontalBounds(x + 1, y) + inHorizontalBounds(x, y - 1) + inHorizontalBounds(x + 1, y - 1);// num of vertical vecs sampled

    const float hor_comp = (horizontalVectors[horizontalVecIndex(x, y)].y * inHorizontalBounds(x, y) + horizontalVectors[horizontalVecIndex(x + 1, y)].y * inHorizontalBounds(x + 1, y) + horizontalVectors[horizontalVecIndex(x, y - 1)].y * inHorizontalBounds(x, y - 1) + horizontalVectors[horizontalVecIndex(x + 1, y - 1)].y * inHorizontalBounds(x + 1, y - 1)) / num_hors;// bounds calcs redone to avoid branching(may be slower on some systems)

    return vec2(hor_comp, verticalVectors[verticalVecIndex(x, y)].y);
}

inline __device__ vec2 get_previous_value_V(const int x, const int y) {
    // get the velocity value the vector comes from and advect it
    // semi lagrangian advection
    vec2 ret = verticalVectors[verticalVecIndex(x, y)];
    const vec2 previous_path = sample_surrounding_vecs_V(x, y) * -1;
    const vec2 pos = vec2(x + 0.5f, y) + previous_path;

    const int px = cellXFromPos(pos);
    const int py = cellYFromPos(pos);
    if (!inCellBounds(px, py)) { return ret; }

    ret.y = 0.0f; // gonna add to it later

    // right bottom cell
    const int cellxy_offset[] = {px+1, px-1, py+1, py-1};
    unsigned char num_sampled = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned char xo = (i < 2) * cellxy_offset[i]; // avoid branching and only have one loop
        const unsigned char yo = (i >= 2) * cellxy_offset[i];
        if (!inCellBounds(px + xo, py + yo)) { continue; } // branching unavoidable

        num_sampled += 2;
        ret.y += verticalVectors[upVecIndex(px + xo, py + yo)].y + verticalVectors[downVecIndex(px + xo, py + yo)].y;
    }
    ret.y /= num_sampled;
    return ret;
}

inline __device__ vec2 get_previous_value_H(const int x, const int y) {
    // get the velocity value the vector comes from and advect it
    // semi lagrangian advection
    vec2 ret = horizontalVectors[horizontalVecIndex(x, y)];
    const vec2 previous_path = sample_surrounding_vecs_H(x, y) * -1;
    const vec2 pos = vec2(x, y + 0.5f) + previous_path;

    const int px = cellXFromPos(pos);
    const int py = cellYFromPos(pos);
    if (!inCellBounds(px, py)) { return ret; }

    ret.x = 0.0f; // gonna add to it later

    // right bottom cell
    const int cellxy_offset[] = { px + 1, px - 1, py + 1, py - 1 };
    unsigned char num_sampled = 0;

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned char xo = (i < 2) * cellxy_offset[i]; // avoid branching and only have one loop
        const unsigned char yo = (i >= 2) * cellxy_offset[i];
        if (!inCellBounds(px + xo, py + yo)) { continue; } // branching unavoidable

        num_sampled += 2;
        ret.x += horizontalVectors[rightVecIndex(px + xo, py + yo)].y + horizontalVectors[leftVecIndex(px + xo, py + yo)].y;
    }
    ret.x /= num_sampled;
    return ret;
}

__global__ void advectionKernel() {
    // advects 2-4 vectors instead of 1 vector to distribute workload amongst more threads
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    const int cx = id % grid_l;
    const int cy = id / grid_l;
    
    // buffer must be used to prevent reads and writes at the same time, and to keep results accurate
    horizontalVectorsBuffer[leftVecIndex(cx, cy)] = get_previous_value_H(cx, cy);
    verticalVectorsBuffer[upVecIndex(cx, cy)] = get_previous_value_V(cx, cy);

    // unavoidable branching(i think)
    if (cx == grid_l - 1) {
        horizontalVectorsBuffer[rightVecIndex(cx, cy)] = get_previous_value_H(cx+1, cy);
    }
    if (cy == grid_h - 1) {
        verticalVectorsBuffer[downVecIndex(cx, cy)] = get_previous_value_V(cx, cy+1);
    }
}

#define threads_advection 512
#define blocks_advection grid_l * grid_h / threads_advection
void semiLagrangianAdvection() {
    advectionKernel << <threads_advection, blocks_advection >> > ();
    swapBuffer << <1, 1 >> > ();
}

// alloc and mem moving functions
//*****************************************************************************************************************************************************************************************

char cpuVecs[(grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2)];
bool cpuBarrier[grid_l * grid_h];

char* deviceVecPointer;
char* deviceVecBufferPointer;

void allocDeviceVars() {
    // tmp cpu pointer used
    #ifdef DEBUG
    cudaError_t m1, m2, c1, c2;
    m1 = cudaMalloc((void**)(&deviceVecPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    c1 = cudaMemcpyToSymbol(vectors, &deviceVecPointer, sizeof(char*));
    
    m2 = cudaMalloc((void**)(&deviceVecBufferPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    c2 = cudaMemcpyToSymbol(vectorBuffer, &deviceVecBufferPointer, sizeof(char*));

    printf("alloc one     malloc: %s | copy: %s\n", cudaGetErrorString(m1), cudaGetErrorString(c1));
    printf("alloc two     malloc: %s | copy: %s\n", cudaGetErrorString(m2), cudaGetErrorString(c2));
    #endif

    #ifndef DEBUG
    cudaMalloc((void**)(&temp), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    cudaMemcpyToSymbol(vectors, &temp, sizeof(char*));

    cudaMalloc((void**)(&temp), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    cudaMemcpyToSymbol(vectorBuffer, &temp, sizeof(char*));
    #endif
}

void moveMainArrayToCPU() {
    cudaError_t e = cudaMemcpy(cpuVecs, deviceVecPointer, (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    printf("copy vecs: %s\n", cudaGetErrorString(e));
#endif
}

void moveBarrierToCPU() {
    cudaError_t e = cudaMemcpyFromSymbol(cpuBarrier, barrier, grid_l * grid_h * sizeof(bool));
#ifdef DEBUG
    printf("copy barrier: %s\n", cudaGetErrorString(e));
#endif
}

// sampling vector field for drawing
//*****************************************************************************************************************************************************************************************

struct color {
    unsigned char r, g, b;

    __host__ __device__ color() : r(0), g(0), b(0){}
    __host__ __device__ color(float red, float green, float blue) : r(red), g(green), b(blue) {}
};

color sampleFieldVelocityMagnitude(const int x, const int y, float threshold) {
    const float total = fabs(horizontalVectorsCPU[rightVecIndex(x, y)].x - horizontalVectorsCPU[leftVecIndex(x, y)].x + verticalVectorsCPU[upVecIndex(x, y)].y - verticalVectorsCPU[downVecIndex(x, y)].y);
    float magnitude = total / threshold;
    magnitude = (magnitude > 1.0f) ? 1.0f : magnitude;
    return color(magnitude * 255, 0, 0);
}

color sampleFieldVelocityDirectionalMagnitude(const int x, const int y, float threshold) {
    const float totalPos = fabs(horizontalVectorsCPU[rightVecIndex(x, y)].x + verticalVectorsCPU[upVecIndex(x, y)].y);
    const float totalNeg = fabs(horizontalVectorsCPU[leftVecIndex(x, y)].x + verticalVectorsCPU[downVecIndex(x, y)].y);

    float magnitudePos = totalPos / threshold;
    float magnitudeNeg = totalNeg / threshold;

    magnitudePos = (magnitudePos > 1.0f) ? 1.0f : magnitudePos;
    magnitudeNeg = (magnitudeNeg > 1.0f) ? 1.0f : magnitudeNeg;

    return color(magnitudePos * 255, magnitudeNeg * 255, 0);
}

unsigned char cpuColors[grid_l * grid_h];

void fillColorArray(float threshold, char* sampleType) {
    if (strcmp(sampleType, "magnitude") == 0) {
        for (int x = 0; x < grid_l; x++) {
            for (int y = 0; y < grid_h; y++) {
                ((color*)cpuColors)[x + y * grid_l] = sampleFieldVelocityMagnitude(x, y, threshold);
                if (cpuBarrier[x + y * grid_l]) {
                    ((color*)cpuColors)[x + y * grid_l] = color(255, 255, 255);
                }
            }
        }
    }

    if (strcmp(sampleType, "directional magnitude") == 0) {
        for (int x = 0; x < grid_l; x++) {
            for (int y = 0; y < grid_h; y++) {
                ((color*)cpuColors)[x + y * grid_l] = sampleFieldVelocityDirectionalMagnitude(x, y, threshold);
                if (cpuBarrier[x + y * grid_l]) {
                    ((color*)cpuColors)[x + y * grid_l] = color(255, 255, 255);
                }
            }
        }
    }
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
    // alloc and set all global device arrays/pointers to 0
    allocDeviceVars();
    resetVecs();
    resetBars();

    // for now, make a square barrier
    for (int x = 100; x < 200; x++) {
        for (int y = 200; y < 300; y++) {
            setBar(x, y);
        }
    }

    moveMainArrayToCPU();
    moveBarrierToCPU();

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
    //uint8_t pixels[grid_h * grid_l * 3];
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_l, grid_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cpuColors);
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
        fillColorArray(3.0f, "directional magnitude");
        // **
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_l, grid_h, 0, GL_RGB, GL_UNSIGNED_BYTE, cpuColors);
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