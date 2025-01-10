
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

cudaError_t ercall;
#define CCALL(call) ercall = call; if(cudaSuccess != ercall){fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(ercall)); exit(EXIT_FAILURE);}

//#define DEBUG //if defined, extra stuff is printed(shouldn't slow down program noticeably)

// init of structs and methods as well as global vars and respective functions and macros
//*****************************************************************************************************************************************************************************************

// sizes for cfd
#define grid_l 480
#define grid_h 270

// overrelaxation
#define overrelax_const 1.0f


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
__device__ char* vectors; // main array
__device__ char* vectorBuffer; // buffer for divergence and advection

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

#define rightVecIndex(cellX, cellY) horizontalVecIndex(cellX+1, cellY) 
#define leftVecIndex(cellX, cellY) horizontalVecIndex(cellX, cellY)
#define upVecIndex(cellX, cellY) verticalVecIndex(cellX, cellY)
#define downVecIndex(cellX, cellY) verticalVecIndex(cellX, (cellY+1))

#define verticalVecIndex(x, y) (x + y * (grid_l))
#define horizontalVecIndex(x, y) (x + y * (grid_l+1))

#define inVerticalBounds(x, y) ((x) >= 0 && (x) < grid_l && (y) >= 0 && (y) <= grid_h)
#define inHorizontalBounds(x, y) ((x) >= 0 && (x) <= grid_l && (y) >= 0 && (y) < grid_h)

#define inCellBounds(x, y) (x >= 0 && x < grid_l && y >= 0 && y < grid_h)

#define cellXFromPos(p) (int)p.x
#define cellYFromPos(p) (int)p.y

// init grid
inline __device__ void init_vec() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    // set all vals to 0
    horizontalVectors[rightVecIndex(x, y)] = vec2();
    verticalVectors[upVecIndex(x, y)] = vec2();
    if (y == grid_h - 1) {
        verticalVectors[downVecIndex(x, y)] = vec2();
    }
    if (x == 0) {
        horizontalVectors[leftVecIndex(x, y)] = vec2();
    }
}

inline __device__ void init_vecBuffer() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    // set all vals to 0
    horizontalVectorsBuffer[rightVecIndex(x, y)] = vec2();
    verticalVectorsBuffer[upVecIndex(x, y)] = vec2();
    if (y == grid_h - 1) {
        verticalVectorsBuffer[downVecIndex(x, y)] = vec2();
    }
    if (x == 0) {
        horizontalVectorsBuffer[leftVecIndex(x, y)] = vec2();
    }
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

__global__ void setHorizontalVec(const vec2 v, const int x, const int y) {
    set_horizontal_vec_cell(v, x, y);
}

void setHorVecs(const vec2 v, const int x, const int y) {
    setHorizontalVec << <1, 1 >> > (v, x, y);
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

__global__ void resetVectorsBuffer() {
    init_vecBuffer();
}

__global__ void resetBarriers() {
    init_barrier();
}

void resetVecs() {
    resetVectors << <512, grid_l * grid_h / 512 >> > ();
}

void resetVecsBuf() {
    resetVectorsBuffer << <512, grid_l* grid_h / 512 >> > ();
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
inline __device__ double calc_divergence(const int x, const int y) {
    return (verticalVectors[upVecIndex(x, y)].y * inCellBounds(x, y) - verticalVectors[downVecIndex(x, y)].y * inCellBounds(x, y) + horizontalVectors[rightVecIndex(x, y)].x * inCellBounds(x, y) - horizontalVectors[leftVecIndex(x, y)].x * inCellBounds(x, y)) * overrelax_const;
}

inline __host__ __device__ long int xorRand(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

#define L 0
#define R 1
#define T 2
#define B 3
inline __device__ void apply_divergence(const int x, const int y) {
    bool affected_cells[4];
    ((char32_t*)affected_cells)[0] = 0;

    unsigned char num_affected = 0;

    if (barrier[x + y * grid_l]) { return; }

    // get rights
    #pragma unroll
    for (int xo = -1; xo <= 1; xo += 2) {
        if (!inCellBounds(x+xo, y) || barrier[x + xo + y * grid_l]) { continue; }
        
        num_affected += 1;
        affected_cells[(xo + 1) / 2] = true;
    }

    // get bottoms
    #pragma unroll
    for (int yo = -1; yo <= 1; yo += 2) {
        if (!inCellBounds(x, y + yo) || barrier[x + (y + yo) * grid_l]) { continue; }
        
        num_affected += 1;
        affected_cells[(yo + 5) / 2] = true;
    }

    // if by some chance this passes:
    if (num_affected == 0) { return; }

    const float divergence = calc_divergence(x, y) / (float)num_affected;

    // subtract the divergence equally from each affected vector(not blocked by a barrier)

    verticalVectorsBuffer[upVecIndex(x, y)].y -= divergence * affected_cells[T]; // up
    verticalVectorsBuffer[downVecIndex(x, y)].y += divergence * affected_cells[B]; // down
    horizontalVectorsBuffer[rightVecIndex(x, y)].x -= divergence * affected_cells[R]; // right
    horizontalVectorsBuffer[leftVecIndex(x, y)].x += divergence * affected_cells[L]; // left
}

// divergence equations are solved(variables eliminated) using gaussian elimination, and each iteration is done in 2 passes
// each pass is either "white" or "black", and these colors represent the squares on a checkerboard

// threads per block divergence
#define threads_divergence 256
#define blocks_divergence (grid_l * grid_h) / threads_divergence/2

// divergence kernel "white"
__global__ void divergenceGaussianW() {
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId*2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 0);

    if (cellId >= grid_l * grid_h) { return; }
    apply_divergence(cellX, cellY); // may remove this func due to overhead
}

// divergence kernel "black"
__global__ void divergenceGaussianB() {
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId*2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 1);

    if (cellId >= grid_l * grid_h) { return; }
    apply_divergence(cellX, cellY); // may remove this func due to overhead
}

// add buffer to main
__global__ void addBufferW() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellId = id;
    if (cellId >= grid_l * grid_h) return; // avoid out-of-bounds access

    // ensure no accessing same mem
    verticalVectors[upVecIndex(cellId % grid_l, cellId / grid_l)].y += verticalVectorsBuffer[upVecIndex(cellId % grid_l, cellId / grid_l)].y;
    horizontalVectors[leftVecIndex(cellId % grid_l, cellId / grid_l)].x += horizontalVectorsBuffer[leftVecIndex(cellId % grid_l, cellId / grid_l)].x;

    if (cellId % grid_l == grid_l - 1) {
        horizontalVectors[rightVecIndex(cellId % grid_l, cellId / grid_l)].x += horizontalVectorsBuffer[rightVecIndex(cellId % grid_l, cellId / grid_l)].x;
    }

    if (cellId / grid_l == grid_h - 1) {
        verticalVectors[downVecIndex(cellId % grid_l, cellId / grid_l)].y += verticalVectorsBuffer[downVecIndex(cellId % grid_l, cellId / grid_l)].y;
    }
}

void addBuf() {
    int totalThreads = grid_l * grid_h;
    int blocks = (totalThreads + 511) / 512;
    addBufferW << <blocks, 512 >> > ();
    CCALL(cudaDeviceSynchronize());
}


// cpu function to call kernels
void gaussianDivergenceSolver(const int passes) {
    cudaDeviceSynchronize();
    for (int p = 0; p < passes; p++) {
        resetVecsBuf();
        divergenceGaussianB << <threads_divergence, blocks_divergence >> > ();
        cudaDeviceSynchronize();
        divergenceGaussianW << <threads_divergence, blocks_divergence>> > ();
        cudaDeviceSynchronize();
        addBuf();


        // call in reverse order
        resetVecsBuf();
        divergenceGaussianW << <threads_divergence, blocks_divergence >> > ();
        cudaDeviceSynchronize();
        divergenceGaussianB << <threads_divergence, blocks_divergence >> > ();
        cudaDeviceSynchronize();
        addBuf();
    }
}

// advection functions and kernel
//*****************************************************************************************************************************************************************************************

#define threads_advection 512
#define blocks_advection grid_l * grid_h / threads_advection / 2

// simple cell-based advection. not as precise as advection for each individual vector, but for a first try this is fine
__global__ void advectionKernelW() {
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 0);

    if (!inCellBounds(cellX, cellY)) { return; }

    const vec2 prev_pos = vec2(cellX+0.5f, cellY+0.5f) - horizontalVectors[rightVecIndex(cellX, cellY)] - horizontalVectors[leftVecIndex(cellX, cellY)] - verticalVectors[upVecIndex(cellX, cellY)] - verticalVectors[downVecIndex(cellX, cellY)];

    const int prevCellX = (int)prev_pos.x;
    const int prevCellY = (int)prev_pos.y;

    if (barrier[cellX + cellY * grid_l] || barrier[prevCellX + prevCellY * grid_l] || !inCellBounds(prevCellX, prevCellY) || !inCellBounds(cellX, cellY)) { return; }

    horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] = horizontalVectors[rightVecIndex(prevCellX, prevCellY)] * (cellX == grid_l - 1 ? 1.0f : 0.5f);
    horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] = horizontalVectors[leftVecIndex(prevCellX, prevCellY)] * (cellX == 0 ? 1.0f : 0.5f);

    verticalVectorsBuffer[upVecIndex(cellX, cellY)] = verticalVectors[upVecIndex(prevCellX, prevCellY)] * (cellY == 0 ? 1.0f : 0.5f);
    verticalVectorsBuffer[downVecIndex(cellX, cellY)] = verticalVectors[downVecIndex(prevCellX, prevCellY)] * (cellY == grid_h - 1 ? 1.0f : 0.5f);
}

__global__ void advectionKernelB() {
    const int cellId = threadIdx.x + blockIdx.x * blockDim.x;
    const int cellY = cellId * 2 / grid_l;
    const int cellX = 2 * cellId % grid_l + 1 * (cellY % 2 == 1);

    if (!inCellBounds(cellX, cellY)) { return; }

    const vec2 prev_pos = vec2(cellX + 0.5f, cellY + 0.5f) - horizontalVectors[rightVecIndex(cellX, cellY)] - horizontalVectors[leftVecIndex(cellX, cellY)] - verticalVectors[upVecIndex(cellX, cellY)] - verticalVectors[downVecIndex(cellX, cellY)];

    const int prevCellX = (int)prev_pos.x;
    const int prevCellY = (int)prev_pos.y;

    if (barrier[cellX + cellY * grid_l] || barrier[prevCellX + prevCellY * grid_l] || !inCellBounds(prevCellX, prevCellY) || !inCellBounds(cellX, cellY)) { return; }

    horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] = horizontalVectorsBuffer[rightVecIndex(cellX, cellY)] + horizontalVectors[rightVecIndex(prevCellX, prevCellY)] * (cellX == grid_l - 1 ? 1.0f : 0.5f);
    horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] = horizontalVectorsBuffer[leftVecIndex(cellX, cellY)] + horizontalVectors[leftVecIndex(prevCellX, prevCellY)] * (cellX == 0 ? 1.0f : 0.5f);

    verticalVectorsBuffer[upVecIndex(cellX, cellY)] = verticalVectorsBuffer[upVecIndex(cellX, cellY)] + verticalVectors[upVecIndex(prevCellX, prevCellY)] * (cellY == 0 ? 1.0f : 0.5f);
    verticalVectorsBuffer[downVecIndex(cellX, cellY)] = verticalVectorsBuffer[downVecIndex(cellX, cellY)] + verticalVectors[downVecIndex(prevCellX, prevCellY)] * (cellY == grid_h - 1 ? 1.0f : 0.5f);
}

__global__ void copyFromBuffer() {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int x = id % grid_l;
    const int y = id / grid_l;

    horizontalVectors[rightVecIndex(x, y)] = horizontalVectors[rightVecIndex(x, y)];
    verticalVectors[downVecIndex(x, y)] = verticalVectorsBuffer[downVecIndex(x, y)];

    if (x == 0) {
        horizontalVectors[leftVecIndex(x, y)] = horizontalVectors[leftVecIndex(x, y)];
    }
    if (y == 0) {
        verticalVectors[upVecIndex(x, y)] = verticalVectorsBuffer[upVecIndex(x, y)];
    }
}

void semiLagrangianAdvection() {
    CCALL(cudaDeviceSynchronize());
    advectionKernelW << <threads_advection, blocks_advection >> > ();
    CCALL(cudaDeviceSynchronize());
    advectionKernelB << <threads_advection, blocks_advection >> > ();
    CCALL(cudaDeviceSynchronize());
    copyFromBuffer << <threads_advection, blocks_advection*2 >> > ();
    CCALL(cudaDeviceSynchronize());
    advectionKernelB << <threads_advection, blocks_advection >> > ();
    CCALL(cudaDeviceSynchronize());
    advectionKernelW << <threads_advection, blocks_advection >> > ();
    CCALL(cudaDeviceSynchronize());
    copyFromBuffer << <threads_advection, blocks_advection * 2 >> > ();
    CCALL(cudaDeviceSynchronize());
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
    cudaMalloc((void**)(&deviceVecPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    cudaMemcpyToSymbol(vectors, &deviceVecPointer, sizeof(char*));

    cudaMalloc((void**)(&deviceVecBufferPointer), (grid_l * (grid_h + 1) + grid_h * (grid_l + 1)) * sizeof(vec2));
    cudaMemcpyToSymbol(vectorBuffer, &deviceVecBufferPointer, sizeof(char*));
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
    const float total = fabs(horizontalVectorsCPU[rightVecIndex(x, y)].x) + fabs(horizontalVectorsCPU[leftVecIndex(x, y)].x) + fabs(verticalVectorsCPU[upVecIndex(x, y)].y) - fabs(verticalVectorsCPU[downVecIndex(x, y)].y);
    float magnitude = total / threshold;
    magnitude = (magnitude > 1.0f) ? 1.0f : magnitude;
    return color(magnitude * 255, 0, 0);
}

color sampleFieldVelocityDirectionalMagnitude(const int x, const int y, float threshold) {
    const float totalPos = fabs(horizontalVectorsCPU[rightVecIndex(x, y)].x) + fabs(verticalVectorsCPU[upVecIndex(x, y)].y);
    const float totalNeg = fabs(horizontalVectorsCPU[leftVecIndex(x, y)].x) + fabs(verticalVectorsCPU[downVecIndex(x, y)].y);

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
            for (int y = grid_h-1; y >= 0; y--) {
                ((color*)cpuColors)[x + (grid_h-1-y) * grid_l] = sampleFieldVelocityMagnitude(x, y, threshold);
                if (cpuBarrier[x + y * grid_l]) {
                    ((color*)cpuColors)[x + (grid_h - 1 - y) * grid_l] = color(255, 255, 255);
                }
            }
        }
    }

    if (strcmp(sampleType, "directional magnitude") == 0) {
        for (int x = 0; x < grid_l; x++) {
            for (int y = grid_h-1; y >= 0; y--) {
                ((color*)cpuColors)[x + (grid_h-y-1) * grid_l] = sampleFieldVelocityDirectionalMagnitude(x, y, threshold);
                if (cpuBarrier[x + y * grid_l]) {
                    ((color*)cpuColors)[x + (grid_h-1-y) * grid_l] = color(255, 255, 255);
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


// called every frame
void updateFluid(float v) {
    
    gaussianDivergenceSolver(500);
    semiLagrangianAdvection();
    moveMainArrayToCPU();
    fillColorArray(v, "directional magnitude");
}

int main()
{
    // alloc and set all global device arrays/pointers to 0
    allocDeviceVars();
    resetVecs();
    resetBars();

    float fluidvel = 20.0f;

    // for now, make a square barrier
    int xcenter = 100;
    int ycenter = 100;
    int radius = 30;
    for (int x = 0; x < grid_l; x++) {
        for (int y = 0; y < grid_h; y++) {
            if (sqrt((x - xcenter) * (x - xcenter) + (y - ycenter) * (y - ycenter)) < radius) {
                setBar(x, y);
            }
        }
    }

    for (int x = 0; x < 20; x++) {
        for (int y = 0; y < grid_h; y++) {
            setHorVecs(vec2(fluidvel, 0.0f), x, y);
        }
    }

    moveBarrierToCPU();
    

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Cuda-openGL Interop", NULL, NULL);

    glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);  // Remove window decorations
    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);  // Make the window non-resizable

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

    clock_t start, end;
    int frametime = 0;
    unsigned int frame = 0;
    while (!glfwWindowShouldClose(window))
    {
        start = clock();
        processInput(window);
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        int ind = 0;
        // **
        // dodaj boje tu u pixels
        updateFluid(fluidvel);
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
        end = clock();
        if (frame % 10 == 1) {
            int fps = 10000 / (frametime);
            fps = fps < 100 ? fps : 99;
            printf("\rFPS: %d", fps);
            frametime = 0;
        }
        frame++;
        frametime += end - start;
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