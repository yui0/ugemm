//---------------------------------------------------------
//	Cat's eye
//
//		©2020 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os gpgpu_gl4.c -o gpgpu_gl4 `pkg-config --libs --cflags gl egl gbm` -lglfw
// dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel glfw-
#include "gpgpu_gl4.h"

// https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm.htm
#define TS 32u
#define WPT 8u                 // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS 4u                 // The reduced tile-size in one dimension  TS/WPT
#define TSDK 16u               // The tile-size in dimension K (for kernel 5 only)
#define LPT ((TSDK*WPT)/(TS))  // The amount of loads-per-thread (assume TSN==TSM)
static const char compute_shader_source[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = TS, local_size_y = RTS, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
  float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
  float B[];
};
layout (std430, binding = 2) writeonly buffer ssbC {
  float C[];
};
uniform int param[16]; // 0:M 1:N 2:K

shared float Asub[TSDK][TS];    // Local memory to fit a tile of A and B
shared float Bsub[TS][TSDK+2u];

void main() {
    int M = param[0];
    int N = param[1];
    int K = param[2];

    // Thread identifiers
    uint row = gl_LocalInvocationID.x; // Local row ID (max: TS)
    uint col = gl_LocalInvocationID.y; // Local col ID (max: TS/WPT == RTS)
    uint globalRow = TS*gl_WorkGroupID.x + row; // Row ID of C (0..M)
    uint globalCol = TS*gl_WorkGroupID.y + col; // Col ID of C (0..N)

    if (M<=globalRow) return;
    if (N<=globalCol) return;

    // Initialise the accumulation registers
    float acc[WPT];
    for (uint w=0u; w < WPT; w++) acc[w] = 0.0;

    // Loop over all tiles
    uint numTiles = K/TSDK;
    for (uint t=0u; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        for (uint l=0u; l < LPT; l++) {
            uint tiledIndex = TSDK*t + col + l*RTS;
            uint indexA = (tiledIndex)*M + TS*gl_WorkGroupID.x + row;
            uint indexB = (tiledIndex)*N + TS*gl_WorkGroupID.y + row;
            Asub[col + l*RTS][row] = A[indexA];
            Bsub[row][col + l*RTS] = B[indexB];
        }
        // Synchronise to make sure the tile is loaded
        barrier();

        // Perform the computation for a single tile
        for (uint k=0u; k < TSDK; k++)
            for (uint w=0u; w < WPT; w++)
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];

        // Synchronise before loading the next tile
        barrier();
    }
    // Store the final result in C
    for (uint w=0u; w < WPT; w++)
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
}

);

GLuint sgemm_gl_program;
void sgemm_gl_init(int s1, int s2, int s3)
{
	coInit();
	sgemm_gl_program = coCreateShaderProgram(compute_shader_source);

	int size[] = {s1, s2, s3};
	coCreateBuffer(size, 3);
}
void sgemm_gl_finish()
{
	coDeleteBuffer();
	coDeleteProgram(sgemm_gl_program);
}
inline void sgemm_gl(char ta, char tb, int m, int n, int k, float *a, float *b, float *c)
{
	int param[16];
	param[0] = m;
	param[1] = n;
	param[2] = k;
	coWrite(0, m*k*sizeof(float), a);
	coWrite(1, k*n*sizeof(float), b);
	coRun(sgemm_gl_program, m/TS+1, n/RTS+1, 1, param);
	coRead(2, m*n*sizeof(float), c);
	//for (int i=0; i<100; i++) printf("%f ", c[i]);
}

