//---------------------------------------------------------
//	Cat's eye
//
//		©2020 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os gpgpu_gl4.c -o gpgpu_gl4 `pkg-config --libs --cflags gl egl gbm` -lglfw
// dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel glfw-
#include "gpgpu_gl4.h"

// https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm.htm
#define TSM 128                     // The tile-size in dimension M
#define TSN 128                     // The tile-size in dimension N
#define TSK 16                      // The tile-size in dimension K
#define WPTM 8                      // The amount of work-per-thread in dimension M
#define WPTN 8                      // The amount of work-per-thread in dimension N
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B
#define RTSM 16    // The reduced tile-size in dimension M (TSM/WPTM number of threads)
#define RTSN 16    // The reduced tile-size in dimension N (TSN/WPTN number of threads)
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
static const char compute_shader_source[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = RTSM, local_size_y = RTSN, local_size_z = 1) in;
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

shared float Asub[TSK][TSM];    // Local memory to fit a tile of A and B
shared float Bsub[TSN][TSK+2];

void main() {
    int M = param[0];
    int N = param[1];
    int K = param[2];

    // Thread identifiers
    int tidm = int(gl_LocalInvocationID.x);  // Local row ID (max: TSM/WPTM == RTSM)
    int tidn = int(gl_LocalInvocationID.y);  // Local col ID (max: TSN/WPTN == RTSN)
    int offsetM = TSM*int(gl_WorkGroupID.x); // Work-group offset
    int offsetN = TSN*int(gl_WorkGroupID.y); // Work-group offset

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    for (int wm=0; wm < WPTM; wm++) {
        for (int wn=0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0;
        }
    }
    // Loop over all tiles
    int numTiles = K/TSK;
    int t=0;
    do {
        // Load one tile of A and B into local memory
        for (int la=0; la < LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }
        // Synchronise to make sure the tile is loaded
        barrier();

        // Loop over the values of a single tile
        for (int k=0; k < TSK; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn < WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            for (int wm=0; wm < WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier();

        // Next tile
        t++;
    } while (t < numTiles);

    // Store the final result in C
    for (int wm=0; wm < WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn < WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

);

GLuint sgemm_gl_program;
void sgemm_gl_init(int s1, int s2, int s3)
{
	coInit();
	sgemm_gl_program = coCreateShaderProgram(compute_shader_source);

	int size[] = {s1, s2, s3};
	coCreateBuffer(sgemm_gl_program, size, 3);
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
	coRun(sgemm_gl_program, m/RTSM+1, n/RTSN+1, 1, param);
	coRead(2, m*n*sizeof(float), c);
}

