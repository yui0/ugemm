//---------------------------------------------------------
//	Cat's eye
//
//		©2020 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os gpgpu_gl4.c -o gpgpu_gl4 `pkg-config --libs --cflags gl egl gbm` -lglfw
// dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel glfw-
#include "gpgpu_gl4.h"

// https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm.htm
static const char compute_shader_source[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
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

void main() {
    int M = param[0];
    //int N = param[1];
    int K = param[2];

    // Thread identifiers
    uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
    uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0;
    for (uint k=0u; k < K; k++)
        acc += A[k*M + globalRow] * B[globalCol*K + k];

    // Store the result
    C[globalCol*M + globalRow] = acc;
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
//	coWrite(2, m*n*sizeof(float), c);
	coRun(sgemm_gl_program, m/8+1, n/8+1, 1, param);
//	coRun(sgemm_gl_program, 1, 1, 1, param);
	coRead(2, m*n*sizeof(float), c);
	for (int i=0; i<100; i++) printf("%f ", c[i]);

//	coRead(0, m*k*sizeof(float), a);
//	for (int i=0; i<100; i++) printf("%f ", a[i]);
}

