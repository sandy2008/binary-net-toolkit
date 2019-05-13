#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define MAX_FILTER_BYTES 12
static const uint8_t bits[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};


/* layer types */
static void blinear_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int m, const int n,
                          const int k);
static void fconv_layer(const float* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);
static void bconv_layer(const uint8_t* A, const uint8_t* F, uint8_t* C,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int m,
                        const int num_f, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);

/* layer helper functions */
static float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std);
static void fconv(const float* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph);
static void bconv(const uint8_t* A, const uint8_t* F, uint8_t* C,
                  const int c_start_idx, const int z, const float Bias,
                  const float Gamma, const float Beta, const float Mean,
                  const float Std, const int w, const int h, const int d,
                  const int kw, const int kh, const int sw, const int sh,
                  const int pw, const int ph, const int pl_w, const int pl_h,
                  const int pl_sw, const int pl_sh, const int pl_pw,
                  const int pl_ph);
static float fdot_3d(const float* A, const uint8_t* B, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh);
static int bdot_3d(const uint8_t* A, const uint8_t* B, const int x, const int y,
                   const int z, const int w, const int h, const int d,
                   const int kw, const int kh);
static int bdot(const uint8_t* A, const uint8_t* B, const int N);

/* indexing functions */
static int idx_2d(const int i, const int j, const int rows);
static int idx_3d(const int i, const int j, const int k, const int rows,
                  const int cols);
static int idx_4d(const int i, const int j, const int k, const int l,
                  const int rows, const int cols, const int depth);
static int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px);
static int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px);

/* Bit functions */
static uint8_t rotr1 (const uint8_t x);
static int popcnt8(const uint8_t v);
static int nthbitset_arr(const uint8_t* const arr, const int n);
static int bslice_2d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int w, const int h, const int kw,
                     const int kh);
static int bslice_2d_filter(uint8_t* const dst, const uint8_t* const src,
                            const int x, const int y, const int w, const int h,
                            const int kw, const int kh);
static int bslice_4d(uint8_t* const dst, const uint8_t* const src, const int x,
                     const int y, const int zi, const int zj, const int w,
                     const int h, const int d, const int kw, const int kh);
