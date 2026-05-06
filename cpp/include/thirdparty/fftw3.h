/*
 * Minimal fftw3.h stub — declares only what phycam-eval uses.
 * The full libfftw3 runtime is at /usr/lib/aarch64-linux-gnu/libfftw3.so.3
 * This header exposes only the double-precision C API subset we need.
 *
 * Generated from FFTW 3.3.8 public API (BSD-like licence).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Types --- */
typedef double fftw_complex[2];   /* [0]=real, [1]=imag */
typedef struct fftw_plan_s *fftw_plan;

/* --- Direction constants --- */
#define FFTW_FORWARD  (-1)
#define FFTW_BACKWARD (+1)

/* --- Planner flags --- */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_MEASURE  (0U)

/* --- Core API --- */
fftw_plan fftw_plan_dft_2d(int n0, int n1,
                            fftw_complex *in, fftw_complex *out,
                            int sign, unsigned flags);

fftw_plan fftw_plan_dft_1d(int n,
                            fftw_complex *in, fftw_complex *out,
                            int sign, unsigned flags);

void fftw_execute(const fftw_plan p);
void fftw_execute_dft(const fftw_plan p,
                      fftw_complex *in, fftw_complex *out);
void fftw_destroy_plan(fftw_plan p);

void *fftw_malloc(size_t n);
void  fftw_free(void *p);

/* Normalisation: FFTW does NOT normalise; divide by (n0*n1) for IFFT. */

#ifdef __cplusplus
} /* extern "C" */
#endif
