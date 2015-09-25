/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _CUDA_ACCELERATION_H
#define _CUDA_ACCELERATION_H

#include <cufft.h>

#define BSPLIT 29
#define PADDED_DATA_SIZE 1179648

#if defined(_MSC_VER) || defined(__linux__) || defined(__APPLE__)
#include <builtin_types.h>
#include "boinc_api.h"
#include "s_util.h"
#include "analyzePoT.h"

/*
** petri33:
**
** Cuda 7.5 compiler produces 2 float2 store operations from *(float4*)p = f4;
** These helper functions implement st.128 in one operation.
**
** Other helper for cache configuration on load/store operations
*/

// store 1 float caching (likely to be reused soon)
__device__ void inline ST_f_wb(float *addr, float x)
{
  asm("st.global.wb.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 float streaming, not caching (not likely to be reused soon)
__device__ void inline ST_f_cs(float *addr, float x)
{
  asm("st.global.cs.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 float caching L2 (likely to be reused)
__device__ void inline ST_f_cg(float *addr, float x)
{
  asm("st.global.cg.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 float caching L2 (likely to be reused)
__device__ void inline ST_f4_cg(float4 *addr, float4 val)
{
  asm("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(val.x),"f"(val.y),"f"(val.z),"f"(val.w));
}

// store 4 floats
__device__ void inline ST_4f(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4
__device__ void inline ST_f4(float4 *addr, float4 val)
{
  ST_4f(addr, val.x, val.y, val.z, val.w);
}

// store 2 floats caching (likely to be reused soon)
__device__ void inline ST_2f_wb(float2 *addr, float x, float y)
{
  asm("st.global.wb.v2.f32 [%0], {%1,%2};" :: "l"(addr) ,"f"(x),"f"(y));
}

// store float2 (likely to be reused soon)
__device__ void inline ST_f2_wb(float2 *addr, float2 val)
{
  ST_2f_wb(addr, val.x, val.y);
}

// store 4 floats caching (likely to be reused soon)
__device__ void inline ST_4f_wb(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.wb.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4 (likely to be reused soon)
__device__ void inline ST_f4_wb(float4 *addr, float4 val)
{
  ST_4f_wb(addr, val.x, val.y, val.z, val.w);
}

// store 2 floats streaming (not likely to be reused immediately)
__device__ void inline ST_2f_cs(float2 *addr, float x, float y)
{
  asm("st.global.cs.v2.f32 [%0], {%1,%2};" :: "l"(addr) ,"f"(x),"f"(y));
}

// store float4 (no likely to be reused immediately)
__device__ void inline ST_f2_cs(float2 *addr, float2 val)
{
  ST_2f_cs(addr, val.x, val.y);
}

// store 4 floats streaming (not likely to be reused immediately)
__device__ void inline ST_4f_cs(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4 (no likely to be reused immediately)
__device__ void inline ST_f4_cs(float4 *addr, float4 val)
{
  ST_4f_cs(addr, val.x, val.y, val.z, val.w);
}

//load through nonuniform cache

//ca = L1,L2
__device__ float inline LDG_f_ca(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm("ld.global.ca.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_ca(float2 *addr, const int offset)
{
  float2 v;
  addr += offset;
  asm("ld.global.ca.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr));
  return v; 
}

__device__ float4 inline LDG_f4_ca(float4 *addr, const int offset)
{
  float4 v;
  addr += offset;
  asm("ld.global.ca.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr));
  return v; 
}

//cg = L2
__device__ float inline LDG_f_cg(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm("ld.global.cg.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_cg(float2 *addr, const int offset)
{
  float2 v;
  addr += offset;
  asm("ld.global.cg.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr));
  return v; 
}

__device__ float4 inline LDG_f4_cg(float4 *addr, const int offset)
{
  float4 v;
  addr += offset;
  asm("ld.global.cg.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr));
  return v; 
}

//streaming once through L1,L2
__device__ float inline LDG_f_cs(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm("ld.global.cs.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_cs(float2 *addr, const int offset)
{
  float2 v;
  addr += offset;
  asm("ld.global.cs.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr));
  return v; 
}

__device__ float4 inline LDG_f4_cs(float4 *addr, const int offset)
{
  float4 v;
  addr += offset;
  asm("ld.global.cs.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr));
  return v; 
}

//last use L1,L2 (same as cs on global addresses)
__device__ float inline LDG_f_lu(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm("ld.global.lu.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_lu(float2 *addr, const int offset)
{
  float2 v;
  addr += offset;
  asm("ld.global.lu.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr));
  return v; 
}

__device__ float4 inline LDG_f4_lu(float4 *addr, const int offset)
{
  float4 v;
  addr += offset;
  asm("ld.global.lu.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr));
  return v; 
}

__device__ float4 inline LDE_f4_cs(float4 *addr, const int offset)
{
  float4 v;
  addr += offset;
  asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr));
  return v; 
}

__device__ float inline LDE_f_cs(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm("ld.global.cs.f32 {%0}, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}


__device__ void inline prefetch_l1(void *addr)
{
    asm volatile ("prefetch.global.L1 [%0];"::"l"(addr) );
}

__device__ void inline prefetch_l2(void *addr)
{
    asm volatile ("prefetch.global.L2 [%0];"::"l"(addr) );
}



__device__ float inline __fmul_sat(float a, float b)
{
  float res;
  asm("mul.rn.sat.f32 %0, %1, %2 ;" : "=f"(res) : "f"(a), "f"(b));
  return res;
}

__device__ double inline __cvt_d_rzi(double a)
{
  double res;
  asm("cvt.rzi.f64.f64 %0, %1 ;" : "=d"(res) : "d"(a));
  return res;
}

__device__ float inline __cvt_f_rzi(float a)
{
  float res;
  asm("cvt.rzi.f32.f32 %0, %1 ;" : "=f"(res) : "f"(a));
  return res;
}


extern double *angle_range;
extern unsigned cmem_rtotal;

extern cudaEvent_t fftDoneEvent;
extern cudaEvent_t summaxDoneEvent;
extern cudaEvent_t powerspectrumDoneEvent;

extern cudaEvent_t autocorrelationDoneEvent;
extern cudaEvent_t autocorrelationRepackDoneEvent;
extern cudaEvent_t ac_reduce_partialEvent; 

extern cudaEvent_t meanDoneEvent;
extern cudaEvent_t tripletsDoneEvent;
extern cudaEvent_t pulseDoneEvent;
extern cudaEvent_t pulseCalcDoneEvent;
extern cudaEvent_t gaussDoneEvent;
extern cudaEvent_t gaussDoneEvent2;

extern cudaStream_t fftstream0; // fft, triplet
extern cudaStream_t fftstream1; // pulse
extern cudaStream_t cudaAutocorrStream; // autocorr
extern cudaStream_t pulseStream; // pulse
extern cudaStream_t tripletStream; // triplet
extern cudaStream_t gaussStream; // gauss
extern cudaStream_t summaxStream; // summax

//Public function prototypes
int  cudaAcc_initializeDevice(int devPref, int usePolling);
int  cudaAcc_initialize(sah_complex* cx_DataArray, int NumDataPoints, int gauss_pot_length, unsigned long nsamples, 
			double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh,
			double pulse_display_thresh, double PulseThresh, int PulseMax,
			double sample_rate, long acfftlen );
int  cudaAcc_initialize_pulse_find(double pulse_display_thresh, double PulseThresh, int PulseMax);
int cudaAcc_InitializeAutocorrelation(int ac_fftlen);
void cudaAcc_free();
void cudaAcc_free_Gaussfit();
void cudaAcc_free_pulse_find();
void cudaAcc_free_AutoCorrelation();
void cudaAcc_CalcChirpData(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray);
void cudaAcc_CalcChirpData_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream);
void cudaAcc_CalcChirpData_sm13(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray);
void cudaAcc_CalcChirpData_sm13_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream);
int cudaAcc_fftwf_plan_dft_1d(int FftNum, int FftLen, int NumDataPoints);
void cudaAcc_execute_dfts(int FftNum, int offset);
void cudaAcc_GetPowerSpectrum(int numpoints, int FftNum, int offset, cudaStream_t str, float *dct_In, int fftlen);
void cudaAcc_SetPowerSpectrum(float* PowerSpectrum);
int  cudaAcc_initializeGaussfit(const PoTInfo_t& PoTInfo, int gauss_pot_length, unsigned int nsamples, double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh);
int cudaAcc_GaussfitStart(int gauss_pot_length, double best_gauss_score, bool noscore, int offset);
int cudaAcc_fetchGaussfitFlags(int gauss_pot_length, double best_gauss_score);
int cudaAcc_processGaussFit(int gauss_pot_length, double best_gauss_score);
void cudaAcc_summax(int fftlen, int offset);
void cudaAcc_summax_x(int fftlen);
int cudaAcc_calculate_mean(int PulsePoTLen, float triplet_thresh, int AdvanceBy, int FftLength, int offset);
int cudaAcc_find_triplets(int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength, int offset);
int cudaAcc_fetchTripletFlags(bool SkipTriplet, int PulsePoTLen, int AdvanceBy, int FftLength, int offset);
int cudaAcc_fetchPulseFlags(bool SkipPulse, int PulsePoTLen, int AdvanceBy, int FftLength, int offset);
int cudaAcc_processTripletResults(int PulsePoTLen, int AdvanceBy, int FftLength);//int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength);
int cudaAcc_find_pulses(float best_pulse_score, int PulsePotLen, int AdvanceBy, int fft_length, int offset);
int cudaAcc_processPulseResults(int PulsePoTLen, int AdvanceBy, int FftLength);//int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength);

//V7 Autocorrelation 
int cudaAcc_FindAutoCorrelations(int ac_fftlen, int offset);
int cudaAcc_GetAutoCorrelation(float *AutoCorrelation, int ac_fftlen, int fft_num);

//Referenced globals
extern float3* PowerSpectrumSumMax;
extern int gSetiUseCudaDevice;
extern cudaDeviceProp gCudaDevProps;
//extern float ac_TotalSum;
//extern float ac_Peak;
//extern int ac_PeakBin;
extern float3 *dev_ac_partials;
extern float3 *blockSums;
// DEFINES
#define USE_CUDA 1   // Allows the CUDA code path to be compiled in and used if defined

//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 ) 
        slices *= 2;
    return dim3( nblocks/slices, slices );
}

#if (CUDART_VERSION < 4000)
	#define CUDASYNC cudaThreadSynchronize()
#else
	#define CUDASYNC cudaDeviceSynchronize()
#endif

#endif //_MSC_VER

#endif //_CUDA_ACCELERATION_H
