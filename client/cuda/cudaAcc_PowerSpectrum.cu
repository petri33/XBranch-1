#include "cudaAcceleration.h"

#ifdef USE_CUDA
#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

//2, 16, 128
#define B 2
#define D 16
#define D2 8
#define blockDim_x 32

#if 1
//__launch_bounds__(blockDim_x)
__global__ void cudaAcc_GPS_kernel_mod3SM(float2 * FreqData, float * PowerSpectrum) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx   = threadIdx.x + D * iblock * blockDim_x; 
  
  float4 a[D];
  float2 b[D];
  float4 *ip = ((float4 *)FreqData) + sidx;
  float2 *op = ((float2 *)PowerSpectrum) + sidx;

  for(int i = 0; i < D; i++) 
    {
      a[i] =  LDG_f4_cs(&ip[i * blockDim_x], 0);
    }
    
  for(int i = 0; i < D; i++) 
    {
      b[i].x = a[i].x * a[i].x + a[i].y * a[i].y;
      b[i].y = a[i].z * a[i].z + a[i].w * a[i].w;
    }
    
  for(int i = 0; i < D; i++) 
    {
      ST_f2_cs(&op[i * blockDim_x], b[i]);
    }
}



__global__ void cudaAcc_GPS_kernel_mod3SM_repack(float2 *FreqData, float *PowerSpectrum, float *dct_In) 
{
  int sidx = threadIdx.x + blockIdx.x * 32;
  int sidxp = sidx + (blockIdx.y << 17);
  FreqData += sidxp;
  PowerSpectrum += sidxp;
  float2 a;
  float b;

  a = LDG_f2_cs(FreqData, 0);
  b = a.x * a.x + a.y * a.y;

  ST_f_cs(PowerSpectrum, b);

  float4 *dct = (float4*)dct_In + (blockIdx.y << 18) + sidx;
  float4 *dct2 = (float4*)dct_In + (blockIdx.y << 18) + (2*131072-1) - sidx;
  float4 t = make_float4(0.0f, 0.0f, b, 0.0f);
  
  ST_f4_cs(dct, t);
  ST_f4_cs(dct2, t);
}

#else
__global__ void cudaAcc_GPS_kernel_mod3SM(float2 * FreqData, float * PowerSpectrum) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx = threadIdx.x + B*iblock*blockDim_x; 
  
  float2 a[B];
  
#pragma unroll
  for(int i = 0; i < B; i++) 
    {
      a[i] = FreqData[sidx+i*blockDim_x]; 
    }

#pragma unroll
  for(int i = 0; i < B; i++)
    PowerSpectrum[sidx+i*blockDim_x] =__fadd_rn( __fmul_rn(a[i].x,a[i].x),__fmul_rn(a[i].y,a[i].y)); 
}

__global__ void cudaAcc_GPS_kernel_mod3( int NumDataPoints, float2* FreqData, float* PowerSpectrum) 
{
  const int sidx = (blockIdx.x * blockDim.x + threadIdx.x); 
  
  float ax,ay;
  
  if ( sidx < NumDataPoints )
    {
      ax = FreqData[sidx].x;
      ay = FreqData[sidx].y;
      PowerSpectrum[sidx] =  __fadd_rn( __fmul_rn(ax,ax),__fmul_rn(ay,ay)); 
    }
}
#endif

void cudaAcc_GetPowerSpectrum(int numpoints, int FftNum, int offset, cudaStream_t stream, float *dct_In, int fftlen) 
{
  if(fftlen == 131072)
    {
      dim3 block(32, 1, 1);
      dim3 grid( ((131072) + (block.x) - 1) / (block.x), 8, 1); // does 8*131072 (1M) points
      cudaStreamWaitEvent(stream, fftDoneEvent, 0);
      
      cudaAcc_GPS_kernel_mod3SM_repack<<<grid, block, 0, stream>>>(dev_WorkData + FftNum * 2 * PADDED_DATA_SIZE + offset, dev_PowerSpectrum + offset, dct_In); //
    }
  else
    {
      dim3 block(blockDim_x, 1, 1);
      //	dim3 grid((numpoints + (block.x*B) - 1) / (block.x*B), 1, 1);
      dim3 grid = grid2D((numpoints + (block.x*B*D) - 1) / (block.x*B*D));
      
      cudaStreamWaitEvent(stream, fftDoneEvent, 0);
      
      cudaAcc_GPS_kernel_mod3SM<<<grid, block, 0, stream>>>(dev_WorkData + FftNum * 2 * PADDED_DATA_SIZE + offset, dev_PowerSpectrum + offset); //
    }
  
  cudaEventRecord(powerspectrumDoneEvent, stream);
}

#endif //USE_CUDA
