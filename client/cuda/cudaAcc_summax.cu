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

#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"

#include "cudaAcc_utilities.h"

float3 *dev_PowerSpectrumSumMax;
float3 *PowerSpectrumSumMax;

extern __shared__ float3 partial[];
// At least 32elems per sum
template <int iterations>
__global__ void cudaAcc_summax32_kernel(float *input, float3 *output) 
{    
  const int tid = threadIdx.x;
  const int y = blockIdx.y * blockDim.y;
  const int gridX = blockDim.x;
  const int start = y * gridX * iterations;
  const int end = (y + 1) * gridX * iterations;
  

  float maximum = 0.0f;    
  int pos = 0;
  int i = start + tid;
  float *ip = &input[i];
  float val = *ip, val2, val3, val4;
  float sum = val, sum2 = 0.0f;

  if(tid > 0) 
    { // Original max omits first element of every fft
      if(val > maximum)
	pos = i;
      maximum = max(maximum, val);
    }

  ip = &input[start + tid + gridX];  
  for(i = start + tid + gridX; i < (end-gridX*3); i += gridX) 
    {
      val  = LDG_f_cs(ip, 0); ip += gridX;  
      val2 = LDG_f_cs(ip, 0); ip += gridX;  
      val3 = LDG_f_cs(ip, 0); ip += gridX;  
      val4 = LDG_f_cs(ip, 0); ip += gridX;  

      sum +=  val + val2;
      sum2 += val3 + val4;
      if(val > maximum)
	{
	  pos = i;
	  maximum = val;
	}
      i += gridX;
      
      if(val2 > maximum)
	{
	  pos = i;
	  maximum = val2;
	}
      i += gridX;
      
      if(val3 > maximum)
	{
	  pos = i;
          maximum = val3;
	}
      i += gridX;
      
      if(val4 > maximum)
	{
	  pos = i;
          maximum = val4;
	}
    }
  
  // tail
  for(; i < end; i += gridX) 
    {
      val = LDG_f_cs(ip, 0);
      ip += gridX;  
      sum = sum + val;
      if(val > maximum)
	{
	  pos = i;
	  maximum = val;
	}
    }

  float a, ay, az;
  float3 *pp = &partial[tid];
  *pp = make_float3(a = sum+sum2, ay = maximum, az = pos - start);
  int padd = (gridX>>1)*12;

  for(i = gridX >> 1; i > 0; i >>= 1) 
    {
      __syncthreads(); 
      if(tid < i) 
	{            
	  a = a + (*(float3 *)(((char *)pp)+padd)).x;
	  float by = (*(float3 *)(((char *)pp)+padd)).y;
	  float bz = (*(float3 *)(((char *)pp)+padd)).z;

	  if(by > ay)
	    {
	      ay = by;
	      az = bz;
	    }

	  pp[0].x = a;
	  pp[0].y = ay;
	  pp[0].z = az;
	  
	  padd >>= 1;
	}
    }

  if(tid == 0) 
    {        
      output[blockIdx.y] = make_float3(a, ay, az);
    }
}


//less than 32 elems
template<int n>
__global__ void cudaAcc_summax_kernel(float* input, float3* output)
{
  const int tid = threadIdx.x;
  const int y = blockIdx.y * blockDim.y;
  const int gridX = gridDim.x * blockDim.x;
  const int start = y * gridX;
  
  const int n1 = n - 1;
  const int width = blockDim.x / n;
  
  float val = input[start + tid];

  partial[tid] = make_float3(val, ((tid & n1) == 0 ? 0.0f : val), tid & n1); // Original max omits first element of every fft

  float a = partial[tid].x;
  float ay = partial[tid].y;
  float az = partial[tid].z;
  for(int i = n >> 1; i > 0; i >>= 1)
    {
      __syncthreads(); 
      if((tid & n1) < i) 
	{
	  a = a + partial[tid+i].x;
	  float by = partial[tid+i].y;
	  float bz = partial[tid+i].z;

	  if(by > ay)
	    {
	      ay = by;
	      az = bz;
	    }
	  
	  partial[tid].x = a;
	  partial[tid].y = ay;
	  partial[tid].z = az;
	}
    }
  __syncthreads();
  
  if(tid < width)
    {
      output[blockIdx.y * width + tid] = make_float3(a, ay, az); //partial[tid * n];
    }
}

//Jason:---------------------------------------------------------------------------
template<class T>
struct SharedMemory
{
  __device__ inline operator       T*()
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
  
  __device__ inline operator const T*() const
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

template <unsigned int fftlen> 
__global__ void cudaAcc_SM(float *PowerSpectrum, float3 *devPowerSpectrumSumMax) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx = (iblock*blockDim.x + threadIdx.x); 
  int tid = threadIdx.x;
  
  float3 *sdata = SharedMemory<float3>();
  
  float mySum = sdata[tid].x = PowerSpectrum[sidx];
  sdata[tid].z = tid;
  sdata[tid].y = (tid == 0 ? 0.0f :  mySum);
  __syncthreads();

  if (tid < fftlen/2) // last part of reduction is warp synchronous;
    {
      volatile float3* smem = sdata;
      if(fftlen >=  64) 
	{
	  smem[tid].x = mySum = mySum + smem[tid + 32].x;
	  smem[tid].y = max(smem[tid].y, smem[tid + 32].y);
	  smem[tid].z = smem[tid].y > smem[tid + 32].y ? smem[tid].z : smem[tid + 32].z; 
	}
      if(fftlen >=  32) 
	{
	  smem[tid].x = mySum = mySum + smem[tid + 16].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 16].y);
	  smem[tid].z = smem[tid].y > smem[tid + 16].y ? smem[tid].z : smem[tid + 16].z; 
	}
      if(fftlen >=  16) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  8].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 8].y);
	  smem[tid].z = smem[tid].y > smem[tid + 8].y ? smem[tid].z : smem[tid + 8].z; 
	}
      if(fftlen >=   8) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  4].x;
	  smem[tid].y = max(smem[tid].y, smem[tid + 4].y);
	  smem[tid].z = smem[tid].y > smem[tid + 4].y ? smem[tid].z : smem[tid + 4].z; 
	}
      if(fftlen >=   4) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  2].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 2].y);
	  smem[tid].z = smem[tid].y > smem[tid + 2].y ? smem[tid].z : smem[tid + 2].z; 
	}
      if(fftlen >=   2) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  1].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 1].y);
	  smem[tid].z = smem[tid].y > smem[tid + 1].y ? smem[tid].z : smem[tid + 1].z; 
	}
    }
  
  if(tid==0)
    {
      devPowerSpectrumSumMax[sidx/fftlen] = sdata[0];
    }
}

// TODO: optimize Memcpy, download data only when are going to be reported: 
//							if (si.score > best_spike->score || best_spike->s.fft_len == 0)
//							if (si.s.peak_power > (swi.analysis_cfg.spike_thresh))
void cudaAcc_summax(int fftlen, int offset) 
{
  cudaStreamWaitEvent(summaxStream, powerspectrumDoneEvent, 0);

  dim3 block2(fftlen, 1, 1);
  dim3 grid2 = grid2D((cudaAcc_NumDataPoints + block2.x - 1) / block2.x);
  
  if(fftlen >= 32 && cudaAcc_NumDataPoints/fftlen < 65536)
    {
      int optimal_block_x;
      //if(gCudaDevProps.major > 2)
	optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 1024)); 
      //else
	//optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 512));
      dim3 block(optimal_block_x, 1, 1);;
      dim3 grid(1, cudaAcc_NumDataPoints / fftlen, 1);     
      int iterations = fftlen/block.x;
      
    
      switch(iterations)
      {
        case 1:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<1><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 2:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<2><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 4:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<4><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<8><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<16><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<32><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<64><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<128><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
        case 256:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<256><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	default:
	  fprintf(stderr, "cudaAcc_summax32_kernel template error\r\n");
	  fflush(stderr);
	  *(char *)0 = 0;
      }     
    }
  else 
    {
      // Occupancy Calculator: 128 for cc1.x, 256 for cc2.x
      dim3 block(128, 1, 1);
      if(gCudaDevProps.major >= 2) 
	{
#if CUDART_VERSION >= 3000
//	  cudaFuncSetCacheConfig(cudaAcc_summax32_kernel, cudaFuncCachePreferShared); // Set this at init time
#endif
	  block.x = 256; //256 //128
	}
      dim3 grid(1, cudaAcc_NumDataPoints / block.x, 1);
      switch (fftlen) 
	{
	case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<128><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<64><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<32><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<16><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<8><<<grid, block, (block.x * sizeof(float3)),summaxStream>>>(dev_PowerSpectrum + offset, dev_PowerSpectrumSumMax)),true);
	  break;
	}
    }
  
  
  cudaMemcpyAsync(PowerSpectrumSumMax, dev_PowerSpectrumSumMax, (cudaAcc_NumDataPoints / fftlen) * sizeof(*dev_PowerSpectrumSumMax), cudaMemcpyDeviceToHost, summaxStream);    
  cudaEventRecord(summaxDoneEvent, summaxStream);
}


void cudaAcc_summax_x(int fftlen) // not used anymore
{
  dim3 block2(fftlen, 1, 1);
  dim3 grid2 = grid2D((cudaAcc_NumDataPoints + block2.x - 1) / block2.x);
  
  if(fftlen >= 32 && cudaAcc_NumDataPoints/fftlen < 65536)
    {
      int optimal_block_x;

      optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 1024));
      
      dim3 block(optimal_block_x, 1, 1);;
      dim3 grid(1, cudaAcc_NumDataPoints / fftlen, 1);     
      int iterations = fftlen/block.x;

      switch(iterations)
      {
        case 1:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<1><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 2:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<2><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 4:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<4><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<8><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<16><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<32><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<64><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<128><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 256:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<256><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	default:
	  fprintf(stderr, "cudaAcc_summax32_kernel template error\r\n");
	  fflush(stderr);
	  *(char *)0 = 0;
      }     
    }
  else 
    {
      // Occupancy Calculator: 128 for cc1.x, 256 for cc2.x
      dim3 block(128, 1, 1);
      if(gCudaDevProps.major >= 2) 
	{
#if CUDART_VERSION >= 3000
//	  cudaFuncSetCacheConfig(cudaAcc_summax32_kernel, cudaFuncCachePreferShared); // Set this at init time
#endif
	  block.x = 256; //64
	}
      dim3 grid(1, cudaAcc_NumDataPoints / block.x, 1);
      switch (fftlen) 
	{
	case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<128><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<64><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<32><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<16><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<8><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	}
    }
  
  
  cudaMemcpyAsync(PowerSpectrumSumMax, dev_PowerSpectrumSumMax, (cudaAcc_NumDataPoints / fftlen) * sizeof(*dev_PowerSpectrumSumMax), cudaMemcpyDeviceToHost);    
}

#endif //USE_CUDA
