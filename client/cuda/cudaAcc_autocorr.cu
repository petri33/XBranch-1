
#include "cudaAcceleration.h"

#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

//#define B 2


#define RPI 64
#define RPIB 8
#define RPBY 4

#define B 8
#define RPS 64
//must be 256 in current implementation
#define RDP 256

float2 *dev_AutoCorrIn;
float2 *dev_AutoCorrOut;

bool gCudaAutocorrelation = false;

float3 *blockSums;
float3 *dev_ac_partials;

//float ac_TotalSum;
//float ac_Peak;
//int ac_PeakBin;


__global__ void 
//__launch_bounds__(RPI, 8)
ac_RepackInputKernelP(float *PowerSpectrum, float2 *dct_In, int acfftlen) 
{
  int sidx = (threadIdx.x + blockIdx.x*RPI + (blockIdx.y*RPI*RPIB)); 
  int nDestPoints = acfftlen * 4; // (gridDim.y*RPI*RPIB)*4;
  int didx1 = sidx<<4; 
  int didx2 = ((nDestPoints-2)<<3)-didx1; 

  PowerSpectrum += sidx + blockIdx.z * acfftlen;
  float4 *dct1 = (float4*)((char *)dct_In + didx1 + blockIdx.z * acfftlen * 32);
  float4 *dct2 = (float4*)((char *)dct_In + didx2 + blockIdx.z * acfftlen * 32);
  float4 t = make_float4(0.0f, 0.0f, PowerSpectrum[0], 0.0f);
      
  *dct1 = t;
  *dct2 = t;
}



__global__ void 
//__launch_bounds__(RPS, 8)
ac_RepackScaleKernelP(float2 *src, float2 *dst, int acfftlen) 
{
  int didx = ((threadIdx.x + blockIdx.x*RPS + blockIdx.y*RPS*B));  //packing into float2s
  int sidx = didx << 1; //((threadIdx.x + blockIdx.x*RPS*B)*2);

//printf("bx=%d, by=%d, tx=%d, sidx=%d\r\n", blockIdx.x, blockIdx.y, threadIdx.x, sidx);
  dst += didx + blockIdx.z * acfftlen;
  src += sidx + blockIdx.z * acfftlen * 4;

  float4 t = ((float4 *)src)[0];
  float2 a = make_float2(t.x, t.z);

  a.x *= a.x;
  a.y *= a.y;

  dst[0] = a;
}


extern  __shared__ float acpartial[]; 

__global__ void ac_reducePartial(float *ac, float3 *devpartials, int streamIdx)
{
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x*RDP;
  const int bid = blockIdx.x;
  int n = RDP>>1;
  
  float3 *acp = (float3 *)acpartial; 
  float tmp = ac[idx];
  acp[tid].z = idx;
  acp[tid].y = idx >= 1 ? tmp : 0.0f; 
  acp[tid].x = tmp;
  
  __syncthreads();
  
  volatile float3 *dp = &acp[tid];
  int fadd = n * 12;
#pragma unroll 2
  for(; n > 32; n >>= 1)
    {
      if(tid < n)
	{
	  float a =  __fadd_rn(dp[0].x, (*(float3*)(((char *)dp)+fadd)).x);
	  // peak power & its bin
	  float pp = (*(float3*)(((char *)dp)+fadd)).y;
	  float pb = (*(float3*)(((char *)dp)+fadd)).z;
	  fadd >>= 1;
	  bool b = pp > dp[0].y;
          dp[0].x = a;
	  if(b)
	    {
	      dp[0].y = pp;
	      dp[0].z = pb;
	    }
	}
      __syncthreads();
    }

#pragma unroll 6
  for(; n > 0; n >>= 1)
    {
      if(tid < n)
	{
	  float a =  __fadd_rn(dp[0].x, (*(float3*)(((char *)dp)+fadd)).x);
	  // peak power & its bin
	  float pp = (*(float3*)(((char *)dp)+fadd)).y;
	  float pb = (*(float3*)(((char *)dp)+fadd)).z;
	  fadd >>= 1;
	  bool b = pp > dp[0].y;
          dp[0].x = a;
	  if(b)
	    {
	      dp[0].y = pp;
	      dp[0].z = pb;
	    }
	}
    }

  if(tid == 0) 
    {        
      devpartials[bid] = make_float3(dp[0].x, dp[0].y, dp[0].z);
    }
}

__global__ void ac_reducePartial16(float *ac, float3 *devpartials, int ac_fftlen)
{
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x*RDP;
  const int bid = blockIdx.x;
  
  float3 *acp = (float3 *)acpartial; 
  for(int f = 0; f < 8; f++)
    {
      float tmp = ac[idx+f*ac_fftlen*2];
      acp[tid].z = idx;
      acp[tid].y = idx >= 1 ? tmp : 0.0f; 
      acp[tid].x = tmp;

      __syncthreads();
  
      volatile float3 *dp = &acp[tid];
      int n = RDP>>1;
      int fadd = n * 12;

#pragma unroll 2
      for(;n > 32; n >>= 1)
	{
	  if(tid < n)
	    {
	      float a =  __fadd_rn(dp[0].x, (*(float3*)(((char *)(dp))+fadd)).x);
	      // peak power & its bin
	      float pp = (*(float3*)(((char *)(dp))+fadd)).y;
	      float pb = (*(float3*)(((char *)(dp))+fadd)).z;
	      bool b = pp > dp[0].y;
	      dp[0].x = a;
	      if(b)
		{
		  dp[0].y = pp;
		  dp[0].z = pb;
		}
	    }
	  
	  fadd >>= 1;
	  __syncthreads();
	}
      
#pragma unroll 6
      for(; n > 0; n >>= 1)
	{
	  if(tid < n)
	    {
	      float a =  __fadd_rn(dp[0].x, (*(float3*)(((char *)(dp))+fadd)).x);
	      // peak power & its bin
	      float pp = (*(float3*)(((char *)(dp))+fadd)).y;
	      float pb = (*(float3*)(((char *)(dp))+fadd)).z;
	      bool b = pp > dp[0].y;
	      dp[0].x = a;
	      if(b)
		{
		  dp[0].y = pp;
		  dp[0].z = pb;
		}
	    }
	  fadd >>= 1;
	}
      
      
      if(tid == 0) 
	devpartials[bid + f*(ac_fftlen/2)/RDP] = make_float3(dp[0].x, dp[0].y, dp[0].z);
    }
}


//__device__ float2 testPtr[8][10];

int cudaAcc_FindAutoCorrelations(int ac_fftlen, int offset) 
{
  cudaError_t err = cudaStreamWaitEvent(cudaAutocorrStream, powerspectrumDoneEvent, 0);	

  dim3 block(RPI, 1, 1);
  dim3 grid(RPIB, (ac_fftlen + (block.x*RPIB) - 1) / (block.x*RPIB), 8); 
  dim3 block2(RPS, 1, 1);
  dim3 grid2(B, ((ac_fftlen>>1)+block2.x*B-1)/(block2.x*B), 8);

  //Jason: Use 4N-FFT method for Type 2 Discrete Cosine Tranform for now, to match fftw's REDFT10
  // 1 Autocorrelation from global powerspectrum at fft_num*ac_fft_len  (fft_num*ul_NumDataPoints )
  
  //Step 1: Preprocessing - repack relevant powerspectrum into a 4N array with 'real-even symmetry'
  CUDA_ACC_SAFE_LAUNCH( (ac_RepackInputKernelP<<<grid, block, 0, cudaAutocorrStream>>>(dev_PowerSpectrum + offset, dev_AutoCorrIn, ac_fftlen)),true);

  //Step 2: Process the 4N-FFT (Complex to Complex, size is 4 * ac_fft_len)
  cufftExecC2C(cudaAutoCorr_plan, dev_AutoCorrIn, dev_AutoCorrOut, CUFFT_FORWARD);

  //Step 3: Postprocess the FFT result (Scale, take powers & normalise), discarding unused data packing into AutoCorr_in first half for VRAM reuse
  //CUDA_ACC_SAFE_LAUNCH( (
  ac_RepackScaleKernelP<<<grid2, block2, 0, cudaAutocorrStream>>>( dev_AutoCorrOut, dev_AutoCorrIn, ac_fftlen);
  //),true);

  int len = ac_fftlen/2;
  int blksize = RDP; 
  dim3 block3(blksize, 1, 1);
  dim3 grid3(len/blksize, 1, 1);

  CUDA_ACC_SAFE_LAUNCH( (ac_reducePartial16<<<grid3, block3, 3072, cudaAutocorrStream>>>( (float *)(dev_AutoCorrIn), dev_ac_partials, ac_fftlen)),true); // dynamic shared size is len/RDP*sizeof(float3) -> limit 4608

/*
  int len = ac_fftlen/2;
  int blksize = RDP; 
  dim3 block3(blksize, 1, 1);
  dim3 grid3(len/blksize, 1, 1);

  for(int fft_num = 0; fft_num < 8; fft_num++)
    {
      CUDA_ACC_SAFE_LAUNCH( (ac_reducePartial<<<grid3, block3, 3072, cudaAutocorrStream>>>( (float *)(dev_AutoCorrIn + fft_num * ac_fftlen), dev_ac_partials+fft_num*(ac_fftlen/2)/RDP, fft_num)),true); // dynamic shared size is len/RDP*sizeof(float3) -> limit 4608
    }

*/

  err = cudaMemcpyAsync(blockSums, dev_ac_partials, 8*(ac_fftlen/2)/RDP*sizeof(float3), cudaMemcpyDeviceToHost, cudaAutocorrStream);
  if(cudaSuccess != err) { fprintf(stderr, "Autocorr - memcpyAsync %d", 0); exit(0); }

  err = cudaEventRecord(autocorrelationDoneEvent[0], cudaAutocorrStream);
  if(cudaSuccess != err) { fprintf(stderr, "Autocorr done %d", 0); exit(0); }

  return 0;
}





// TODO (half done): start all autocorrs. start all datadownloads. do ALL peak finds. cudasync. process all autocorr results.

int cudaAcc_GetAutoCorrelation(float *AutoCorrelation, int ac_fftlen, int fft_num)
{
  int len = ac_fftlen/2;
  int blksize = RDP; 
  float rac_TotalSum = 0, ac_Peak = 0;
  int ac_PeakBin = 0;
  cudaError_t err;

  if(fft_num == 0)
    {
      err = cudaEventSynchronize(autocorrelationDoneEvent[0]); // host (CPU) code waits for the all (specific) GPU task to complete
      if(cudaSuccess != err) { fprintf(stderr, "GetAutocorr - sync %d", fft_num); exit(0); }
    }

  for(int b = 0; b < len/blksize; b++)
    {
      rac_TotalSum += blockSums[b + fft_num*len/blksize].x;
      if(blockSums[b + fft_num*len/blksize].y > ac_Peak)
	{
	  ac_Peak = blockSums[b + fft_num*len/blksize].y;
	  ac_PeakBin = b;
	}
    }

  blockSums[fft_num*len/blksize].x = rac_TotalSum;
  blockSums[fft_num*len/blksize].y = ac_Peak;
  blockSums[fft_num*len/blksize].z = blockSums[fft_num*len/blksize + ac_PeakBin].z;
  
  return 0;
}


#endif //USE_CUDA
