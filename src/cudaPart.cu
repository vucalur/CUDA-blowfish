#include "headers.h"
#include <stdio.h>
#include <stdlib.h>

const int THREADS_PER_BLOCK = 10;
const int BLOCKS_COUNT = 250;
//__constant__ static int constant_elementsNumberDiv2;
//__constant__ static int constant_elementsDiv2PerBlock;
//__constant__ static int constant_elementsDiv2PerThread;

__constant__     static KeyData constant_key;

void cudaInit(const int elementsNumber, const KeyData * key) {
//	int *pelementsNumber;
//	int *pelementsDiv2PerBlock;
//	int *pelementsDiv2PerThread;
//	pelementsNumber = (int *) malloc(sizeof(int));
//	pelementsDiv2PerBlock =(int *)  malloc(sizeof(int));
//	pelementsDiv2PerThread= (int *) malloc(sizeof(int));
//	int elementsDiv2PerBlock = (elementsNumber / 2 + BLOCKS_COUNT - 1) / BLOCKS_COUNT;
//	int elementsDiv2PerThread = (elementsDiv2PerBlock + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//	*pelementsNumber = elementsNumber;
//	*pelementsDiv2PerBlock = elementsDiv2PerBlock;
//	*pelementsDiv2PerThread= elementsDiv2PerThread;

	cudaMemcpyToSymbol(constant_key, key, sizeof(KeyData));
//	cudaMemcpyToSymbol(constant_elementsNumberDiv2, &elementsNumber, sizeof(int));
//	cudaMemcpyToSymbol(constant_elementsDiv2PerThread, &elementsDiv2PerThread, sizeof(int));
//	cudaMemcpyToSymbol(constant_elementsDiv2PerBlock, &elementsDiv2PerBlock, sizeof(int));
//	cudaMemcpyToSymbol(constant_elementsNumberDiv2, pelementsNumber, sizeof(int));
//	cudaMemcpyToSymbol(constant_elementsDiv2PerThread, pelementsDiv2PerThread, sizeof(int));
//	cudaMemcpyToSymbol(constant_elementsDiv2PerBlock, pelementsDiv2PerBlock, sizeof(int));
}

__device__ ulong d_F(ulong a) {
	ulong FF = 0xFFL;
	return ((constant_key.sbox[0][(a >> 24) & FF] + constant_key.sbox[1][(a >> 16) & FF])
			^ (constant_key.sbox[2][(a >> 8) & FF])) + constant_key.sbox[3][(a) & FF];
}

__device__ void d_encryptBlock(ulong *l, ulong *r) {
	int a;
	for (int i = 0; i < 16; ++i) {
		*l = (*l) ^ (constant_key.p[i]);
		*r = d_F(*l) ^ (*r);
		a = *l;
		*l = *r;
		*r = a;
	}
	a = *l;
	*l = *r;
	*r = a;
	*r = (*r) ^ (constant_key.p[16]);
	*l = (*l) ^ (constant_key.p[17]);
}

/*
 __device__ void d_decryptBlock(ulong *l, ulong *r) {
 int a;
 *l = (*l) ^ (constant_key.p[17]);
 *r = (*r) ^ (constant_key.p[16]);
 a = *l;
 *l = *r;
 *r = a;
 for (int i = 15; i >= 0; --i) {
 a = *l;
 *l = *r;
 *r = a;
 *r = d_F(con, *l) ^ (*r);
 *l = (*l) ^ (constant_key.p[i]);
 }
 }*/

__global__ void d_encryptBuffer(uint *buffer) {
	ulong ll, lr;
//	for (int i = constant_elementsDiv2PerBlock * blockIdx.x + constant_elementsDiv2PerThread * threadIdx.x;
//			i < min(constant_elementsDiv2PerBlock * blockIdx.x + constant_elementsDiv2PerThread * (threadIdx.x + 1),
//							constant_elementsNumberDiv2); ++i) {
//		ll = (ulong) buffer[2 * i];
//		lr = (ulong) buffer[2 * i + 1];
//		d_encryptBlock(&ll, &lr);
//		buffer[2 * i] = (uint) ll;
//		buffer[2 * i + 1] = (uint) lr;
//	}

//	for (int i = FILE_PART_SIZE / 2 / BLOCKS_COUNT * blockIdx.x + (FILE_PART_SIZE / 2 + BLOCKS_COUNT - 1) / BLOCKS_COUNT / THREADS_PER_BLOCK  * threadIdx.x;
//			i < min( FILE_PART_SIZE / 2 / BLOCKS_COUNT * blockIdx.x + (FILE_PART_SIZE / 2 + BLOCKS_COUNT - 1) / BLOCKS_COUNT / THREADS_PER_BLOCK * (threadIdx.x + 1),
//					FILE_PART_SIZE /2 ); ++i) {
//		ll = (ulong) buffer[2 * i];
//		lr = (ulong) buffer[2 * i + 1];
//		d_encryptBlock(&ll, &lr);
//		buffer[2 * i] = (uint) ll;
//		buffer[2 * i + 1] = (uint) lr;
//	}

//	for (int i = 200 * blockIdx.x + 40 * threadIdx.x;
//			i < min(200 * blockIdx.x + 40 * (threadIdx.x + 1), FILE_PART_SIZE / 2); ++i) {
//		ll = (ulong) buffer[2 * i];
//		lr = (ulong) buffer[2 * i + 1];
//		d_encryptBlock(&ll, &lr);
//		buffer[2 * i] = (uint) ll;
//		buffer[2 * i + 1] = (uint) lr;
//	}

//	for (int i = blockDim.x * blockIdx.x + 1000 * threadIdx.x;
//			i
//					< min(blockDim.x * blockIdx.x + 1000 * (threadIdx.x + 1),
//							min(FILE_PART_SIZE / 2, blockDim.x * (blockIdx.x + 1))); ++i) {
//		ll = (ulong) buffer[2 * i];
//		lr = (ulong) buffer[2 * i + 1];
//		d_encryptBlock(&ll, &lr);
//		buffer[2 * i] = (uint) ll;
//		buffer[2 * i + 1] = (uint) lr;
//	}

//	for (int i = 0; i < FILE_PART_SIZE / 2; ++i) {
//		ll = (ulong) buffer[2 * i];
//		lr = (ulong) buffer[2 * i + 1];
//		d_encryptBlock(&ll, &lr);
//		buffer[2 * i] = (uint) ll;
//		buffer[2 * i + 1] = (uint) lr;
//	}
//	int eN = FILE_PART_SIZE / 2; // elementsNumber
//	int mTPB = 256; // maxThreadsPerBlock
//	// int mBC = 256; // maxBlocksCount
//	int ePT = 1000; // elementsPerThread	ADJUSTABLE
//
//
//	int tTN = (eN + ePT - 1) / ePT; // totalThreadsNeeded
//
//	int bC = (tTN + mTPB - 1) / mTPB; // CALCULATED: min possible
//
//	int tPB = (tTN + bC - 1) / bC; // CALCULATED: max possible


//	for (int i = ePT * (tPB * blockIdx.x + threadIdx.x);
//				i < ePT * (tPB * (blockIdx.x + 1) + threadIdx.x); ++i) {
//	FILE_PART_SIZE:	1000000
//	eN:	500000
//	ePT:	100
//	tTN:	5000
//	bC:	20
//	tPB:	250

	int limit = 100 * (250 * blockIdx.x + threadIdx.x + 1);
	for (int i = 100 * (250 * blockIdx.x + threadIdx.x); i < limit; ++i) {
		ll = (ulong) buffer[2 * i];
		lr = (ulong) buffer[2 * i + 1];
		d_encryptBlock(&ll, &lr);
		buffer[2 * i] = (uint) ll;
		buffer[2 * i + 1] = (uint) lr;
	}
}

void cudaEncryptBuffer(const uint * const bufferIn, uint * const bufferOut) {
	// WARNING: doesn't work this way:
	//	static int initialised = 0;
	uint * d_buffer = NULL;
//	if (initialised == 0) {
//		initialised = 1;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_buffer, sizeof(uint) * FILE_PART_SIZE));
//	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_buffer, bufferIn, sizeof(uint) * FILE_PART_SIZE, cudaMemcpyHostToDevice));

//	int threadsPerBlock = 256;
//	int elemsPerThread = 1000;
//	int blocksPerGrid = 200; // (FILE_PART_SIZE / 2 + threadsPerBlock * elemsPerThread - 1) / (threadsPerBlock * elemsPerThread);
////    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//	d_encryptBuffer<<<blocksPerGrid, threadsPerBlock>>>(d_buffer);


//	d_encryptBuffer<<<BLOCKS_COUNT, THREADS_PER_BLOCK>>>(d_buffer);
	d_encryptBuffer<<<20, 250>>>(d_buffer);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
//	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(bufferOut, d_buffer, sizeof(int) * FILE_PART_SIZE, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree((void*) d_buffer));
	// FIXME : never frees
}

/*
 // TODO : to cuda
 void cudaDecryptBuffer(const uint * const bufferIn, uint * const bufferOut) {
 ulong ll, lr;
 for (int it = 0; it < FILE_PART_SIZE; it += 2) {
 ll = (ulong) bufferIn[it];
 lr = (ulong) bufferIn[it + 1];
 decryptBlock(key, &ll, &lr);
 bufferOut[it] = (uint) ll;
 bufferOut[it + 1] = (uint) lr;
 }
 }
 */
