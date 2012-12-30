#include "headers.h"
#include <stdio.h>
#include <stdlib.h>

__constant__ static int c_eN;
__constant__ static int c_tPB;
__constant__ static int c_ePT;
__constant__     static KeyData constant_key;

static int filePartSize;
static int eN; // elementsNumber, element == 2 array cells
static const int mTPB = 512; // maxThreadsPerBlock
static const int mBC = 512; // maxBlocksCount

static int ePT = 500; // elementsPerThread	ADJUSTABLE

static int tTN; // totalThreadsNeeded
static int bC; // blocks Count CALCULATED: min possible
static int tPB; // threads Per Block CALCULATED: max possible

static double kernelTime;
static cudaEvent_t start, stop;


void cudaInit(const int _filePartSize, const KeyData * key, const int ePT) {
	filePartSize = _filePartSize;
	if (filePartSize % 2 != 0) {
		char communicate[200];
		sprintf(communicate,
				"Unable to execute CUDA kernel launch with:\n"
				"fPS %d\nePT %d\nbC %d\ntPB %d\n"
				"filePartSize must be divisible by 2",
				filePartSize, ePT, bC, tPB);
		showErrAndQuit(communicate);
	}
	eN = filePartSize / 2;
	tTN = (eN + ePT - 1) / ePT; // totalThreadsNeeded
	bC = (tTN + mTPB - 1) / mTPB; // blocks Count CALCULATED: min possible
	tPB = (tTN + bC - 1) / bC; // threads Per Block CALCULATED: max possible

	if (bC > mBC) {
		char communicate[200];
		sprintf(communicate, "Unable to execute CUDA kernel launch with:\n"
				"fPS %d\nePT %d\nbC %d\ntPB %d\n"
				"Total threads needed exceeds maximum available (512^2)",
				filePartSize, ePT, bC, tPB);
		showErrAndQuit(communicate);
	}

	cudaMemcpyToSymbol(constant_key, key, sizeof(KeyData));
	cudaMemcpyToSymbol(c_eN, &eN, sizeof(int));
	cudaMemcpyToSymbol(c_tPB, &tPB, sizeof(int));
	cudaMemcpyToSymbol(c_ePT, &ePT, sizeof(int));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	kernelTime = 0;	// explicitly
}

void cudaPrintStats(int verbose) {
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("%s%f ms\n", (verbose) ? "total CUDA kernel time:\t" : "", kernelTime);
	if (verbose) {
		printf("CUDA kernel launches parameters:\n"
				"eN\t%d\n"
				"ePT\t%d\n"
				"tTN\t%d\n"
				"bC\t%d\n"
				"tPB\t%d\n",
				eN, ePT, tTN, bC, tPB);
	}
}

__device__  __inline__ ulong d_F(ulong a) {
	//	ulong FF = 0xFFL; // introduced as hardcoded constant
	return ((constant_key.sbox[0][(a >> 24) & 0xFFL] + constant_key.sbox[1][(a >> 16) & 0xFFL])
			^ (constant_key.sbox[2][(a >> 8) & 0xFFL])) + constant_key.sbox[3][(a) & 0xFFL];
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
		*r = d_F(*l) ^ (*r);
		*l = (*l) ^ (constant_key.p[i]);
	}
}

__global__ void d_encryptBuffer(uint *buffer) {
	ulong ll, lr;

	//	const int limit = c_ePT * (c_tPB * blockIdx.x + threadIdx.x + 1);
	for (int i = c_ePT * (c_tPB * blockIdx.x + threadIdx.x);
			i < min(c_ePT * (c_tPB * blockIdx.x + threadIdx.x + 1), c_eN); ++i) {
		ll = (ulong) buffer[2 * i];
		lr = (ulong) buffer[2 * i + 1];
		d_encryptBlock(&ll, &lr);
		buffer[2 * i] = (uint) ll;
		buffer[2 * i + 1] = (uint) lr;
	}
}

__global__ void d_decryptBuffer(uint *buffer) {
	ulong ll, lr;

	//	const int limit = min(c_ePT * (c_tPB * blockIdx.x + threadIdx.x + 1), c_eN);
	for (int i = c_ePT * (c_tPB * blockIdx.x + threadIdx.x);
			i < min(c_ePT * (c_tPB * blockIdx.x + threadIdx.x + 1), c_eN); ++i) {
		ll = (ulong) buffer[2 * i];
		lr = (ulong) buffer[2 * i + 1];
		d_decryptBlock(&ll, &lr);
		buffer[2 * i] = (uint) ll;
		buffer[2 * i + 1] = (uint) lr;
	}
}

void cudaEncryptBuffer(const uint * const bufferIn, uint * const bufferOut) {
	static uint * d_buffer = NULL;
	if (d_buffer == NULL) {
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_buffer, sizeof(uint) * filePartSize));
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_buffer, bufferIn, sizeof(uint) * filePartSize, cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);
	d_encryptBuffer<<<bC, tPB>>>(d_buffer);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	kernelTime += (double) time;

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(bufferOut, d_buffer, sizeof(int) * filePartSize, cudaMemcpyDeviceToHost));

	// CUDA_CHECK_RETURN(cudaFree((void*) d_buffer)); // FIXME : never frees
}

void cudaDecryptBuffer(const uint * const bufferIn, uint * const bufferOut) {
	static uint * d_buffer = NULL;
	if (d_buffer == NULL) {
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_buffer, sizeof(uint) * filePartSize));
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_buffer, bufferIn, sizeof(uint) * filePartSize, cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);
	d_decryptBuffer<<<bC, tPB>>>(d_buffer);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	kernelTime += (double) time;

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(bufferOut, d_buffer, sizeof(int) * filePartSize, cudaMemcpyDeviceToHost));

	// CUDA_CHECK_RETURN(cudaFree((void*) d_buffer)); // FIXME : never frees
}
