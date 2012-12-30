#ifndef HEADERS_H_
#define HEADERS_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>

/**  COMMON  **/
typedef unsigned int uint;
typedef unsigned char uchar;


typedef struct {
	ulong sbox[4][256];
	ulong p[18];
} KeyData;

/**  BLOWFISH  **/

#define showErrAndQuit(msg) __showErrAndQuit(msg, __LINE__);

void __showErrAndQuit(const char *msg, int lineNum);

/** Same as showErrAndQuit, but doesn't print line number nor file */
void showCommunicateAndQuit(const char *msg);

void cpuInit();

void cpuPrintStats(int verbose);

ulong F(const KeyData*const keyData, ulong a);

void encryptBlock(const KeyData*const keyData, ulong *l, ulong *r);

void decryptBlock(const KeyData* const keyData, ulong *l, ulong *r);

void initBlowfish(KeyData *keyData, const uchar* const  key, const int keyLength);

void encryptFile(FILE* dataFile, FILE* output, KeyData *key, const int computeOnCuda);

void decryptdFile(FILE* dataFile, FILE* output, KeyData *key, const int computeOnCuda);

void initKeysData(KeyData **pkey);


/**  CUDA  **/

void cudaInit(const int _filePartSize, const KeyData * key, const int ePT);

void cudaPrintStats(int verbose);

void cudaEncryptBuffer(const uint * const bufferIn, uint * const bufferOut);

void cudaDecryptBuffer(const uint * const bufferIn, uint * const bufferOut);

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#endif /* HEADERS_H_ */
