#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>
#include "cudaUtil.h"
#include "secp256k1.h"


void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression);

void waitForKernel();

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);
cudaError_t setPrivateKeyIncrement(const secp256k1::uint256 &value);
cudaError_t setPrivateKeyBuffer(unsigned int *ptr);
cudaError_t setNibbleLimit(unsigned int nibble);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();
bool runNibbleSequenceDiagnostics(unsigned int nibbleLength);

#endif