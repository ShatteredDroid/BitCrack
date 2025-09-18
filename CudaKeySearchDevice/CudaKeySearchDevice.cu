#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeySearchDevice.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"

__constant__ unsigned int _INC_X[8];

__constant__ unsigned int _INC_Y[8];

__constant__ unsigned int *_CHAIN[1];

__constant__ unsigned int _INC_KEY[8];

__constant__ unsigned int *_PRIVATE_KEYS[1];

__constant__ unsigned int _NIBBLE_LIMIT;

__constant__ unsigned int _ITERATION_OFFSET[8];

static unsigned int *_chainBufferPtr = NULL;


__device__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}


/**
 * Allocates device memory for storing the multiplication chain used in
 the batch inversion operation
 */
cudaError_t allocateChainBuf(unsigned int count)
{
    cudaError_t err = cudaMalloc(&_chainBufferPtr, count * sizeof(unsigned int) * 8);

    if(err) {
        return err;
    }

    err = cudaMemcpyToSymbol(_CHAIN, &_chainBufferPtr, sizeof(unsigned int *));
    if(err) {
        cudaFree(_chainBufferPtr);
    }

    return err;
}

void cleanupChainBuf()
{
    if(_chainBufferPtr != NULL) {
        cudaFree(_chainBufferPtr);
        _chainBufferPtr = NULL;
    }
}

/**
 *Sets the EC point which all points will be incremented by
 */
cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
    unsigned int xWords[8];
    unsigned int yWords[8];

    x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
    if(err) {
        return err;
    }

    return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}

cudaError_t setPrivateKeyIncrement(const secp256k1::uint256 &value)
{
    unsigned int words[8];
    value.exportWords(words, 8, secp256k1::uint256::BigEndian);

    return cudaMemcpyToSymbol(_INC_KEY, words, sizeof(unsigned int) * 8);
}

cudaError_t setIterationOffset(const secp256k1::uint256 &value)
{
    unsigned int words[8];
    value.exportWords(words, 8, secp256k1::uint256::BigEndian);

    return cudaMemcpyToSymbol(_ITERATION_OFFSET, words, sizeof(unsigned int) * 8);
}

cudaError_t setPrivateKeyBuffer(unsigned int *ptr)
{
    return cudaMemcpyToSymbol(_PRIVATE_KEYS, &ptr, sizeof(unsigned int *));
}

cudaError_t setNibbleLimit(unsigned int nibble)
{
    return cudaMemcpyToSymbol(_NIBBLE_LIMIT, &nibble, sizeof(unsigned int));
}



__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}


__device__ __forceinline__ void addUint256(unsigned int value[8], const unsigned int addend[8])
{
    add_cc(value[7], value[7], addend[7]);
    addc_cc(value[6], value[6], addend[6]);
    addc_cc(value[5], value[5], addend[5]);
    addc_cc(value[4], value[4], addend[4]);
    addc_cc(value[3], value[3], addend[3]);
    addc_cc(value[2], value[2], addend[2]);
    addc_cc(value[1], value[1], addend[1]);
    addc(value[0], value[0], addend[0]);
}

__device__ __forceinline__ bool hasNibbleSequence(const unsigned int value[8], unsigned int nibbleLength)
{
    if(nibbleLength <= 1) {
        return false;
    }

    bool first = true;
    unsigned int prev = 0;
    unsigned int count = 0;

    for(int word = 0; word < 8; word++) {
        unsigned int w = value[word];

        for(int shift = 28; shift >= 0; shift -= 4) {
            unsigned int nibble = (w >> shift) & 0x0f;

            if(first) {
                prev = nibble;
                count = 1;
                first = false;
            } else if(nibble == prev) {
                count++;
                if(count >= nibbleLength) {
                    return true;
                }
            } else {
                prev = nibble;
                count = 1;
            }
        }
    }

    return false;
}


__device__ void setResultFound(int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
    CudaDeviceResult r;

    r.block = blockIdx.x;
    r.thread = threadIdx.x;
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(&r, sizeof(r));
}

__device__ void doIteration(int pointsPerThread, int compression)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();
    unsigned int *privPtr = _PRIVATE_KEYS[0];
    unsigned int nibbleLimit = _NIBBLE_LIMIT;
    bool useNibble = (nibbleLimit > 0) && (privPtr != NULL);

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];
        bool skip = false;

        if(useNibble) {
            unsigned int candidateKey[8];
            readInt(privPtr, i, candidateKey);
            addUint256(candidateKey, _ITERATION_OFFSET);
            skip = hasNibbleSequence(candidateKey, nibbleLimit);
        }

        readInt(xPtr, i, x);

        if(!skip) {
            unsigned int digest[5];

            if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
                unsigned int y[8];
                readInt(yPtr, i, y);

                hashPublicKey(x, y, digest);

                if(checkHash(digest)) {
                    setResultFound(i, false, x, y, digest);
                }
            }

            if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {
                hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

                if(checkHash(digest)) {
                    unsigned int y[8];
                    readInt(yPtr, i, y);
                    setResultFound(i, true, x, y, digest);
                }
            }
        }

        beginBatchAdd(_INC_X, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);

    }
}

__device__ void doIterationWithDouble(int pointsPerThread, int compression)
{
    unsigned int *chain = _CHAIN[0];
    unsigned int *xPtr = ec::getXPtr();
    unsigned int *yPtr = ec::getYPtr();
    unsigned int *privPtr = _PRIVATE_KEYS[0];
    unsigned int nibbleLimit = _NIBBLE_LIMIT;
    bool useNibble = (nibbleLimit > 0) && (privPtr != NULL);

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];
        bool skip = false;

        if(useNibble) {
            unsigned int candidateKey[8];
            readInt(privPtr, i, candidateKey);
            addUint256(candidateKey, _ITERATION_OFFSET);
            skip = hasNibbleSequence(candidateKey, nibbleLimit);
        }

        readInt(xPtr, i, x);

        if(!skip) {
            unsigned int digest[5];

            // uncompressed
            if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
                unsigned int y[8];
                readInt(yPtr, i, y);
                hashPublicKey(x, y, digest);

                if(checkHash(digest)) {
                    setResultFound(i, false, x, y, digest);
                }
            }

            // compressed
            if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {

                hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

                if(checkHash(digest)) {

                    unsigned int y[8];
                    readInt(yPtr, i, y);

                    setResultFound(i, true, x, y, digest);
                }
            }
        }

        beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);

    }
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(int points, int compression)
{
    doIteration(points, compression);
}

__global__ void keyFinderKernelWithDouble(int points, int compression)
{
    doIterationWithDouble(points, compression);
}
