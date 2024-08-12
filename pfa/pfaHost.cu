 /*
File Name: pfaHost.cu
*/

#include <cuda_runtime.h>
#include "pfa.h"
#include "kdtreeSearch.h"
#include "../parameters.h"
#include "../utils/pfaUtils.h"
// printf for debug
#include <stdio.h>

// Global Variables
#define PRFA_USING_PINNED_MEM true

template<typename T>
T pfaDiagram;       // short2*
DTYPE *tree_d;
int treeArraySize;

size_t picMemSize;      // Size (in bytes) of pic
size_t treeMemSize;    // Size (in bytes) of sites

#include "prfa2DKernel.h"

// Initialize necessary memory for 2D Voronoi Diagram computation
#define ULL unsigned long long
void pfaInitialization(const int tree_array_size, int PIC_WIDTH) {
    picMemSize = (ULL) POW_DIM(PIC_WIDTH) * (ULL)DIM * (ULL)sizeof(short);
    if (cudaMalloc((void **) &pfaDiagram<short2*>, picMemSize) != cudaSuccess)
        printf("pfaDiagram<short2*> cudaMalloc error\n");
    treeArraySize = tree_array_size;
    treeMemSize = (ULL)tree_array_size * (ULL) DIM * (ULL)sizeof(DTYPE);
    if (cudaMalloc((void **) &tree_d, treeMemSize) != cudaSuccess)
        printf("tree_d cudaMalloc error\n");
}
#undef ULL

// Deallocate all allocated memory
void pfaDeinitialization() {
    cudaFree(pfaDiagram<short2*>);
    cudaFree(tree_d);
}

// Copy input to GPU
void pfaInitialize(DTYPE *tree_h) {
    if (cudaMemcpy(tree_d, tree_h, treeMemSize, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("pfaInitialize memcpy error\n");
}

void pfa2DCompute(int PIC_WIDTH, int K, int TREE_H) {
    dim3 prfaBlockDim = dim3(PRFA_BLOCK_SIZE_2D_X/THREAD_DIM, PRFA_BLOCK_SIZE_2D_Y/THREAD_DIM);
    dim3 prfaGridDim = dim3(PIC_WIDTH/PRFA_BLOCK_SIZE_2D_X, PIC_WIDTH/PRFA_BLOCK_SIZE_2D_Y);

    const unsigned int kNN_found_shared_size = sizeof(int) * (PRFA_BLOCK_SIZE_2D_X * PRFA_BLOCK_SIZE_2D_Y / THREAD_DIM / THREAD_DIM) * K;
    const unsigned int flooding_shared_size = sizeof(short2) * (PRFA_BLOCK_SIZE_2D_X * PRFA_BLOCK_SIZE_2D_Y);
    const unsigned int max_shared_size = kNN_found_shared_size > flooding_shared_size ? kNN_found_shared_size : flooding_shared_size;
    // const unsigned int para_kdtreeSearch = TREE_H - 1 + TREE_H - 1;


    //printf("max_shared_size: %i,  kNN_found_shared_size: %i, total: %i\n", max_shared_size, kNN_found_shared_size, max_shared_size + kNN_found_shared_size);
    prfa_2D_shared_mem_opt_kernel<<<prfaGridDim, prfaBlockDim, max_shared_size + kNN_found_shared_size>>>(pfaDiagram<short2*>, tree_d, PIC_WIDTH, kNN_found_shared_size, K, TREE_H);
}


// Compute 2D Voronoi diagram, return execution time
float pfaVoronoiDiagram(short *diagram, DTYPE *tree_h, float *dur_H2D, float *dur_kernel, float *dur_D2H, int PIC_WIDTH, int K, int TREE_H, bool print_iter) {
    cudaDeviceSynchronize();    // debug
#if PRFA_USING_PINNED_MEM == true
    DTYPE *tree_pinned;
    short *diagram_pinned;
    cudaMallocHost((void**)&tree_pinned, treeMemSize);
    cudaMallocHost((void**)&diagram_pinned, picMemSize);
    for (int i = 0; i < treeMemSize / sizeof(DTYPE); ++i) {
        tree_pinned[i] = tree_h[i];
    }
    memset(diagram_pinned, 0, picMemSize);
#endif
    cudaDeviceSynchronize();    // debug

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // CPU-GPU data transfer, GPU computation, GPU-CPU data transfer
    cudaEvent_t H2D_start, H2D_stop, kernel_start, kernel_stop, D2H_start, D2H_stop;
    cudaEventCreate(&H2D_start);
    cudaEventCreate(&H2D_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&D2H_start);
    cudaEventCreate(&D2H_stop);
    cudaEventRecord(start,0);

    /* ------ execution begin ------ */
    // Init
    cudaEventRecord(H2D_start,0);
#if PRFA_USING_PINNED_MEM == true
    pfaInitialize(tree_pinned);
#else
    pfaInitialize(tree_h);
#endif
    cudaEventRecord(H2D_stop,0);
    cudaEventSynchronize(H2D_stop);

    cudaStreamSynchronize(0);
    
    // Computation
    cudaEventRecord(kernel_start,0);
    pfa2DCompute(PIC_WIDTH, K, TREE_H);
    cudaEventRecord(kernel_stop,0);
    cudaEventSynchronize(kernel_stop);

    cudaStreamSynchronize(0);

    // Copy back the result
    cudaEventRecord(D2H_start,0);
#if PRFA_USING_PINNED_MEM == false
    cudaMemcpy(diagram, pfaDiagram<short2*>, picMemSize, cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(diagram_pinned, pfaDiagram<short2*>, picMemSize, cudaMemcpyDeviceToHost);
#endif
    cudaEventRecord(D2H_stop,0);
    cudaEventSynchronize(D2H_stop);
    /* ------- execution end ------- */

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float dur;
    cudaEventElapsedTime(&dur,start,stop);

    float dur1, dur2, dur3;
    cudaEventElapsedTime(&dur1, H2D_start, H2D_stop);
    cudaEventElapsedTime(&dur2, kernel_start, kernel_stop);
    cudaEventElapsedTime(&dur3, D2H_start, D2H_stop);

    if(print_iter == true){
        printf("CUDA timer PFA execution time: %.*fms\n", 4, dur);
        printf("H2D: %.*fms, GPU computation: %.*fms, D2H:  %.*fms\n", 4, dur1, 4, dur2, 4, dur3);
    }
    *dur_H2D = dur1;
    *dur_kernel = dur2;
    *dur_D2H = dur3;

#if PRFA_USING_PINNED_MEM == true
    memcpy(diagram, diagram_pinned, picMemSize);
    cudaFree(diagram_pinned);
    cudaFree(tree_pinned);
#endif
    cudaDeviceSynchronize();

    return dur;
}

