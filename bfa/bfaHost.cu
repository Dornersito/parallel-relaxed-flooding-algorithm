 /*
File Name: bfaHost.cu
*/

#include <cuda_runtime.h>
#include "bfa.h"
#include "../parameters.h"
#include "../utils/pfaUtils.h"
// printf for debug
#include <stdio.h>


// Global Variables
int *bfaDiagram;
template<typename T>
T bfaSites;                 // short2*
size_t bfaPicMemSize;      // Size (in bytes) of pic
size_t bfaSitesMemSize;    // Size (in bytes) of sites
int sites_num;

template <typename T>   // short2 *sites
__global__ void brute_force_2D_kernel(int *voronoi, T sites, int sites_num, int PIC_WIDTH) {
    short2 p_coord = make_short2(blockIdx.x * blockDim.x + threadIdx.x, 
        blockIdx.y * blockDim.y + threadIdx.y);
    int p_idx = p_coord.y * PIC_WIDTH + p_coord.x;
    short2 best_site = sites[0];
    int site_idx = 0;
    for (int i = 1; i < sites_num; ++i) {
        int d1 = dist2_short2(best_site, p_coord);
        int d2 = dist2_short2(sites[i], p_coord);
        if (d2 < d1) {  // the new site is closer to p
            best_site = sites[i];
            site_idx = i;
        }
    }
    voronoi[p_idx] = site_idx;
}

// Initialize necessary memory for 2D Voronoi Diagram computation
#define ULL unsigned long long
void bfaInitialization(const int sites_number, int PIC_WIDTH) {
    sites_num = sites_number;
    bfaPicMemSize = (ULL) POW_DIM(PIC_WIDTH) * (ULL)sizeof(int); 
    cudaMalloc((void **) &bfaDiagram, bfaPicMemSize);
    bfaSitesMemSize = (ULL)sites_num * (ULL) DIM * (ULL)sizeof(short);
    cudaMalloc((void **) &bfaSites<short2*>, bfaSitesMemSize);
}
#undef ULL

// Deallocate all allocated memory
void bfaDeinitialization() {
    cudaFree(bfaDiagram);
    cudaFree(bfaSites<short2*>);
}

void bfaInitialize(short *sites) {
        cudaMemcpy(bfaSites<short2*>, sites, bfaSitesMemSize, cudaMemcpyHostToDevice);
}

void bfa2DCompute(int PIC_WIDTH) {
    dim3 blockDim = dim3(32, 32);
    dim3 gridDim = dim3(PIC_WIDTH/32, PIC_WIDTH/32);
    brute_force_2D_kernel<<<gridDim, blockDim>>>(bfaDiagram, bfaSites<short2*>, sites_num, PIC_WIDTH);
}

// Compute 2D Voronoi diagram
float bfaVoronoiDiagram(int *diagram, short *sites, int PIC_WIDTH) {
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    /* ------ execution begin ------ */
    // Init
    bfaInitialize(sites);

    // Computation
    bfa2DCompute(PIC_WIDTH);
    
    // Copy back the result
    if (cudaMemcpy(diagram, bfaDiagram, bfaPicMemSize, cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("bfa result memcpy error\n");
    /* ------- execution end ------- */ 
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float dur;
    cudaEventElapsedTime(&dur,start,stop);

    printf("CUDA timer Brute Force execution time: %.*fms\n", 4, dur);
    return dur;
}
