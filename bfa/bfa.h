/*
File Name: bfa.h
*/

#ifndef __BF_CUDA_H__
#define __BF_CUDA_H__

// Initialize CUDA and allocate memory for 2D
extern "C" void bfaInitialization(const int sites_num);

// Deallocate memory in GPU for 2D
extern "C" void bfaDeinitialization(); 

// Compute 2D Voronoi diagram
extern "C" float bfaVoronoiDiagram(int *diagram, short *sites);

#endif
