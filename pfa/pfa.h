/*
File Name: pfa.h
*/

#include "kdtree.h"

#ifndef __PFA_CUDA_H__
#define __PFA_CUDA_H__

// Initialize CUDA and allocate memory for 2D
extern "C" void pfaInitialization(const int tree_array_size);

// Deallocate memory in GPU for 2D
extern "C" void pfaDeinitialization();

// Compute 2D Voronoi diagram
extern "C" float pfaVoronoiDiagram(short *diagram, DTYPE *tree_h, 
    float *dur_H2D, float *dur_kernel, float *dur_D2H);

#endif
