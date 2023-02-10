/*
File Name: kdtree.h
*/

#include "../parameters.h"

#ifndef kdtree_h
#define kdtree_h

struct kd_node_t{
    int x[DIM];
    struct kd_node_t *left, *right;
};

#endif /* kdtree_h */