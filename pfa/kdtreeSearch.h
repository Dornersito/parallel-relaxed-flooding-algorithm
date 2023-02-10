/*
File Name: kdtreeSearch.h
*/

#include <stdio.h>
#include "../parameters.h"
#include "../utils/pfaUtils.h"
#include "float.h"

#ifndef kdtreeSearch_h
#define kdtreeSearch_h

// this header is used in device

// only use 2 stacks instead of 3 stacks
// node_stack, dx2_stack: remain unchanged, i_stack: stored in an int
// T_p stands for type of ref point coordinates, T_t stands for type of target (query) point coordinates
// T_d stands for type of distance, should be float or double to prevent overflow
template<typename T_p, typename T_t, typename T_d>
__device__ void k_nearest_found_idx_two_stacks_2D(const T_p *tree, const T_t *target, int *found_idx, T_d *dst_k) {
    int node_stack[TREE_H - 1];
    T_d dx2_stack[TREE_H - 1];
    int i_stack = 0;    // only for 2D tree. inititalized: 1 << 0 = 0, if a bit == 0, axis is 0, else axis is 1
    int s1 = 0;         // s1 = top 

    int found_count = 0;
    dst_k[K - 1] = FLT_MAX;     // replace d2_best_k: best k-th dist
    
    int cur_node_idx = 1;
    short cur_i = 0;    // axis direction
    
    while (cur_node_idx > 0) {
        T_d d2, dx;
        d2 = (target[0] - tree[cur_node_idx * DIM + 0]) * (target[0] - tree[cur_node_idx * DIM + 0]) +
            (target[1] - tree[cur_node_idx * DIM + 1]) * (target[1] - tree[cur_node_idx * DIM + 1]);
        dx = tree[cur_node_idx * DIM + cur_i] - target[cur_i];
        
        if (d2 < dst_k[K - 1]) {
            int j = found_count == K ? K - 1 : found_count;
            for (; j > 0 && dst_k[j - 1] > d2; --j) {
                found_idx[j] = found_idx[j - 1];
                dst_k[j] = dst_k[j - 1];
            }
            found_idx[j] = cur_node_idx;
            dst_k[j] = d2;
            if (found_count < K)
                ++found_count;
        }

        if (cur_node_idx * 2 < (1 << TREE_H)) {
            // push one node into stack, update cur_node_idx with the other one
            cur_i = 1 - cur_i;
            if (tree[(cur_node_idx * 2 + 1 * (dx > 0)) * DIM + 0] != -1) {  // if dx > 0, cur_node * 2 + 1, else cur_node * 2
                cur_i == 1 ? i_stack |= 1 << s1 : i_stack &= ~(1 << s1);    // set s1-th bit to cur_i
                node_stack[s1] = cur_node_idx * 2 + 1 * (dx > 0);
                dx2_stack[s1++] = dx * dx;
            }
            cur_node_idx = cur_node_idx * 2 + 1 * (dx <= 0);    // if dx > 0, cur_node * 2, else cur_node * 2 + 1
            if (tree[cur_node_idx * DIM + 0] != -1)
                continue;   // directly go to next iter
        }

        cur_node_idx = 0;
        // read from the stack (cur_node_idx and cur_i)
        while (s1 > 0) {
            if (dx2_stack[--s1] < dst_k[K - 1]) {
                // Note: Eliminate dx2_stack does not reduce the access to tree array
                cur_node_idx = node_stack[s1];
                cur_i = (i_stack >> s1) & 1;
                break;
            }
        }
    }
}

// check the 14th bit of x and y crd for the existence of left and right child
// only use 2 stacks instead of 3 stacks
// node_stack, dx2_stack: remain unchanged, i_stack: stored in an int
// T_p stands for type of ref point coordinates, T_t stands for type of target (query) point coordinates
// T_d stands for type of distance, should be float or double to prevent overflow
template<typename T_p, typename T_t, typename T_d>
__device__ void k_nearest_found_idx_two_stacks_compressed_node_2D(const T_p *tree, const T_t *target, int *found_idx, T_d *dst_k) {
    int node_stack[TREE_H - 1];
    T_d dx2_stack[TREE_H - 1];
    int i_stack = 0;    // only for 2D tree. inititalized: 1 << 0 = 0, if a bit == 0, axis is 0, else axis is 1
    int s1 = 0;         // s1 = top 

    int found_count = 0;
    dst_k[K - 1] = FLT_MAX;     // replace d2_best_k: best k-th dist
    
    int cur_node_idx = 1;
    short cur_i = 0;    // axis direction
    
    while (cur_node_idx > 0) {
        T_d d2, dx;
        // 00: no right nor left, 01: only left, 10: only right. 11: right and left
        int has_children = (check_child_kdtree(tree[cur_node_idx * DIM + 1]) << 1) + 
            check_child_kdtree(tree[cur_node_idx * DIM + 0]);
        d2 = POW_DIM(target[0] - read_node_kdtree(tree[cur_node_idx * DIM + 0])) + 
            POW_DIM(target[1] - read_node_kdtree(tree[cur_node_idx * DIM + 1]));
        dx = read_node_kdtree(tree[cur_node_idx * DIM + cur_i]) - target[cur_i];
        
        if (d2 < dst_k[K - 1]) {
            int j = found_count == K ? K - 1 : found_count;
            for (; j > 0 && dst_k[j - 1] > d2; --j) {
                found_idx[j] = found_idx[j - 1];
                dst_k[j] = dst_k[j - 1];
            }
            found_idx[j] = cur_node_idx;
            dst_k[j] = d2;
            if (found_count < K)
                ++found_count;
        }

        if (cur_node_idx * 2 < (1 << TREE_H)) {
            // push one node into stack, update cur_node_idx with the other one
            cur_i = 1 - cur_i;
            
            if (has_children >> (1 * (dx > 0))) {  // if dx > 0, check right, else check left
                cur_i == 1 ? i_stack |= 1 << s1 : i_stack &= ~(1 << s1);    // set s1-th bit to cur_i
                node_stack[s1] = cur_node_idx * 2 + 1 * (dx > 0);
                dx2_stack[s1++] = dx * dx;
            }
            cur_node_idx = cur_node_idx * 2 + 1 * (dx <= 0);    // if dx > 0, go to left (cur_node * 2), else right (cur_node * 2 + 1)
            if (has_children >> (1 * (dx <= 0))) {
                continue;   // directly go to next iter
            }
        }

        cur_node_idx = 0;
        // read from the stack (cur_node_idx and cur_i)
        while (s1 > 0) {
            if (dx2_stack[--s1] < dst_k[K - 1]) {
                cur_node_idx = node_stack[s1];
                cur_i = (i_stack >> s1) & 1;
                break;
            }
        }
    }
}

#endif /* kdtreeSearch_h */