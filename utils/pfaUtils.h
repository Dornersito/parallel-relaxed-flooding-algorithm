/*
File Name: pfaUtils.h
*/

#include "../parameters.h"
#include <cuda_runtime.h>

#ifndef pfaUtils_h
#define pfaUtils_h

#define POW_DIM(x) (x)*(x)   // DIM is 2
#define TOID2(x, y, w) (y*w+x)

// input point idx i and coord dimension d to read the value
#define FD_IDX(i, d) d*K+i
// shared memory bank conflicts
#define s_bias(n) (n+n/32)

// set n-th bit to 1
#define set_bit(x,n) x|=(1<<n)
// set n-th bit to 0
#define clear_bit(x,n) x&=~(1<<n)
// assign n-th bit to b
#define assign_bit(x,n,b) x=(x&~(1<<n))|(b<<n)


// read a short value, but mask out the 14th bit
// -16384 ~ 16383
#define read_node_kdtree(x) (x&~(1<<14))
// read the 14th bit of a short value
#define check_child_kdtree(x) ((x>>14)&1)
// assign the 14th bit to 1 if has child, only for 2D problems (left and right tree)
#define assign_child_kdtree(x,b) assign_bit(x,14,b)


__forceinline__ __device__ int dist2_short2(short2 p1, short2 p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

__forceinline__ __device__ float dist2_float2_short2(float2 p1, short2 p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

/*
 * @brief if x < min, return min; if x > max, return max
 * @param x     input
 * @param min   threshold 1
 * @param max   threshold 2
 * @return 
 */
__forceinline__ __device__ short clamp(short x, short min, short max) {
    return x < min ? min : max < x ? max : x;
}

/*
 * @brief if x < min, return min; if x > max, return max
 * @param x     input
 * @param min   threshold 1
 * @param max   threshold 2
 * @return 
 */
__forceinline__ __device__ int clamp(int x, int min, int max) {
    return x < min ? min : max < x ? max : x;
}


#endif /* pfaUtils_h */