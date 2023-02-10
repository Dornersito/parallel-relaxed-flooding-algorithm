/*
File Name: prfa2DKernel.h
*/

#include "../parameters.h"
#include "../utils/pfaUtils.h"
#include "kdtreeSearch.h"

// Parallel Relaxed Flooding Algorithm
// Compute found[0], then found[1], found[2] sqeuentially to decrease the size of claimed_queue.
// Use float for target pixel of each subregion.
// Storing s_found_idx instead of s_found (using coord) to reduce the shared memory size
__global__ void prfa_2D_shared_mem_opt_kernel(short2 *voronoi, DTYPE *tree) {
	const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // thread id in 1 dimension
	
    // x_min_large, for the large (entire) voronoi diagram
    const short x_min_l = PRFA_BLOCK_SIZE_2D_X * blockIdx.x + THREAD_DIM * threadIdx.x;
    const short y_min_l = PRFA_BLOCK_SIZE_2D_Y * blockIdx.y + THREAD_DIM * threadIdx.y;
	
    // find k nearest sites, target is float type
    float target[DIM];   			// center of this thread subregion
    target[0] = x_min_l + THREAD_DIM / 2 - 0.5;
    target[1] = y_min_l + THREAD_DIM / 2 - 0.5;

	// notice: use FD_IDX to access the elements

	// int and float are both 4B, found array and dst array have the same size
	const unsigned int kNN_found_shared_size = sizeof(int) * (PRFA_BLOCK_SIZE_2D_X * PRFA_BLOCK_SIZE_2D_Y / THREAD_DIM / THREAD_DIM) * K;
	// each thread needs THREAD_DIM * THREAD_DIM
	const unsigned int flooding_shared_size = sizeof(short2) * (PRFA_BLOCK_SIZE_2D_X * PRFA_BLOCK_SIZE_2D_Y);

	const unsigned int max_shared_size = kNN_found_shared_size > flooding_shared_size ? kNN_found_shared_size : flooding_shared_size;
	__shared__ char smem[max_shared_size + kNN_found_shared_size];

	int *s_found_idx = reinterpret_cast<int*>(smem);
	float *s_dst_k = reinterpret_cast<float*>(smem + kNN_found_shared_size);

#if COMPRESS_KD_NODE == true
	k_nearest_found_idx_two_stacks_compressed_node_2D<DTYPE, float, float>(tree, target, s_found_idx + tid * K, s_dst_k + tid * K);
#else
	k_nearest_found_idx_two_stacks_2D<DTYPE, float, float>(tree, target, s_found_idx + tid * K, s_dst_k + tid * K);
#endif	
	__syncthreads();

	// init voronoi with found[0]
	for (int ty = y_min_l; ty < y_min_l + THREAD_DIM; ++ty)
		for (int tx = x_min_l; tx < x_min_l + THREAD_DIM; ++tx) {
#if COMPRESS_KD_NODE == true
			short sx = read_node_kdtree(tree[s_found_idx[tid * K + 0] * DIM + 0]), sy = read_node_kdtree(tree[s_found_idx[tid * K + 0] * DIM + 1]);
#else
			short sx = tree[s_found_idx[tid * K + 0] * DIM + 0] , sy = tree[s_found_idx[tid * K + 0] * DIM + 1];
#endif
			voronoi[ty * PIC_WIDTH + tx] = make_short2(sx, sy);
		}
	
	__syncthreads();
    
    /* ------ relaxed flooding computation  ------ */
	short2 *s_claimed_queue = reinterpret_cast<short2*>(smem + kNN_found_shared_size);
	short2 *claimed_queue = s_claimed_queue + tid * THREAD_DIM * THREAD_DIM;

    for (int f_idx = 1; f_idx < K; ++f_idx) {		// found site index
		// maximum THREAD_DIM * THREAD_DIM pixels in queue
    	// q_p1 = pointer to the front of queue, q_p2 = pointer to the (back + 1) of queue
		short q_p1 = 0, q_p2 = 1;	// reset

#if COMPRESS_KD_NODE == true
		short sx = read_node_kdtree(tree[s_found_idx[tid * K + f_idx] * DIM + 0]), sy = read_node_kdtree(tree[s_found_idx[tid * K + f_idx] * DIM + 1]);
#else
		short sx = tree[s_found_idx[tid * K + f_idx] * DIM + 0], sy = tree[s_found_idx[tid * K + f_idx] * DIM + 1];
#endif
		short tx = clamp(sx, x_min_l, (short)(x_min_l + THREAD_DIM - 1));
		short ty = clamp(sy, y_min_l, (short)(y_min_l + THREAD_DIM - 1));
		claimed_queue[0] = make_short2(tx, ty);
		// footmark of all 16 pixels in a subregion, accessed pixel (bit) will be set to 1
		short footmark = 0 | (1 << (ty - y_min_l) * THREAD_DIM + (tx - x_min_l));

		while (q_p1 != q_p2) {
			short tx = claimed_queue[q_p1].x, ty = claimed_queue[q_p1++].y;		// pop
			short2 prev_site = voronoi[ty * PIC_WIDTH + tx];

			// 1. if claimed_queue[q_p1].site is closer to voronoi[p_idx], mark this pixel with current site s
			// 2. foreach neighboring pixel nbr_pixel.
			//      1. if dst(s, nbr_pixel) < dst[p_idx]: mark nbr_pixel with s
			//      2. if dst(s, nbr_pixel) >= dst[p_idx]: compare the gradient of 4 neighbors and push to claimed_queue

			// compare the distance with prev_site
			int dst = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy);
			bool claim_new = dst < (tx - prev_site.x) * (tx - prev_site.x) + (ty - prev_site.y) * (ty - prev_site.y);
			if (claim_new)
				voronoi[ty * PIC_WIDTH + tx] = make_short2(sx, sy);

			for (int dir = 0; dir < 4; ++dir) {
				short tx_nbr = tx, ty_nbr = ty;
				bool grad_passed = false;		// gradient check passed

				if (dir == 0) {			// right	
					if (++tx_nbr > x_min_l + THREAD_DIM - 1)
						continue;
					grad_passed = prev_site.x < sx;
				} else if (dir == 1) {	// left
					if (--tx_nbr < x_min_l)
						continue;
					grad_passed = prev_site.x > sx;
				} else if (dir == 2) {	// up
					if (++ty_nbr > y_min_l + THREAD_DIM - 1)
						continue;
					grad_passed = prev_site.y < sy;
				} else if (dir == 3) {	// down
					if (--ty_nbr < y_min_l)
						continue;
					grad_passed = prev_site.y > sy;
				}

				if (!claim_new && !grad_passed)
					continue;
				short fm_bit_idx = 1 << ((ty_nbr - y_min_l) * THREAD_DIM + (tx_nbr - x_min_l));		// footmark bit
				if ((footmark & fm_bit_idx) == 0) {	// not accessed before
					claimed_queue[q_p2++] = make_short2(tx_nbr, ty_nbr);
					footmark |= fm_bit_idx;
				}
			}
		}
    }
}
