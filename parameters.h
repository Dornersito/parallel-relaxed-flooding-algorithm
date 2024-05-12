/*
File Name: parameters.h
*/

#ifndef parameters_h
#define parameters_h

#define DTYPE short     // point data type
#define DIM 2           // 2D tasks
// const static int PIC_WIDTH = 2048;

// if points are not generated randomly (but from an image), set SITES_NUMBER to 0
// static int SITES_NUMBER = 0;

// 2D 0.01%: 256-7, 512-26, 1024-105, 2048-419, 4096-1678, 8192-6711, 16384-26844
// 2D 0.1%: 256-66, 512-262, 1024-1049, 2048-4194, 4096-16777, 8192-67109, 16384-268435
// 2D 1%: 256-655, 512-2621, 1024-10486, 2048-41943, 4096-167772, 8192-671089, 16384-2684355
// static int SITES_NUMBER = 2684355;

// type of distributions of randomly generated datasets
// enum class E_DISTRIBUTION {
//     uniform,
//     normal,
//     clusters,
//     alignments,
// };
// const static E_DISTRIBUTION DISTRIBUTION = E_DISTRIBUTION::uniform;
#define RNG_CLUSTER_COUNT 20
#define RNG_LINE_COUNT 4
// sqrt(41943) = 205, sqrt(4194) = 65, sqrt(419) = 20
// pow(41943, 1/3) = 35, pow(4194, 1/3) = 17, pow(419, 1/3) = 7 

// must be power of 2
#define PRFA_BLOCK_SIZE_2D_X 32
#define PRFA_BLOCK_SIZE_2D_Y 32

#define THREAD_DIM  4                   // this affects footmark in PRFA's kernel

// K nearest points for k-d tree
#define K 5
#define COMPRESS_KD_NODE true
#define TREE_H 9 // log2(2, SITES_NUMBER), for k-d tree
// 15: 4
// 2D 0.01%  105: 7, 419: 9, 1678: 11, 6711: 13, 26844: 15
// 2D 0.1%  1049: 11, 4194: 13, 16777: 15, 67109: 17, 268435: 19
// 2D 1%    10486: 14, 41943: 16, 167772: 18, 671089: 20, 2684355: 22

#define MARKER -32768

#endif /* parameters_h */