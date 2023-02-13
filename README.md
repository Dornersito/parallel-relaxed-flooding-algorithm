## Parallel Relaxed Flooding Algorithm (PRFA)

A GPU-based parallel algorithm for Voronoi diagram generation.

### Usage

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `./PRFA`

### Input & Output
Input: Randomly generated data or txt file with point data.
- When using txt file as input, set `USE_IMAGE_INPUT` to `true` and modify the values of `IMAGE_INPUT_PREFIX` and `IMAGE_FILE_NUMBER`.

Output: Set `PRINT_VORONOI` to `true` and modify the value of `IMAGE_OUTPUT_PREFIX`.

## Parameter List

In `parameter.h`,

| <div style = "width:120px"> Parameter </div> | Description |
| - | - |
| `DTYPE` | Data type of point coordinates. Only use `short` in this program. |
| `DIM` | Dimension of point data. 2D in this program. |
| `int PIC_WIDTH` | Image width. 1024, 2048, 4096, and 8192 are tested. |
| `int SITES_NUMBER` | Reference point number for random data. It should be set to 0 when using image (txt) input. |
| `E_DISTRIBUTION DISTRIBUTION` | Type of the random point distribution. Uniform, normal, clusters, or alignments. |
| `RNG_CLUSTER_COUNT` | Parameter for the random number generator. Cluster count in the clusters distribution. |
| `RNG_LINE_COUNT` | Parameter for the random number generator. Alignment (Line) count in the alignments distribution. |
| `PRFA_BLOCK_SIZE_2D_X` | Block size x of the PRFA CUDA kernel. |
| `PRFA_BLOCK_SIZE_2D_Y` | Block size y of the PRFA CUDA kernel. |
| `THREAD_DIM` | Edge length of a subregion. The `footmark` variable in the PRFA kernel is related with this parameter. |
| `K` | Relevant point count for each subregion. |
| `COMPRESS_KD_NODE` | If `true`, use 1 bit in the x- (y-) coord of a node to represent the existence of the left (right) child. |
| `TREE_H` | Height of the k-d tree. It should be set to `log2(2, SITES_NUMBER)`. |
| `MARKER` | A sentinel value. |


In `main.cpp`,

| <div style = "width:240px"> Parameter </div> | Description |
| - | - |
| `bool CHECK_FAULT` | If `true`, run brute force algorithm to verify the results of PRFA. IF `false`, only run PRFA. |
| `bool PRINT_FAULT` | If `true`, print all incorrect pixel coordinates. Only available when `CHECK_FAULT == true`. |
| `bool ITER_TEST` | If `true`, run PRFA multiple times to get an average result. Notice that the result of the first iteration is discarded. If `false`, only run PRFA one time. |
| `int ITER_COUNT` | Count of iterations when `ITER_TEST == true`. |
| `bool USE_IMAGE_INPUT` | If `true`, take a txt file as point input. Notice that `SITES_NUMBER` should be set to 0 and `PIC_WIDTH` should be set to the image's size. If `false`, use random point data. |
| `bool PRINT_VORONOI` | If `true`, print the pixel labels of the diagram into a file. |
| `string IMAGE_INPUT_PREFIX` | Input image (txt) file directory. | 
| `string IMAGE_OUTPUT_PREFIX` | Output image (txt) file directory. | 
| `string IMAGE_FILE_NUMBER` | Image (txt) file name. | 


### Random Number Generator
For randomly generated data, choose between four distributions. Assume that the point number is $N$.
- Uniform distribution.
- Normal distribution. The standard deviation is $0.2W$, where $W$ is the image width.
- Clusters distribution, cluster number = $\sqrt{N}$.
- Alignments distribution, alignment number = $\sqrt[3]{N}$.