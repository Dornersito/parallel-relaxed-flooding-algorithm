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
- When using txt file as input, set `USE_IMAGE_INPUT` to true and modify the values of `IMAGE_INPUT_PREFIX` and `IMAGE_FILE_NUMBER`.

Output: Set `PRINT_VORONOI` to true and modify the value of `IMAGE_OUTPUT_PREFIX`.

### Random Number Generator
For randomly generated data, choose between four distributions. Assume that the point number is $N$.
- Uniform distribution.
- Normal distribution. The standard deviation is $0.2W$, where $W$ is the image width.
- Clusters distribution, cluster number = $\sqrt{N}$.
- Alignments distribution, alignment number = $\sqrt[3]{N}$.