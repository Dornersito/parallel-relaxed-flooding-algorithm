/*
File Name: main.cpp
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>  // setprecision in fstream

#include "parameters.h"
#include "utils/pfaUtils.h"
#include "utils/numberGenerator.hpp"

#include "pfa/pfa.h"
#include "pfa/kdtreeMake.h"

#include "bfa/bfa.h"

enum class E_DISTRIBUTION {
    uniform,
    normal,
    clusters,
    alignments,
};


static const bool CHECK_FAULT = false;       // check incorrect results of pixel labels
static const bool PRINT_FAULT = false;       // print all incorrect pixel crds
static const bool ITER_TEST = true;
static const int ITER_COUNT = 5 + 1;       // skip the 1st iter for average time
static const bool USE_IMAGE_INPUT = false;  // if true, change the SITES_NUMBER to 0 and set PIC_WIDTH to the image's size
static const bool PRINT_VORONOI = false;

// test data file, nuclei_frame0XX.txt
static const std::string IMAGE_INPUT_PREFIX = "../images/nuclei_EDF0";
static const std::string IMAGE_OUTPUT_PREFIX = "../images/diagram_output/diagram_EDF0";
static const std::string IMAGE_FILE_NUMBER = "07.txt";

float dur_global_prfa[ITER_COUNT], dur_H2D_global_prfa[ITER_COUNT], dur_kernel_global_prfa[ITER_COUNT], dur_D2H_global_prfa[ITER_COUNT];
int error_count_global_prfa[ITER_COUNT];

int dur_idx = 0;

// instantiated in runTest
RNG_virtual *RNG_generator_p;

// for point p_i, inputPoints[DIM*i] = p_i.x, inputPoints[DIM*i+1] = p_i.y
short *inputPoints;
struct kd_node_t *kdtree_p, *root;
DTYPE *kdtree;
int kdtree_array_size;  // = next_pow_of_2(SITE_NUM) for normal kdtree

// if a site in the pixel, inputVoronoi[DIM*(p_i.y * width + p_i.x)] = p_i.x, 
// inputVoronoi[DIM*(p_i.y * width + p_i.x)+1] = p_i.y.
// Otherwise inputVoronoi[y*width+x] = inputVoronoi[y*width+x+1] = MARKER
short *pfaOutputVoronoi;
short *inputVoronoi;

int *voronoi_int;   // convert from cuda output, is used for Delaunay
int *brute_voronoi_int;


int next_pow_of_2(int num) {
    int e = 1;
    bool all_zero = true;
    for (int n = num; n > 1; n>>=1) {
        if (n & 0x01) all_zero = false;
        ++e;
    }
    return all_zero ? num : (1 << e);
}

// Generate input points
void generateRandomPoints(int width, int nPoints, E_DISTRIBUTION DISTRIBUTION) {
    int tx, ty, tz;
    if (ITER_TEST) {
        RNG_generator_p->rand_init(dur_idx);
    } else {
        RNG_generator_p->rand_init(100);
    }

    for (int i = 0; i < POW_DIM(width) * DIM; ++i) {
        inputVoronoi[i] = MARKER;
        pfaOutputVoronoi[i] = -1;
    }

    for (int i = 0; i < nPoints; i++) {
        do {
            if (DISTRIBUTION == E_DISTRIBUTION::clusters) {
                double rng_output[2];
                RNG_generator_p->random_one_dim(rng_output);
                tx = int(rng_output[0] * width); ty = int(rng_output[1] * width);
            } else if (DISTRIBUTION == E_DISTRIBUTION::alignments) {
                double rng_output[2];
                RNG_generator_p->random_one_dim(rng_output);
                tx = int(rng_output[0] * width); ty = int(rng_output[1] * width);
            } else {
                tx = int(RNG_generator_p->random_one() * width); ty = int(RNG_generator_p->random_one() * width);
            }
        } while (inputVoronoi[(ty * width + tx) * 2] != MARKER); 

        inputVoronoi[(ty * width + tx) * 2] = tx;
        inputVoronoi[(ty * width + tx) * 2 + 1] = ty;
        inputPoints[i * 2] = tx;
        inputPoints[i * 2 + 1] = ty;
    }
}

// read point data from an txt file
bool readInputFile(int PIC_WIDTH) {
    // read from txt
    int SITES_NUMBER = 0;
    memset(inputPoints, 0, sizeof(inputPoints));
    std::ifstream ifs;
    std::string line;
    if (ITER_TEST == true) {
        std::string dir_path_front = IMAGE_INPUT_PREFIX;
        dir_path_front += std::to_string((dur_idx / 10));
        dir_path_front += std::to_string((dur_idx % 10));
        ifs.open(dir_path_front + ".txt", std::fstream::in);
    } else 
        ifs.open(IMAGE_INPUT_PREFIX + IMAGE_FILE_NUMBER, std::fstream::in);

    if (!ifs.is_open()) {
        printf("Failed to open the file!\n");
        return false;
    }
    getline(ifs, line);      // the first line is a comment
    while(getline(ifs, line)) {
        int space_idx = line.find(' ');
        int x_crd = stoi(line.substr(0, space_idx)), y_crd = stoi(line.substr(space_idx + 1, line.length() - 1));
        inputPoints[SITES_NUMBER * 2] = (short)x_crd;
        inputPoints[SITES_NUMBER * 2 + 1] = (short)y_crd;
        SITES_NUMBER += 1;
    }

    for (int j = 0; j < PIC_WIDTH; ++j) {
        for (int i = 0; i < PIC_WIDTH; ++i) {
            pfaOutputVoronoi[(j * PIC_WIDTH + i) * 2] = -1;
            pfaOutputVoronoi[(j * PIC_WIDTH + i) * 2 + 1] = -1;
            inputVoronoi[(j * PIC_WIDTH + i) * 2] = MARKER;
            inputVoronoi[(j * PIC_WIDTH + i) * 2 + 1] = MARKER;
        }
    }

    for (int i = 0; i < SITES_NUMBER; ++i) {
        int x_crd = inputPoints[i * 2], y_crd = inputPoints[i * 2 + 1];
        inputVoronoi[(y_crd * PIC_WIDTH + x_crd) * 2] = x_crd;
        inputVoronoi[(y_crd * PIC_WIDTH + x_crd) * 2 + 1] = y_crd;
    }

    ifs.close();
    return true;
}

// Deinitialization
// memory allocated in GPU are freed in runTest()
void deinitialization() {
    free(inputPoints);
    free(inputVoronoi);
    free(pfaOutputVoronoi);
    if (CHECK_FAULT)
        free(brute_voronoi_int);
    if (PRINT_VORONOI)
        free(voronoi_int);
}


// Init    
#define ULL unsigned long long
void initialization(int PIC_WIDTH, int SITES_NUMBER) {
    if (USE_IMAGE_INPUT == true)
        inputPoints     = (short *) malloc((ULL)POW_DIM(PIC_WIDTH) * (ULL)DIM * (ULL)sizeof(short));
    else
        inputPoints     = (short *) malloc((ULL)SITES_NUMBER * (ULL)DIM * (ULL)sizeof(short));
    
    inputVoronoi    = (short *) malloc((ULL) POW_DIM(PIC_WIDTH) * (ULL)DIM * (ULL)sizeof(short)); 
    pfaOutputVoronoi   = (short *) malloc((ULL) POW_DIM(PIC_WIDTH) * (ULL)DIM * (ULL)sizeof(short));
    if (CHECK_FAULT)
        brute_voronoi_int = (int *) malloc((ULL) POW_DIM(PIC_WIDTH) * (ULL)sizeof(int));  // computed by brute force
    if (PRINT_VORONOI)
        voronoi_int = (int *) malloc((ULL) POW_DIM(PIC_WIDTH) * (ULL)sizeof(int));
}

// Init points
void initPoints(int PIC_WIDTH, int SITES_NUMBER) {
    kdtree_array_size = next_pow_of_2(SITES_NUMBER);
    pfaInitialization(kdtree_array_size, PIC_WIDTH);
    if (CHECK_FAULT)
        bfaInitialization(SITES_NUMBER, PIC_WIDTH);
    
    kdtree_p          = (struct kd_node_t *) malloc((ULL)SITES_NUMBER * (ULL)sizeof(struct kd_node_t)); 
    kdtree            = (DTYPE *) malloc((ULL)kdtree_array_size * (ULL)DIM * (ULL)sizeof(DTYPE));
}
#undef ULL

// Verify the output Voronoi Diagram
void verifyResult(int PIC_WIDTH) {
    int errorCount = 0;
    int m_tx, m_ty, b_site_idx, b_tx, b_ty;
    double dist, myDist, correct_dist;

    for (int j = 0; j < PIC_WIDTH; j++) {
        for (int i = 0; i < PIC_WIDTH; i++) {
            // coord(i, j) to id (index)
            int id = j * PIC_WIDTH + i;

            m_tx = pfaOutputVoronoi[id * 2];
            m_ty = pfaOutputVoronoi[id * 2 + 1];
            b_site_idx = brute_voronoi_int[id];
            b_tx = inputPoints[b_site_idx * 2];
            b_ty = inputPoints[b_site_idx * 2 + 1];
            myDist = (m_tx-i) * (m_tx-i) + (m_ty-j) * (m_ty-j);
            correct_dist = (b_tx-i) * (b_tx-i) + (b_ty-j) * (b_ty-j);

            if (correct_dist == myDist && (m_tx != b_tx || m_ty != b_ty)) {
                pfaOutputVoronoi[id * 2] = b_tx;
                pfaOutputVoronoi[id * 2 + 1] = b_ty;
            }

            if (correct_dist != myDist) {
                if (PRINT_FAULT) {
                    printf("Fault coord: (%d, %d)\n", i, j);
                    printf("    Correct site num and coord: %d, (%d, %d)\n", b_site_idx, b_tx, b_ty);
                    printf("    Wrong site coord: (%d, %d)\n", m_tx, m_ty);
                }
                ++errorCount;
            }
        }
    }
    if (ITER_TEST)
        error_count_global_prfa[dur_idx] = errorCount;
}

void convert_output_diagram(int *voronoi, std::string pic_name, int PIC_WIDTH, int SITES_NUMBER) {
    printf("Converting pixel labels\n");
    for (int j = 0; j < PIC_WIDTH; ++j) {
        for (int i = 0; i < PIC_WIDTH; ++i) {
            int sx = pfaOutputVoronoi[(j * PIC_WIDTH + i) * 2];
            int sy = pfaOutputVoronoi[(j * PIC_WIDTH + i) * 2 + 1];
            for (int s_idx = 0; s_idx < SITES_NUMBER; ++s_idx) {
                if (inputPoints[s_idx * 2] == sx && inputPoints[s_idx * 2 + 1] == sy) {
                    voronoi[j * PIC_WIDTH + i] = s_idx;
                }
            }
        }
        if (j % 100 == 0)
            printf("line %d complete\n", j);
    }

    std::fstream fs;
    // each pixel is the point label
    fs.open(pic_name, std::fstream::out);
    fs << PIC_WIDTH << std::endl;
    fs << SITES_NUMBER << std::endl;
    for (int y = 0; y < PIC_WIDTH; ++y) {
        for (int x = 0; x < PIC_WIDTH; ++x) {
            fs << voronoi[y * PIC_WIDTH + x] << ' ';
        }
        fs << std::endl;
    }
    fs.close();
    std::cout << pic_name << " output complete" << std::endl;
}


// Run the tests
void runTests(int PIC_WIDTH, int SITES_NUMBER, E_DISTRIBUTION DISTRIBUTION, int K) {
    // RNG instances, uniform = 0, normal = 1, clusters = 2, alignments = 3
    if (DISTRIBUTION == E_DISTRIBUTION::uniform) {
        printf("uniform distribution\n");
        RNG_generator_p = new KISS_RNG_uniform();
    } else if (DISTRIBUTION == E_DISTRIBUTION::normal) {
        printf("normal distribution\n");
        RNG_generator_p = new RNG_normal();
    } else if (DISTRIBUTION == E_DISTRIBUTION::clusters) {
        printf("clusters distribution\n");
        RNG_generator_p = new RNG_clusters();
    } else if (DISTRIBUTION == E_DISTRIBUTION::alignments) {
        printf("alignments distribution\n");
        RNG_generator_p = new RNG_alignments();
    }

    if (USE_IMAGE_INPUT == false)
        generateRandomPoints(PIC_WIDTH, SITES_NUMBER, DISTRIBUTION);
    else {
        if (readInputFile(PIC_WIDTH) == false) {     // 2D only
            printf("Fail to read input file!\n");
            return;
        }
    }

    initPoints(PIC_WIDTH, SITES_NUMBER);

    for (int i = 0; i < SITES_NUMBER; ++i) {
        kdtree_p[i].x[0] = inputPoints[i * 2];
        kdtree_p[i].x[1] = inputPoints[i * 2 + 1];
    }
    root = make_tree(kdtree_p, kdtree_p + SITES_NUMBER, 0);
    convert_tree(root, kdtree, 1, kdtree_array_size);

    printf("Image size: %dx%d\n", PIC_WIDTH, PIC_WIDTH);
    printf("Point count: %d\n", SITES_NUMBER);
    printf("K = %d\n", K);
    printf("-----------------\n");
    
    float dur_cuda = 0;     // execution time recorded by cuda timer

    if (CHECK_FAULT) {
        dur_cuda = bfaVoronoiDiagram(brute_voronoi_int, inputPoints, PIC_WIDTH); 
        printf("BFA completed.\n");
        
    }

    dur_cuda = pfaVoronoiDiagram(pfaOutputVoronoi, kdtree, 
        &dur_H2D_global_prfa[dur_idx], &dur_kernel_global_prfa[dur_idx], &dur_D2H_global_prfa[dur_idx], PIC_WIDTH, K);

    if (ITER_TEST)
        dur_global_prfa[dur_idx] = dur_cuda;

    printf("PRFA completed.\n");
    printf("-----------------\n");

    pfaDeinitialization();
    free(kdtree_p);
    free(kdtree);

    if (CHECK_FAULT) {
        verifyResult(PIC_WIDTH);
        bfaDeinitialization();
    }

    if (PRINT_VORONOI) {
        if (ITER_TEST == true) {
            std::string dir_path_front = IMAGE_OUTPUT_PREFIX;
            dir_path_front += std::to_string((dur_idx / 10));
            dir_path_front += std::to_string((dur_idx % 10));
            convert_output_diagram(voronoi_int, dir_path_front + ".txt", PIC_WIDTH, SITES_NUMBER);
        } else
            convert_output_diagram(voronoi_int, IMAGE_OUTPUT_PREFIX + IMAGE_FILE_NUMBER, PIC_WIDTH, SITES_NUMBER);
    }
}

int main(int argc,char **argv) {
    int PIC_WIDTH = 8192;
    int SITES_NUMBER = 671089;
    int DISTRIBUTION_NUM = 1;
    E_DISTRIBUTION DISTRIBUTION;
    int K = 11;
    int TREE_H2 = log2(SITES_NUMBER) + 1;
    printf("TREE_H: %i\n", TREE_H2);

    switch(DISTRIBUTION_NUM){
        case 1:
            DISTRIBUTION = E_DISTRIBUTION::uniform;
            break;
        case 2:
            DISTRIBUTION = E_DISTRIBUTION::normal;
            break;
        case 3:
            DISTRIBUTION = E_DISTRIBUTION::clusters;
            break;
        case 4:
            DISTRIBUTION = E_DISTRIBUTION::alignments;
            break;
        default:
            return(EXIT_FAILURE);
    }


    initialization(PIC_WIDTH, SITES_NUMBER);

    if (ITER_TEST) {
        double avg_dur_total_prfa = 0, avg_dur1_prfa = 0, avg_dur2_prfa = 0, avg_dur3_prfa = 0;
        double avg_error_count_prfa = 0;

        for (dur_idx = 0; dur_idx < ITER_COUNT; ++dur_idx) {
            printf("-----------------\n");
            printf("iter num: %d\n", dur_idx);
            runTests(PIC_WIDTH, SITES_NUMBER, DISTRIBUTION, K);
            if (dur_idx == 0) continue;

            avg_dur_total_prfa += dur_global_prfa[dur_idx];
            avg_dur1_prfa += dur_H2D_global_prfa[dur_idx]; avg_dur2_prfa += dur_kernel_global_prfa[dur_idx]; avg_dur3_prfa += dur_D2H_global_prfa[dur_idx];
            avg_error_count_prfa += error_count_global_prfa[dur_idx];
        }
        int iter_count = ITER_COUNT - 1;
        avg_dur_total_prfa /= iter_count;
        avg_dur1_prfa /= iter_count; avg_dur2_prfa /= iter_count; avg_dur3_prfa /= iter_count;
        avg_error_count_prfa /= iter_count * 1.0f;

        printf("-----------------\n");
        printf("PRFA complete, total %d iter\n", iter_count);
        // RNG instances, uniform = 0, normal = 1, clusters = 2, alignments = 3
        if (DISTRIBUTION == E_DISTRIBUTION::uniform) {
            printf("uniform distribution\n");
        } else if (DISTRIBUTION == E_DISTRIBUTION::normal) {
            printf("normal distribution\n");
        } else if (DISTRIBUTION == E_DISTRIBUTION::clusters) {
            printf("clusters distribution\n");
        } else if (DISTRIBUTION == E_DISTRIBUTION::alignments) {
            printf("alignments distribution\n");
        }
        printf("Image width = %d, point count = %d, K = %d\n", PIC_WIDTH, SITES_NUMBER, K);
        printf("avg dur = %.4fms\n", avg_dur_total_prfa);
        printf("avg dur H2D = %.4fms, avg dur kernel = %.4fms, avg dur D2H = %.4fms\n", avg_dur1_prfa, avg_dur2_prfa, avg_dur3_prfa);
        printf("avg error count = %.4f\n", avg_error_count_prfa);
    } else {
        runTests(PIC_WIDTH, SITES_NUMBER, DISTRIBUTION, K);
    }
    deinitialization();
	return 0;
}