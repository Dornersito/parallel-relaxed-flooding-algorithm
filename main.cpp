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
#include <filesystem>


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


// static const bool CHECK_FAULT = true;       // check incorrect results of pixel labels
//static const bool PRINT_FAULT = true;       // print all incorrect pixel crds
static const bool ITER_TEST = true;
// static const int ITER_COUNT = 5 + 1;       // skip the 1st iter for average time
static const bool USE_IMAGE_INPUT = false;  // if true, change the SITES_NUMBER to 0 and set PIC_WIDTH to the image's size
// static const bool PRINT_VORONOI = true;

// test data file, nuclei_frame0XX.txt
static const std::string IMAGE_INPUT_PREFIX = "../images/nuclei_EDF0";
static const std::string IMAGE_OUTPUT_PREFIX = "../images/diagram_output/diagram_EDF0";
static const std::string IMAGE_FILE_NUMBER = "07.txt";

// float dur_global_prfa[ITER_COUNT], dur_H2D_global_prfa[ITER_COUNT], dur_kernel_global_prfa[ITER_COUNT], dur_D2H_global_prfa[ITER_COUNT];
// int error_count_global_prfa[ITER_COUNT];

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
void deinitialization(bool PRINT_VORONOI, bool CHECK_FAULT) {
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
void initialization(int PIC_WIDTH, int SITES_NUMBER, bool PRINT_VORONOI, bool CHECK_FAULT) {
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
void initPoints(int PIC_WIDTH, int SITES_NUMBER, bool CHECK_FAULT) {
    kdtree_array_size = next_pow_of_2(SITES_NUMBER);
    pfaInitialization(kdtree_array_size, PIC_WIDTH);
    if (CHECK_FAULT)
        bfaInitialization(SITES_NUMBER, PIC_WIDTH);
    
    kdtree_p          = (struct kd_node_t *) malloc((ULL)SITES_NUMBER * (ULL)sizeof(struct kd_node_t)); 
    kdtree            = (DTYPE *) malloc((ULL)kdtree_array_size * (ULL)DIM * (ULL)sizeof(DTYPE));
}
#undef ULL

// Verify the output Voronoi Diagram
void verifyResult(int PIC_WIDTH, int *error_count_global_prfa, bool PRINT_FAULT) {
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

    // Obtener la ruta del directorio
    std::filesystem::path dir = std::filesystem::path(pic_name).parent_path();
    std::cout << "RUTA: " << dir << std::endl;
    // Verificar si la carpeta existe
    if (!std::filesystem::exists(dir)) {
        // Crear la carpeta si no existe
        std::filesystem::create_directories(dir);
    }

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
        // if (j % 100 == 0) printf("line %d complete\n", j);
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
void runTests(int PIC_WIDTH, int SITES_NUMBER, E_DISTRIBUTION DISTRIBUTION, int K, int TREE_H, std::string output_path, bool print_iter,
    float *dur_global_prfa, float *dur_H2D_global_prfa, float *dur_kernel_global_prfa, float *dur_D2H_global_prfa, int *error_count_global_prfa,
    bool PRINT_VORONOI, bool CHECK_FAULT, bool PRINT_FAULT) {
    // RNG instances, uniform = 0, normal = 1, clusters = 2, alignments = 3
    if (DISTRIBUTION == E_DISTRIBUTION::uniform) {
        if(print_iter) printf("uniform distribution\n");
        RNG_generator_p = new KISS_RNG_uniform();
    } else if (DISTRIBUTION == E_DISTRIBUTION::normal) {
        if(print_iter) printf("normal distribution\n");
        RNG_generator_p = new RNG_normal();
    } else if (DISTRIBUTION == E_DISTRIBUTION::clusters) {
        if(print_iter) printf("clusters distribution\n");
        RNG_generator_p = new RNG_clusters();
    } else if (DISTRIBUTION == E_DISTRIBUTION::alignments) {
        if(print_iter) printf("alignments distribution\n");
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

    initPoints(PIC_WIDTH, SITES_NUMBER, CHECK_FAULT);
    //gpuErrchk( cudaPeekAtLastError() );


    for (int i = 0; i < SITES_NUMBER; ++i) {
        kdtree_p[i].x[0] = inputPoints[i * 2];
        kdtree_p[i].x[1] = inputPoints[i * 2 + 1];
    }
    root = make_tree(kdtree_p, kdtree_p + SITES_NUMBER, 0);
    convert_tree(root, kdtree, 1, kdtree_array_size);

    if(print_iter){
        printf("Image size: %dx%d\n", PIC_WIDTH, PIC_WIDTH);
        printf("Point count: %d\n", SITES_NUMBER);
        printf("K = %d\n", K);
        printf("-----------------\n");
    }
    
    float dur_cuda = 0;     // execution time recorded by cuda timer

    if (CHECK_FAULT) {
        dur_cuda = bfaVoronoiDiagram(brute_voronoi_int, inputPoints, PIC_WIDTH); 
        printf("BFA completed.\n");
        
    }

    dur_cuda = pfaVoronoiDiagram(pfaOutputVoronoi, kdtree, 
        &dur_H2D_global_prfa[dur_idx], &dur_kernel_global_prfa[dur_idx], &dur_D2H_global_prfa[dur_idx], PIC_WIDTH, K, TREE_H, print_iter);

    if (ITER_TEST)
        dur_global_prfa[dur_idx] = dur_cuda;

    printf("PRFA completed.\n");
    printf("-----------------\n");

    pfaDeinitialization();
    free(kdtree_p);
    free(kdtree);

    if (CHECK_FAULT) {
        verifyResult(PIC_WIDTH, error_count_global_prfa, PRINT_FAULT);
        bfaDeinitialization();
    }

    if (PRINT_VORONOI) {
        if (ITER_TEST == true) {
            size_t lastSlashPos = output_path.find_last_of("/\\");

            // Extraer el directorio base y el nombre del archivo del path original
            std::string baseDir = output_path.substr(0, lastSlashPos);
            std::string fileName = output_path.substr(lastSlashPos + 1);

            // Encontrar la posición del punto en el nombre del archivo (para la extensión)
            size_t lastDotPos = fileName.find_last_of('.');

            // Si no hay punto, se considera que no tiene extensión
            std::string extension;
            if (lastDotPos != std::string::npos) {
                extension = fileName.substr(lastDotPos);
                fileName = fileName.substr(0, lastDotPos);
            }

            fileName += "-iter";
            fileName += std::to_string((dur_idx / 10));
            fileName += std::to_string((dur_idx % 10));
            fileName += extension;

            std::string output_path = baseDir + "/" + fileName;
            
            convert_output_diagram(voronoi_int, output_path, PIC_WIDTH, SITES_NUMBER);
        } else
            convert_output_diagram(voronoi_int, output_path, PIC_WIDTH, SITES_NUMBER);
    }
}

bool stringToBool(const std::string &str) {
    return str == "true" || str == "1";
}

int main(int argc,char **argv) {
    if(argc < 9 or argc > 10){
        printf("Error. Ejecutar como ./PRFA [PIC_WIDTH] [SITES_NUMBER] [K] [iteraciones] [print_iter] *[export_path]\n");
        return EXIT_FAILURE;
    }



    bool CHECK_FAULT = stringToBool(argv[7]);
    bool PRINT_FAULT = stringToBool(argv[8]);

    int PIC_WIDTH = atoi(argv[1]);
    int SITES_NUMBER = atoi(argv[2]);
    int TREE_H = log2(SITES_NUMBER) + 1;
    int DISTRIBUTION_NUM = 1;
    int K = atoi(argv[3]);
    int ITER_COUNT = atoi(argv[4]) + 1;
    
    E_DISTRIBUTION DISTRIBUTION;
    //printf("TREE_H: %i\n", TREE_H);

    bool print_iter = stringToBool(argv[5]);
    std::string export_path = "";

    bool PRINT_VORONOI = false;
    if(argc == 9) {
        PRINT_VORONOI = true;
        export_path = argv[6];
    }


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


    float *dur_global_prfa, *dur_H2D_global_prfa, *dur_kernel_global_prfa, *dur_D2H_global_prfa;
    int *error_count_global_prfa;

    dur_global_prfa = new float [ITER_COUNT];
    dur_H2D_global_prfa = new float [ITER_COUNT];
    dur_kernel_global_prfa = new float [ITER_COUNT];
    dur_D2H_global_prfa = new float [ITER_COUNT];
    error_count_global_prfa = new int [ITER_COUNT];


    initialization(PIC_WIDTH, SITES_NUMBER, PRINT_VORONOI, CHECK_FAULT);

    if (ITER_TEST) {
        double avg_dur_total_prfa = 0, avg_dur1_prfa = 0, avg_dur2_prfa = 0, avg_dur3_prfa = 0;
        double avg_error_count_prfa = 0;

        for (dur_idx = 0; dur_idx < ITER_COUNT; ++dur_idx) {
            printf("-----------------\n");
            printf("iter num: %d\n", dur_idx);
            runTests(PIC_WIDTH, SITES_NUMBER, DISTRIBUTION, K, TREE_H, export_path, print_iter,
            dur_global_prfa, dur_H2D_global_prfa, dur_kernel_global_prfa, dur_D2H_global_prfa, error_count_global_prfa,
            PRINT_VORONOI, CHECK_FAULT, PRINT_FAULT);
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

        // std::string filename = "./datos_1-error.csv";
        // std::ifstream checkFile(filename);
        // bool fileExists = checkFile.good();
        // checkFile.close();

        // std::ofstream file(filename, std::ios::app);

        // if (!file.is_open()) {
        //     std::cerr << "Error al abrir el archivo " << filename << std::endl;
        //     return 1;
        // }
        // if (!fileExists) {
        //     file << "Image width,Point_count,K,Avg_duration total,Avg_duration H2D,Avg_duration_kernel,Avg_duration_D2H,Avg_error_count,Porcentaje_error\n";
        //     file << PIC_WIDTH << "," << SITES_NUMBER << "," << K << "," << avg_dur_total_prfa << "," << avg_dur1_prfa << "," << avg_dur2_prfa << "," << avg_dur3_prfa << "," << avg_error_count_prfa << "," << 100*avg_error_count_prfa/(PIC_WIDTH * PIC_WIDTH) <<"\n";

        // }
        // else{
        //     file << PIC_WIDTH << "," << SITES_NUMBER << "," << K << "," << avg_dur_total_prfa << "," << avg_dur1_prfa << "," << avg_dur2_prfa << "," << avg_dur3_prfa << "," << avg_error_count_prfa << "," << 100*avg_error_count_prfa/(PIC_WIDTH * PIC_WIDTH) << "\n";
        // }

        // file.close();
        // printf("Se guardaron los datos");

    } else {
        runTests(PIC_WIDTH, SITES_NUMBER, DISTRIBUTION, K, TREE_H, export_path, print_iter,
        dur_global_prfa, dur_H2D_global_prfa, dur_kernel_global_prfa, dur_D2H_global_prfa, error_count_global_prfa,
        PRINT_VORONOI, CHECK_FAULT, PRINT_FAULT);
    }
    deinitialization(PRINT_VORONOI, CHECK_FAULT);
	return 0;
}