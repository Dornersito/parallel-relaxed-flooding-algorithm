/*
File Name: numberGenerator.hpp
*/

#include <climits>
#include <random>   // for normal distribution
#include "../parameters.h"

#define GENERATOR_DIM DIM
#define GENERATOR_CLUSTER_COUNT RNG_CLUSTER_COUNT
#define GENERATOR_LINE_COUNT RNG_LINE_COUNT

#ifndef numberGenerator_hpp
#define numberGenerator_hpp

class RNG_virtual {
public:
    virtual void rand_init(unsigned long seed) = 0;
    virtual double random_one(bool bound = true) {
        return 0.0;
    }
    virtual void random_one_dim(double *output, bool bound = true) {
        return;
    }
};

// Uniform random number generator using KISS algorithm
// https://gist.github.com/3ki5tj/7b1d51e96d1f9bfb89bc

class KISS_RNG_uniform: public RNG_virtual {
private:
    unsigned long z, w, jsr, jcong; // Seeds
    unsigned long znew() 
        { return (z = 36969 * (z & 0xfffful) + (z >> 16)); }
    unsigned long wnew() 
        { return (w = 18000 * (w & 0xfffful) + (w >> 16)); }
    unsigned long MWC()  
        { return ((znew() << 16) + wnew()); }
    unsigned long SHR3()
        { jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5)); }
    unsigned long CONG() 
        { return (jcong = 69069 * jcong + 1234567); }
public:
    void rand_init(unsigned long seed) 
        { z = seed; w = seed; jsr = seed; jcong = seed; }
    unsigned long rand_int_uni()         // [0,2^32-1]
        { return ((MWC() ^ CONG()) + SHR3()); }
    double random_one(bool bound = true)                 // [0,1)
        { return ((double) rand_int_uni() / (double(ULONG_MAX)+1)); }
};

class RNG_normal: public RNG_virtual {
private:
    std::default_random_engine generator;
public:
    void rand_init(unsigned long seed) {
        generator.seed(seed);
    }
    // the mean is 0.5
    // if bound is true, the function will discard values < 0 or >= 1, and regenerate a value
    // notice that (mean = 0.5, sd = 0.2) * W is equivalent to (mean = 0.5 * W, sd = 0.2 * W)
    double random_one(bool bound = true) {
        std::normal_distribution<double> distribution(0.5, 0.2);
        if (!bound)
            return distribution(generator);
        else {
            double number;
            do {
                number = distribution(generator);
            } while (number - __DBL_EPSILON__ < 0.0 || number + __DBL_EPSILON__ >= 1.0);
            return number;
        }
    }
};

// first generate cluster centroids randomly with uniform distribution
// then pick a centroid randomly and generate points around it with normal distribution
class RNG_clusters: public RNG_virtual {
private:
    std::default_random_engine generator;
    double *cluster_centroids_p;
public:
    // also initialize the coordiantes of the cluster centroids
    void rand_init(unsigned long seed) {
        generator.seed(seed);
        cluster_centroids_p = new double[GENERATOR_CLUSTER_COUNT * GENERATOR_DIM];
        std::uniform_real_distribution<double> uni_distribution(0.0, 1.0);
        for (int i = 0; i < GENERATOR_CLUSTER_COUNT * GENERATOR_DIM; ++i) {
            cluster_centroids_p[i] = uni_distribution(generator);
        }
    }
    // output the x-crd in output[0], the y-crd in output[1]
    // if bound is true, the function will discard values < 0 or >= 1, and regenerate a value
    void random_one_dim(double *output, bool bound = true) {
        std::uniform_int_distribution<int> uni_distribution(0, GENERATOR_CLUSTER_COUNT - 1);
        std::normal_distribution<double> norm_distribution(0, 0.04);
        int cluster_idx = uni_distribution(generator);
        for (int d = 0; d < GENERATOR_DIM; ++d)
            output[d] = cluster_centroids_p[cluster_idx * GENERATOR_DIM + d];

        if (!bound) {
            for (int d = 0; d < GENERATOR_DIM; ++d)
                output[d] += norm_distribution(generator);
            return;
        } else {
            double crd_bias[GENERATOR_DIM];
            bool out_of_bound = true;
            while(out_of_bound) {
                out_of_bound = false;
                for (int d = 0; d < GENERATOR_DIM; ++d) {
                    crd_bias[d] = norm_distribution(generator);
                    out_of_bound |= (output[d] + crd_bias[d] - __DBL_EPSILON__ < 0.0) || 
                        (output[d] + crd_bias[d] + __DBL_EPSILON__ >= 1.0);
                }
            }
            for (int d = 0; d < GENERATOR_DIM; ++d)
                output[d] += crd_bias[d];
            return;
        }
    }
};

// Warning: 2D only
// first generate 2l points to determine l lines, where the points are generated randomly with uniform distribution
// then pick a line randomly and generate x-crd with uniform distribution
class RNG_alignments: public RNG_virtual {
private:
    std::default_random_engine generator;
    double *line_params_p;    // [m0, c0, m1, c1, ...], y=mx+c, m is gradient, c is y-intercept
public:
    // also initialize the parameters of the lines
    void rand_init(unsigned long seed) {
        generator.seed(seed + 5);
        line_params_p = new double[GENERATOR_LINE_COUNT * 2];
        std::uniform_real_distribution<double> uni_distribution(0.0, 1.0);
        for (int i = 0; i < GENERATOR_LINE_COUNT; ++i) {
            double x1 = uni_distribution(generator);
            double y1 = uni_distribution(generator);
            double x2 = uni_distribution(generator);
            double y2 = uni_distribution(generator);
            double m = (y1 - y2) / (x1 - x2);
            double c = y1 - m * x1;
            line_params_p[i * 2 + 0] = m;
            line_params_p[i * 2 + 1] = c;
        }
    }
    // Warning: 2D only
    // output the x-crd in output[0], the y-crd in output[1]
    // if bound is true, the function will discard values < 0 or >= 1, and regenerate a value
    void random_one_dim(double *output, bool bound = true) {
        std::uniform_int_distribution<int> uni_distribution_line(0, GENERATOR_LINE_COUNT - 1);
        std::uniform_real_distribution<double> uni_distribution_crd1(0.0, 1.0);
        std::normal_distribution<double> norm_distribution_crd2(0, 0.04);
        int line_idx = uni_distribution_line(generator);
        double m = line_params_p[line_idx * 2 + 0], c = line_params_p[line_idx * 2 + 1];
        if (!bound) {
            double x = uni_distribution_crd1(generator);
            double y = x * m + c;
            output[0] = x + norm_distribution_crd2(generator);
            output[1] = y + norm_distribution_crd2(generator);
            return;
        } else {
            bool out_of_bound = true;
            while(out_of_bound) {
                out_of_bound = false;
                double x = uni_distribution_crd1(generator);
                double y = x * m + c;
                output[0] = x + norm_distribution_crd2(generator);
                output[1] = y + norm_distribution_crd2(generator);
                out_of_bound |= (output[0] - __DBL_EPSILON__ < 0.0) || (output[0] + __DBL_EPSILON__ >= 1.0);
                out_of_bound |= (output[1] - __DBL_EPSILON__ < 0.0) || (output[1] + __DBL_EPSILON__ >= 1.0);
            }
            return;
        }
    }
};

#endif /* numberGenerator_hpp */