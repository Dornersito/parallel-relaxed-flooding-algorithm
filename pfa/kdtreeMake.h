/*
File Name: kdtreeMake.h
*/

#include "kdtree.h"
#include "../parameters.h"
#include "../utils/pfaUtils.h"
#include <algorithm>

#ifndef kdtreeMake_h
#define kdtreeMake_h

// this header is used in host

struct node_cmp {
    node_cmp(int index) : index_(index) {}
    bool operator()(const struct kd_node_t& n1, const struct kd_node_t& n2) const {
        return n1.x[index_] < n2.x[index_];
    }
    int index_;
};

struct kd_node_t* make_tree(struct kd_node_t *begin, struct kd_node_t *end, int i) {
    if (end <= begin)
        return nullptr;
    struct kd_node_t *mid = begin + (end - begin) / 2;
    std::nth_element(begin, mid, end, node_cmp(i));
    i = (i + 1) % DIM;    // dir change
    mid->left = make_tree(begin, mid, i);  // t to n
    mid->right = make_tree(mid + 1, end, i);  // (n + 1) to (t + len)
    return mid;
}

// convert from pointer to array
void convert_tree(struct kd_node_t *root_p, DTYPE *tree, int tree_idx, const int &tree_array_size) {
    if (!root_p) {
        tree[tree_idx * DIM + 0] = -1;
        return;
    }
    for (int d = 0; d < DIM; ++d) {
        tree[tree_idx * DIM + d] = root_p->x[d];
#if COMPRESS_KD_NODE == true
        if (d == 0)
            assign_child_kdtree(tree[tree_idx * DIM + d], 1 * ((root_p->left)!=nullptr));
        else if (d == 1)
            assign_child_kdtree(tree[tree_idx * DIM + d], 1 * ((root_p->right)!=nullptr));
        
#endif
    }
    if (tree_idx * 2 < tree_array_size) {
        convert_tree(root_p->left, tree, tree_idx * 2, tree_array_size);
        convert_tree(root_p->right, tree, tree_idx * 2 + 1, tree_array_size);
    }
}

#endif /* kdtreeMake_h */