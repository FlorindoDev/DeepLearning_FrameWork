#ifndef BACK_PROPAGATION_HEADER
    #define BACK_PROPAGATION_HEADER 0
    #include "../general.h"
    #include "../activation_functions/activation_function.h"
    #include "../loss/loss.h"
    #include "graph.h"

    void find_leaf(array_node* nodes, array_node* leaf);
    void find_leaf_helper(node* node,  array_node* leaf);

    

#endif
