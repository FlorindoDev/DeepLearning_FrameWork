#ifndef BACK_PROPAGATION_HEADER
    #define BACK_PROPAGATION_HEADER 0
    #include "../general.h"
    #include "../activation_functions/activation_function.h"
    #include "../loss/loss.h"
    #include "graph.h"

    void backwards(array_node* leafs, af_array targets);
    void backwards_helper(node* cur_node, node* pre_node, af_array targets);

    void matmul_gradient(node* pre_node, node* node);
    void sum_gradient(node* pre_node, node* node);
    void activation_fun_gradient(node* pre_node, node* node,af_array (*fun)(af_array));
    void softmax_gradient(node* pre_node, node* node);
    void cross_entropy_gradient(node* node, af_array targets);

    void weight_gradient(node* pre_node, node* cur_node);
    void bias_gradient(node* pre_node, node* node);

#endif
