#include "backpropagation.h"


void find_leaf(array_node* nodes, array_node* leaf){

    for(int i = 0 ;i<nodes->length;i++){
        find_leaf_helper(nodes->node[i],leaf);
    }

}

void find_leaf_helper(node* node, array_node* leaf){

    if(node->left == NULL && node->right == NULL){
        if(!is_node_in_array(leaf, node)){
            leaf->node[leaf->length] = node;
            leaf->length++;
        }
        return;
    }
    
    if(node->left != NULL){
        find_leaf_helper(node->left,leaf);
    }

    if(node->right != NULL){
        find_leaf_helper(node->right,leaf);
    }

    return;

}