#include "graph.h"
#include <stdlib.h>

node* alloc_node(node_type type)
{
    node* nodo = calloc(1, sizeof *nodo);
    if (nodo != NULL) {
        nodo->tipo = type;
    }
    return nodo;
}

void init_node(node* nodo, af_array valore,
               af_array gradiente, op operazione,
               node_type tipo, node* left, node* right,node* father_left, node* father_right){
    nodo->valore = valore;
    nodo->gradiente = gradiente;
    nodo->operazione = operazione;
    nodo->tipo = tipo;
    nodo->left = left;
    nodo->right = right;
    nodo->left = father_left;
    nodo->right = father_right;
}

void add_child(node* father,node* child){
    if(father->left == NULL){
        father->left = child;
        child->father_left = father;
        return;
    }else if(father->right == NULL){
        father->right = child;
        child->father_right = father;
        return;
    }

    printf("[!] il padre non ha spazio per il figlio");
}

node* create_val(af_array val){
    node* node = alloc_node(TYPE_VAL);
    init_node(node,val,NULL,OP_NULL,TYPE_VAL,NULL,NULL,NULL,NULL);
    return node;
}


node* add_matmul(node* left_node,node* right_node,op op){

    af_array A = 0;
    dim_t left_dims[4], right_dims[4];

    af_get_dims(&left_dims[0], &left_dims[1],&left_dims[2], &left_dims[3],left_node->valore); 
    af_get_dims(&right_dims[0], &right_dims[1],&right_dims[2], &right_dims[3],right_node->valore);

    if (left_dims[1] != right_dims[0]) {
        printf("[!] dimensioni non copatibili");
    }

    af_matmul(&A,left_node->valore,right_node->valore,AF_MAT_NONE,AF_MAT_NONE);

    node* node = alloc_node(TYPE_OP);
    init_node(node,A,NULL,op,TYPE_OP,NULL,NULL,NULL,NULL);
    add_child(left_node,node);
    add_child(right_node,node);
    return node;
}

node* add_sum(node* left_node,node* right_node,op op){

    af_array A = 0;

    af_add(&A, left_node->valore,right_node->valore, true);

    node* node = alloc_node(TYPE_OP);
    init_node(node,A,NULL,op,TYPE_OP,NULL,NULL,NULL,NULL);
    add_child(left_node,node);
    add_child(right_node,node);
    return node;
}

node* add_activation(node* input_node, float (*fun)(float),op op){

    af_array A = 0;
    af_copy_array(&A, input_node->valore);

    A = matrix_map(A,fun);

    node* node = alloc_node(TYPE_OP);
    init_node(node,A,NULL,op,TYPE_OP,NULL,NULL,NULL,NULL);
    add_child(input_node,node);
    return node;
}

node* add_softmax(node* input_node,op op){

    af_array A = 0;
    af_copy_array(&A, input_node->valore);

    A = softmax(A,1);

    node* node = alloc_node(TYPE_OP);
    init_node(node,A,NULL,op,TYPE_OP,NULL,NULL,NULL,NULL);
    add_child(input_node,node);
    return node;
}

node* add_loss(node* input_node,af_array target,af_array (*fun)(af_array,af_array),op op){

    af_array A = 0;
    af_copy_array(&A, input_node->valore);

    A = fun(A,target);

    node* node = alloc_node(TYPE_OP);
    init_node(node,A,NULL,op,TYPE_OP,NULL,NULL,NULL,NULL);
    add_child(input_node,node);
    return node;
}



node* add_op(node* left_node,node* right_node, af_array target,op op){

    switch (op){
    case OP_MATMUL:
        return add_matmul(left_node,right_node,op);
    case OP_SUM:
        return add_sum(left_node,right_node,op);
    case OP_RELU:
        return add_activation(left_node,relu,op);
    case OP_SOFTMAX:
        return add_softmax(left_node,op);
    case OP_CROSS_ENTROPY:
        return add_loss(left_node,target,cross_entropy,op);
    default:
        return NULL;
    }

}

af_array matrix_map(af_array matrix,float (*fun)(float)){

    dim_t dims[4];
    af_get_dims(&dims[0], &dims[1],&dims[2], &dims[3],matrix);
    af_array aggiornata = 0;
    
    
    for(dim_t i = 0; i<dims[0];i++){
        for (dim_t j = 0; j < dims[1]; j++){
            af_seq posizione[] = {
                {i, i, 1},
                {j, j, 1}
            };

            af_array elemento = 0;
            af_index(&elemento, matrix, 2, posizione);

            float x;
            af_get_scalar(&x, elemento);
            x = fun(x);

            af_array nuovo_valore = 0;
            dim_t dims[] = {1, 1};
            af_create_array(&nuovo_valore, &x, 2, dims, f32);

            aggiornata = 0;
            af_assign_seq(&aggiornata, matrix, 2, posizione, nuovo_valore);

            af_release_array(matrix);
            matrix = aggiornata;

            af_release_array(elemento);
            af_release_array(nuovo_valore);
        }
    }
    return aggiornata;
    
}

int is_node_in_array(array_node* array, node* node){
    int flag=0;
    for(int i=0; i<array->length;i++){
        if(array->node[i] == node) flag = 1;

    }
    return flag;
}




