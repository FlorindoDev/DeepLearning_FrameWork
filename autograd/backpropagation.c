#include "backpropagation.h"

static void _accumulate_gradient(node* node, af_array gradiente);
static af_array _softmax_gradient_helper(af_array predizioni, af_array gradiente_loss);
static node* _other_father(node* child, node* father);

void backwards(array_node* leafs, af_array targets){

    for(int i=0;i<leafs->length;i++){
        backwards_helper(leafs->node[i],NULL,targets);
    }

}

void backwards_helper(node* cur_node, node* pre_node, af_array targets){

    if(cur_node->tipo == TYPE_WEIGHT){
        weight_gradient(pre_node, cur_node);
    }else if(cur_node->tipo == TYPE_BIAS){
        bias_gradient(pre_node, cur_node);
    }

    switch (cur_node->operazione){
    case OP_MATMUL:
        matmul_gradient(pre_node, cur_node);
        break;
    case OP_SUM:
        sum_gradient(pre_node, cur_node);
        break;

    case OP_RELU:
        activation_fun_gradient(pre_node,cur_node,relu_derivative);
        break;

    case OP_SOFTMAX:
        softmax_gradient(pre_node, cur_node);
        break;

    case OP_CROSS_ENTROPY:
        cross_entropy_gradient(cur_node, targets);
        break;

    case OP_NULL:
        break;

    }

    if(cur_node->father_right == NULL && cur_node->father_left == NULL){
        return;
    }

    if(cur_node->father_left != NULL){
        backwards_helper(cur_node->father_left,cur_node, NULL);
    }

    if(cur_node->father_right != NULL){
        backwards_helper(cur_node->father_right,cur_node, NULL);
    }

    return;

}



void weight_gradient(node* pre_node, node* cur_node){
    node* input = _other_father(pre_node, cur_node);
    if(input == NULL) return;

    af_array valore_input = 0, gradiente = 0;
    af_dtype tipo;
    af_get_type(&tipo, pre_node->gradiente);
    af_cast(&valore_input, input->valore, tipo);

    if(input->tipo == TYPE_INPUT){
        // Y = W * X: grad(W) = grad(Y) * X^T.
        af_matmul(&gradiente, pre_node->gradiente, valore_input, AF_MAT_NONE, AF_MAT_TRANS);
    }else{
        // Y = X * W: grad(W) =  H^T * grad(Y) .
        af_matmul(&gradiente, valore_input, pre_node->gradiente, AF_MAT_TRANS, AF_MAT_NONE);
    }
    af_release_array(valore_input);
    _accumulate_gradient(cur_node, gradiente);
}

void activation_fun_gradient(node* pre_node, node* node,double (*fun)(double)){

    af_array valore = 0;
    af_array gradiente_h = 0;
    af_array gradiente_relu = 0;

    struct node* input = _other_father(pre_node, node);

    af_copy_array(&valore,  node->father_left == NULL ? node->father_right->valore : node->father_left->valore);

    af_array derivative_relu = matrix_map(valore,fun);

    af_matmul(&gradiente_h,pre_node->gradiente, input->valore,AF_MAT_NONE, AF_MAT_TRANS);
    af_mul(&gradiente_relu, derivative_relu, gradiente_h, false);
    af_release_array(derivative_relu);

    node->temp = gradiente_h;
    _accumulate_gradient(node, gradiente_relu);

}

void softmax_gradient(node* pre_node, node* node){
    af_array gradeinte_child = pre_node->gradiente;
    
    af_array gradiente = _softmax_gradient_helper(node->valore, gradeinte_child);
    _accumulate_gradient(node, gradiente);
}

void cross_entropy_gradient(node* node, af_array targets){
    af_array gradiente = cross_entropy_derivative(node->father_left->valore,targets);
    _accumulate_gradient(node, gradiente);
}

void sum_gradient(node* pre_node, node* node){
    // Handle distinto: l'accumulo puo' rilasciarlo senza invalidare il figlio.
    af_array gradeinte_child = 0;
    if(af_retain_array(&gradeinte_child, pre_node->gradiente) != AF_SUCCESS) return;
    _accumulate_gradient(node, gradeinte_child);

}

void bias_gradient(node* pre_node, node* node){
    af_array gradiente_bias = 0;
    af_sum(&gradiente_bias, pre_node->gradiente, 0);
    _accumulate_gradient(node, gradiente_bias);
}

void matmul_gradient(node* pre_node, node* node){
    af_array gradeinte_child = 0;
    if(af_retain_array(&gradeinte_child, pre_node->gradiente) != AF_SUCCESS) return;
    _accumulate_gradient(node, gradeinte_child);
}


static node* _other_father(node* child, node* father){
    
    node* altro_padre = NULL;
    if(child->father_left == father){
        altro_padre = child->father_right;
    }else if(child->father_right == father){
        altro_padre = child->father_left;
    }

    return altro_padre;
}

static void _accumulate_gradient(node* node, af_array gradiente){
    if(node->gradiente != NULL && have_tow_child(node)){
        af_array somma = 0;
        af_add(&somma, node->gradiente, gradiente, false);
        af_release_array(node->gradiente);
        af_release_array(gradiente);
        node->gradiente = somma;
    }else{
        node->gradiente = gradiente;
    }
}


static af_array _softmax_gradient_helper(af_array predizioni, af_array gradiente_loss){
    af_array derivata = softmax_derivative(predizioni);
    af_array convertita = 0, derivate_batch = 0, gradiente_batch = 0;
    af_array prodotto = 0, risultato = 0;
    af_dtype tipo;
    af_get_type(&tipo, gradiente_loss);
    af_cast(&convertita, derivata, tipo);

    // [campioni, i, j] -> [i, j, campioni].
    af_reorder(&derivate_batch, convertita, 1, 2, 0, 3);
    // [campioni, i] -> [1, i, campioni].
    af_reorder(&gradiente_batch, gradiente_loss, 2, 1, 0, 3);
    af_matmul(&prodotto, gradiente_batch, derivate_batch, AF_MAT_NONE, AF_MAT_NONE);
    // [1, j, campioni] -> [campioni, j].
    af_reorder(&risultato, prodotto, 2, 1, 0, 3);

    af_release_array(derivata);
    af_release_array(convertita);
    af_release_array(derivate_batch);
    af_release_array(gradiente_batch);
    af_release_array(prodotto);
    return risultato;
}





