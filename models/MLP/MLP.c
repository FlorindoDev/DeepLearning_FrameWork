#include "MLP.h"
#include "../../optimizers/optimizer.h"


/*
################################################################
# RILASCIO DELLA MEMORIA DELLA MLP                              #
################################################################
*/

static void _clear_gradients(node* nodo){
    if(nodo->gradiente != NULL) af_release_array(nodo->gradiente);
    if(nodo->temp != NULL) af_release_array(nodo->temp);
    nodo->gradiente = NULL;
    nodo->temp = NULL;
}

void _clear_mlp(array_node* graph){
    node* operazione = graph->node[graph->length - 1]->left;
    while(operazione != NULL){
        node* successiva = operazione->left;
        _clear_gradients(operazione);
        if(operazione->valore != NULL) af_release_array(operazione->valore);
        free(operazione);
        operazione = successiva;
    }

    for(int i = 0; i < graph->length; i++){
        _clear_gradients(graph->node[i]);
        graph->node[i]->left = NULL;
        graph->node[i]->right = NULL;
    }
}

void _free_mlp(array_node* graph){
    _clear_mlp(graph);
    for(int i = 0; i < graph->length; i++){
        if(graph->node[i]->valore != NULL) af_release_array(graph->node[i]->valore);
        free(graph->node[i]);
    }
    free(graph->node);
    free(graph);
}


/*
################################################################
#                   MLP                                        #
################################################################
*/


array_node* init_graph(int layers[], int nlayers, af_array input){
    
    array_node* graph = calloc(1, sizeof *graph);

    graph->size = 2 * nlayers + 1;
    graph->node = calloc((size_t)graph->size, sizeof *graph->node);

    for(int i = 0; i < graph->size; i++){
        af_array valore = 0;
        node_type tipo;

        if(i == graph->size - 1){
            tipo = TYPE_INPUT;
            if(input != NULL) af_retain_array(&valore, input);
        }else if(i % 2 == 0){
            tipo = TYPE_WEIGHT;
            valore = init_weights(layers[(i/2)-1],layers[i/2]);
        }else{
            tipo = TYPE_BIAS;
            dim_t dims[] = {1, layers[i/2]};
            af_constant(&valore, 0.0, 2, dims, f64);
        }

        graph->node[i] = alloc_node(tipo);
        
        init_node(graph->node[i], valore, NULL, NULL, OP_NULL, tipo, NULL, NULL, NULL, NULL);
        graph->length++;
    }

    return graph;
}

void _update_weights(array_node* graph){
    for(int i = 0; i < graph->length; i++){
        node* current = graph->node[i];
        if(current->tipo == TYPE_WEIGHT || current->tipo == TYPE_BIAS){
            sgd(&current->valore, current->gradiente, 0.01);
        }
    }
}


void train(loss_fun loss,af_array target, array_node* graph, activation_fun activation){
    node* node_loss = NULL;
    node* output = forward(graph,activation);

    if(loss == CROSS_ENTROPY)
        node_loss =  add_op(output,NULL,target,OP_CROSS_ENTROPY);

    array_node leafs = {.node = &node_loss, .size = 1, .length = 1};
    backwards(&leafs,target);
    _update_weights(graph);
}



node* forward(array_node* graph, activation_fun activation){

    _clear_mlp(graph);
    node* node = 0;
    for(int i=0; i<graph->length-1;i++){
        if(i==0){
            node = add_op(graph->node[graph->length-1],graph->node[i],NULL,OP_MATMUL);
        }else if(i%2 == 0){
            node = add_op(node,graph->node[i],NULL,OP_MATMUL);
        }else{
            node = add_op(node,graph->node[i],NULL,OP_SUM);
            if(graph->length-2 != i){
                if(activation == RELU)
                    node = add_op(node,NULL,NULL,OP_RELU);
            }
    
        }

    }

    node =  add_op(node,NULL,NULL,OP_SOFTMAX);
    return node;

}
