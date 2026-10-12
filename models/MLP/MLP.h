#ifndef MLP_HEADER
    #define MLP_HEADER 0
    #include "../../general.h"
    #include "../../autograd/graph.h"
    #include "../../autograd/backpropagation.h"

    //return: [weight,bias,weight,bias,...,input]
    array_node* init_graph(int layers[], int n, af_array input);


    // L'output resta valido fino al prossimo forward, clear_mlp o free_mlp.
    node* forward(array_node* graph, activation_fun);

    // Per la catena creata da forward: libera operazioni e gradienti, conserva pesi, bias e input.
    void _clear_mlp(array_node* graph);

    // Libera l'intera MLP, compreso il suo handle dell'input.
    void _free_mlp(array_node* graph);

    void _update_weights(array_node* graph);

    void train(loss_fun loss ,af_array target, array_node* graph, activation_fun activation);
 

#endif
