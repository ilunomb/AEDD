#include "tp2.h"
#include <stdlib.h>
#include <stdbool.h>

struct node;

typedef struct node node_t;

struct node {
    void* value;
    node_t* next;
    node_t* prev;
};

struct list {
    node_t* head;
    node_t* tail;
    size_t size;
};

struct list_iter {
    list_t* list;
    node_t* curr;
};

list_t *list_new(){
    list_t* list = malloc(sizeof(list_t));

    if(!list) return NULL;

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;

    return list;
}

//Agrego las declaraciones de todas mis funciones auxiliares, ya que el campus no me deja entregar mas de un archivo, y no permite el zip con el .h y el .c,

/*
 * inserta un nodo en la lista
 */
bool list_insert(list_t *list, void *value, node_t *prev_node, node_t *next_node);

/*
 * pop a node from the list
 */
void *list_pop_node(list_t *list, bool pop_head);

/*
 * Crea un iterador
 */
list_iter_t *list_iter_create(list_t *list, bool create_head);

/*
 * Mueve el iterador
 */
bool list_iter_move(list_iter_t *iter, bool move_forward);

/*
 * Devuelve si el iterador está en la posición deseada
 */
bool list_iter_at(const list_iter_t *iter, bool head);

/*
 * Inserta un nodo en la lista por medio del iterador
 */
bool list_iter_insert(list_iter_t *iter, void *value, bool after);


size_t list_length(const list_t *list){
    if(list) return list->size;

    return 0;
}

bool list_is_empty(const list_t *list){
    if(!list) return false;

    return list->size == 0;
}



bool list_insert_head(list_t *list, void *value){
    return list_insert(list, value, NULL, list->head);
}

bool list_insert_tail(list_t *list, void *value){
    return list_insert(list, value, list->tail, NULL);
}


void *list_peek_head(const list_t *list){
    if(!list || list_is_empty(list)) return NULL;

    return list->head->value;
}

void *list_peek_tail(const list_t *list){
    if(!list || list_is_empty(list)) return NULL;

    return list->tail->value;
}

void *list_pop_head(list_t *list) {
    return list_pop_node(list, true);
}

void *list_pop_tail(list_t *list) {
    return list_pop_node(list, false);
}

void list_destroy(list_t *list, void destroy_value(void *)){
    if(!list) return;

    while(!list_is_empty(list)){
        void* value = list_pop_tail(list);
        if(destroy_value) destroy_value(value);
    }

    free(list);
}

list_iter_t *list_iter_create_head(list_t *list){
    return list_iter_create(list, true);
}

list_iter_t *list_iter_create_tail(list_t *list){
    return list_iter_create(list, false);
}

bool list_iter_forward(list_iter_t *iter){
    return list_iter_move(iter, true);
}

bool list_iter_backward(list_iter_t *iter){
    return list_iter_move(iter, false);
}

void *list_iter_peek_current(const list_iter_t *iter){
    if(!iter || list_is_empty(iter->list)) return NULL;

    return iter->curr->value;
}

bool list_iter_at_first(const list_iter_t *iter){
    return list_iter_at(iter, true);
}

bool list_iter_at_last(const list_iter_t *iter){
    return list_iter_at(iter, false);
}

void list_iter_destroy(list_iter_t *iter){
    if(!iter) return;

    free(iter);
}

bool list_iter_insert_after(list_iter_t *iter, void *value){
    return list_iter_insert(iter, value, true);
}

bool list_iter_insert_before(list_iter_t *iter, void *value){
    return list_iter_insert(iter, value, false);
}

void *list_iter_delete(list_iter_t *iter){
    if(!iter || list_is_empty(iter->list)) return NULL;

    if(list_iter_at_last(iter)){
        iter->curr = iter->curr->prev;
        return list_pop_tail(iter->list);
    }

    if(list_iter_at_first(iter)){
        iter->curr = iter->curr->next;
        return list_pop_head(iter->list);
    }

    void* value = iter->curr->value;
    node_t* next_node = iter->curr->next;

    iter->curr->prev->next = iter->curr->next;
    iter->curr->next->prev = iter->curr->prev;

    free(iter->curr);

    iter->curr = next_node;
    iter->list->size--;

    return value;
}


//AUXILIAR FUNCTIONS

bool list_insert(list_t *list, void *value, node_t *prev_node, node_t *next_node){
    if(!list) return false;

    node_t* node = malloc(sizeof(node_t));
    
    if(!node) return false;

    node->next = next_node;
    node->prev = prev_node;
    node->value = value;

    if(prev_node) prev_node->next = node;
    if(next_node) next_node->prev = node;

    if(!prev_node) list->head = node;
    if(!next_node) list->tail = node;

    list->size++;

    return true;
}

void *list_pop_node(list_t *list, bool pop_head) {
    if (!list || list_is_empty(list)) return NULL;

    node_t *node = pop_head ? list->head : list->tail;
    void *value = node->value;
    node_t *adjacent_node = pop_head ? node->next : node->prev;

    if (pop_head) {
        list->head = adjacent_node;
        if (!adjacent_node){
            list->tail = NULL;
        }
        else{
            adjacent_node->prev = NULL;
        }

    } else {
        list->tail = adjacent_node;
        if (!adjacent_node){
            list->head = NULL;
        }
        else{
            adjacent_node->next = NULL;
        }
    }

    free(node);
    list->size--;

    return value;
}

list_iter_t *list_iter_create(list_t *list, bool create_head) {
    if(!list) return NULL;

    list_iter_t* iter = malloc(sizeof(list_iter_t));

    if(!iter) return NULL;

    iter->list = list;
    iter->curr = create_head ? list->head : list->tail;

    return iter;
}

bool list_iter_move(list_iter_t *iter, bool move_forward) {
    if(!iter) return false;

    node_t *boundary_node = move_forward ? iter->list->tail : iter->list->head;
    if(iter->curr == boundary_node) return false;

    iter->curr = move_forward ? iter->curr->next : iter->curr->prev;

    return true;
}

bool list_iter_at(const list_iter_t *iter, bool head) {
    if(!iter) return false;

    node_t *boundary_node = head ? iter->list->head : iter->list->tail;

    return iter->curr == boundary_node;
}

bool list_iter_insert(list_iter_t *iter, void *value, bool after){
    if(!iter) return false;

    if(list_is_empty(iter->list)){
        bool result = list_insert(iter->list, value, NULL, NULL);
        if (result) iter->curr = iter->list->head;
        return result;
    }

    if(after){
        if(list_iter_at_last(iter)) return list_insert(iter->list, value, iter->curr, NULL);
        return list_insert(iter->list, value, iter->curr, iter->curr->next);
    } else {
        if(list_iter_at_first(iter)) return list_insert(iter->list, value, NULL, iter->curr);
        return list_insert(iter->list, value, iter->curr->prev, iter->curr);
    }
}