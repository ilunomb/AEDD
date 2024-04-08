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


size_t list_length(const list_t *list){
    if(list) return list->size;

    return 0;
}

bool list_is_empty(const list_t *list){
    if(!list) return false;

    return list->size == 0;
}

bool list_insert_head(list_t *list, void *value){
    if(!list) return false;

    node_t* node = malloc(sizeof(node_t));
    
    if(!node) return false;

    node->next = list->head;
    node->prev = NULL;
    node->value = value;

    if(list->head) list->head->prev = node;
    if(!list->tail) list->tail = node;

    list->head = node;
    list->size++;

    return true;
}

bool list_insert_tail(list_t *list, void *value){
    if(!list) return false;

    node_t* node = malloc(sizeof(node_t));
    
    if(!node) return false;

    node->next = NULL;
    node->prev = list->tail;
    node->value = value;

    if(list->tail) list->tail->next = node;
    if(!list->head) list->head = node;

    list->tail = node;
    list->size++;

    return true;
}

void *list_peek_head(const list_t *list){
    if(!list || list_is_empty(list)) return NULL;

    return list->head->value;
}

void *list_peek_tail(const list_t *list){
    if(!list || list_is_empty(list)) return NULL;

    return list->tail->value;
}

void *list_pop_head(list_t *list){
    if(!list || list_is_empty(list)) return NULL;

    void* value = list->head->value;
    node_t* new_head = list->head->next;

    if(!new_head) list->tail = NULL; else new_head->prev = NULL;

    free(list->head);

    list->head = new_head;

    return value;    
}

void *list_pop_tail(list_t *list){
    return NULL;
}

void list_destroy(list_t *list, void destroy_value(void *)){
    if(!list) return;

    // while(!list_is_empty(list)){
    //     void* value = list_pop_tail(list);
    //     if(destroy_value) destroy_value(value);
    // }

    free(list);
}

list_iter_t *list_iter_create_head(list_t *list){
    return NULL;
}

list_iter_t *list_iter_create_tail(list_t *list){
    return NULL;
}

bool list_iter_forward(list_iter_t *iter){
    return false;
}

bool list_iter_backward(list_iter_t *iter){
    return false;
}

void *list_iter_peek_current(const list_iter_t *iter){
    return NULL;
}

bool list_iter_at_last(const list_iter_t *iter){
    return false;
}

bool list_iter_at_first(const list_iter_t *iter){
    return false;
}

void list_iter_destroy(list_iter_t *iter){
    return;
}

bool list_iter_insert_after(list_iter_t *iter, void *value){
    return false;
}

bool list_iter_insert_before(list_iter_t *iter, void *value){
    return false;
}

void *list_iter_delete(list_iter_t *iter){
    return NULL;
}