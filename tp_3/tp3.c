#include "tp3.h"
#include <stdlib.h>

typedef struct entry_t {
    const char *key;
    void *value;
} entry_t;

struct dictionary {
    destroy_f destroy;
    int len;
    int cap;
    entry_t *entry;
};

int hash_function(const char *key, int cap);

entry_t *entry_create (const char *key, void *value);

entry_t *entry_create (const char *key, void *value) {
  entry_t *entry = malloc(sizeof(entry_t));
  if (!entry) {
    return NULL;
  }

  entry->key = key;
  entry->value = value;

  return entry;
}

int hash_function(const char *key, int cap) {
  int hash = 0;
  for (int i = 0; key[i] != '\0'; i++) {
    hash += key[i];
  }
  return hash % cap;
}

dictionary_t *dictionary_create(destroy_f destroy) {
  dictionary_t *dict = malloc(sizeof(dictionary_t));

  if (!dict) {
    return NULL;
  }

  dict->len = 0;
  dict->cap = 10;
  dict->destroy = destroy;

  dict->entry = malloc(sizeof(entry_t) * dict->cap);
  if (!dict->entry) {
    free(dict);
    return NULL;
  }

  return dict;
};

bool dictionary_put(dictionary_t *dictionary, const char *key, void *value) {
  if (dictionary->len == dictionary->cap) {
    return false;
  }

  int index = hash_function(key, dictionary->cap);

  entry_t *result = entry_create(key, value);

  if (!result) {
    return false;
  }

  dictionary->entry[index] = *result;
  dictionary->len++;

  return true;
};

void *dictionary_get(dictionary_t *dictionary, const char *key, bool *err) {
  int index = hash_function(key, dictionary->cap);

  if (dictionary->entry[index].key == key) {
    *err = false;
    return dictionary->entry[index].value;
  }

  *err = true;
  return NULL;
};

bool dictionary_delete(dictionary_t *dictionary, const char *key) {
  return true;
};

void *dictionary_pop(dictionary_t *dictionary, const char *key, bool *err) {
  return NULL;
};

bool dictionary_contains(dictionary_t *dictionary, const char *key) {
  int index = hash_function(key, dictionary->cap);

  return dictionary->entry[index].key == key;
};

size_t dictionary_size(dictionary_t *dictionary) {
  return dictionary->len;
};

void dictionary_destroy(dictionary_t *dictionary){
  for (int i = 0; i < dictionary->len; i++) {
    if (dictionary->entry[i].value != NULL) {
      dictionary->destroy(dictionary->entry[i].value);  // Free the value
    }
  }
  free(dictionary->entry);
  free(dictionary);
};