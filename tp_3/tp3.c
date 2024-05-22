#include "tp3.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct entry_t {
    char *key;
    void *value;
    bool deleted;
} entry_t;

struct dictionary {
    destroy_f destroy;
    int len;
    int cap;
    double capacity_factor;
    entry_t **entry;
};

int hash_function(const char *key, int cap);

entry_t *entry_create(const char *key, void *value);

bool rehash(dictionary_t *dictionary);

// Custom strdup implementation
char *custom_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *dup = malloc(len);
    if (!dup) {
        return NULL;
    }
    memcpy(dup, s, len);
    return dup;
}

int hash_function(const char *key, int cap) {
    int hash = 0;
    for (int i = 0; key[i] != '\0'; i++) {
        hash += key[i];
    }
    return hash % cap;
}

entry_t *entry_create(const char *key, void *value) {
    entry_t *entry = malloc(sizeof(entry_t));
    if (!entry) {
        return NULL;
    }

    entry->key = custom_strdup(key);  // Duplicate the key to ensure it is stored safely
    if (!entry->key) {
        free(entry);
        return NULL;
    }

    entry->value = value;
    entry->deleted = false;

    return entry;
}

dictionary_t *dictionary_create(destroy_f destroy) {
    dictionary_t *dict = malloc(sizeof(dictionary_t));

    if (!dict) {
        return NULL;
    }

    dict->len = 0;
    dict->cap = 10;
    dict->capacity_factor = 0.75 * dict->cap;
    dict->destroy = destroy;

    dict->entry = calloc(dict->cap, sizeof(entry_t *));  // Allocate memory for the entry array
    if (!dict->entry) {
        free(dict);
        return NULL;
    }

    return dict;
}

bool rehash(dictionary_t *dictionary) {
    entry_t **old_entry = dictionary->entry;
    int old_cap = dictionary->cap;

    dictionary->cap *= 2;
    dictionary->capacity_factor = 0.75 * dictionary->cap;

    dictionary->entry = calloc(dictionary->cap, sizeof(entry_t *));
    if (!dictionary->entry) {
        dictionary->entry = old_entry;
        dictionary->cap = old_cap;
        return false;
    }

    for (int i = 0; i < old_cap; i++) {
        if (old_entry[i] && !old_entry[i]->deleted) {
            bool result = dictionary_put(dictionary, old_entry[i]->key, old_entry[i]->value);
            if (!result) {
                free(dictionary->entry);
                dictionary->entry = old_entry;
                dictionary->cap = old_cap;
                return false;
            }
            free(old_entry[i]->key);  // Free the duplicated key
            free(old_entry[i]);  // Free the entry
        }
    }

    free(old_entry);
    return true;
}

bool dictionary_put(dictionary_t *dictionary, const char *key, void *value) {
    if (dictionary->len >= dictionary->capacity_factor) {
        bool result = rehash(dictionary);
        if (!result) {
            return false;
        }
    }

    int index = hash_function(key, dictionary->cap);

    entry_t *new_entry = entry_create(key, value);

    if (!new_entry) {
        return false;
    }

    // Avoid collision using linear probing
    while (dictionary->entry[index]) {
        if (dictionary->entry[index]->deleted) {
            free(dictionary->entry[index]->key);
            free(dictionary->entry[index]);
            dictionary->entry[index] = new_entry;
            dictionary->len++;
            return true;
        }

        if (strcmp(dictionary->entry[index]->key, key) == 0) {
            if (dictionary->destroy && dictionary->entry[index]->value) {
                dictionary->destroy(dictionary->entry[index]->value);
            }
            free(dictionary->entry[index]->key);
            free(dictionary->entry[index]);
            dictionary->entry[index] = new_entry;
            return true;
        }
        index = (index + 1) % dictionary->cap;
    }

    dictionary->entry[index] = new_entry;
    dictionary->len++;

    return true;
}

void *dictionary_get(dictionary_t *dictionary, const char *key, bool *err) {
  int index = hash_function(key, dictionary->cap);
  int start_index = index;

  while (dictionary->entry[index]) {
    if (!dictionary->entry[index]->deleted && strcmp(dictionary->entry[index]->key, key) == 0) {
      *err = false;
      return dictionary->entry[index]->value;
    }
    index = (index + 1) % dictionary->cap;
    if (index == start_index) {
      *err = true;
      return NULL;
    }
  }

  *err = true;
  return NULL;
}

bool dictionary_delete(dictionary_t *dictionary, const char *key) {
  int index = hash_function(key, dictionary->cap);
  int start_index = index;

  while (dictionary->entry[index]) {
    if (strcmp(dictionary->entry[index]->key, key) == 0 && !dictionary->entry[index]->deleted) {
      if (dictionary->destroy && dictionary->entry[index]->value) {
        dictionary->destroy(dictionary->entry[index]->value);
      }
      free(dictionary->entry[index]->key);
      free(dictionary->entry[index]); // Free the entry
      dictionary->entry[index] = NULL; // Set the entry pointer to NULL
      dictionary->len--;
      return true;
    }
    index = (index + 1) % dictionary->cap;
    if (index == start_index) {
      return false;
    }
  }

  return false;
}

void *dictionary_pop(dictionary_t *dictionary, const char *key, bool *err) {
    int index = hash_function(key, dictionary->cap);
    int start_index = index;

    while (dictionary->entry[index]) {
        if (strcmp(dictionary->entry[index]->key, key) == 0 && !dictionary->entry[index]->deleted) {
            void *value = dictionary->entry[index]->value;
            free(dictionary->entry[index]->key); // Free the key
            free(dictionary->entry[index]); // Free the entry
            dictionary->entry[index] = NULL; // Set the entry pointer to NULL
            dictionary->len--;
            *err = false;
            return value;
        }
        index = (index + 1) % dictionary->cap;
        if (index == start_index) {
            break;
        }
    }

    *err = true;
    return NULL;
}

bool dictionary_contains(dictionary_t *dictionary, const char *key) {
    int index = hash_function(key, dictionary->cap);
    int start_index = index;

    while (dictionary->entry[index]) {
        if (strcmp(dictionary->entry[index]->key, key) == 0 && !dictionary->entry[index]->deleted) {
            return true;
        }
        index = (index + 1) % dictionary->cap;
        if (index == start_index) {
            break;
        }
    }

    return false;
}

size_t dictionary_size(dictionary_t *dictionary) {
    return dictionary->len;
}

void dictionary_destroy(dictionary_t *dictionary) {
    for (int i = 0; i < dictionary->cap; i++) {
        if (dictionary->entry[i] && !dictionary->entry[i]->deleted) {
            if (dictionary->destroy && dictionary->entry[i]->value) {
                dictionary->destroy(dictionary->entry[i]->value);
            }
            free(dictionary->entry[i]->key);
            free(dictionary->entry[i]);
        }
    }
    free(dictionary->entry);
    free(dictionary);
}
