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
    size_t len;
    size_t cap;
    size_t capacity_factor;
    entry_t **entry;
};

char* custom_strdup(const char* s) {
    char* copy = malloc(strlen(s) + 1);
    if (!copy) {
        return NULL;
    }
    return strcpy(copy, s);
}

size_t hash_function(const char *key, size_t cap) {
    size_t hash = 5381;
    int c;

    while ((c = *key++))
        hash = ((hash << 5) + hash) + c;

    return hash % cap;
}

entry_t *entry_create(const char *key, void *value) {
    entry_t *entry = malloc(sizeof(entry_t));
    if (!entry) {
        return NULL;
    }

    entry->key = custom_strdup(key);
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
    dict->cap = 16;
    dict->capacity_factor = (3 * dict->cap) / 4;
    dict->destroy = destroy;

    dict->entry = calloc(dict->cap, sizeof(entry_t *));
    if (!dict->entry) {
        free(dict);
        return NULL;
    }

    return dict;
}

bool rehash(dictionary_t *dictionary) {
    entry_t **old_entry = dictionary->entry;
    size_t old_cap = dictionary->cap;
    size_t old_len = dictionary->len;

    dictionary->cap *= 2;
    dictionary->capacity_factor = (3 * dictionary->cap) / 4;

    dictionary->entry = calloc(dictionary->cap, sizeof(entry_t *));
    if (!dictionary->entry) {
        dictionary->entry = old_entry;
        dictionary->cap = old_cap;
        dictionary->len = old_len;
        dictionary->capacity_factor = (3 * dictionary->cap) / 4;
        return false;
    }

    dictionary->len = 0;
    for (size_t i = 0; i < old_cap; i++) {
        if (old_entry[i] && !old_entry[i]->deleted) {
            bool result = dictionary_put(dictionary, old_entry[i]->key, old_entry[i]->value);
            if (!result) {
                free(dictionary->entry);
                dictionary->entry = old_entry;
                dictionary->cap = old_cap;
                dictionary->len = old_len;
                dictionary->capacity_factor = (3 * dictionary->cap) / 4;
                return false;
            }
            free(old_entry[i]->key);
            free(old_entry[i]);
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

    size_t index = hash_function(key, dictionary->cap);

    size_t start_index = index;

    entry_t *new_entry = entry_create(key, value);

    if (!new_entry) {
        return false;
    }

    while (dictionary->entry[index]) {
        if (dictionary->entry[index]->deleted) {
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
        
        if (index == start_index) {
            if (dictionary->destroy && new_entry->value) {
                dictionary->destroy(new_entry->value);
            }
            free(new_entry->key);
            free(new_entry);
            return false;
        }
    }

    dictionary->entry[index] = new_entry;
    dictionary->len++;

    return true;
}

void *dictionary_get(dictionary_t *dictionary, const char *key, bool *err) {
    size_t index = hash_function(key, dictionary->cap);
    size_t start_index = index;

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
    size_t index = hash_function(key, dictionary->cap);
    size_t start_index = index;

    while (dictionary->entry[index]) {
        if (!dictionary->entry[index]->deleted && strcmp(dictionary->entry[index]->key, key) == 0) {
            if (dictionary->destroy && dictionary->entry[index]->value) {
                dictionary->destroy(dictionary->entry[index]->value);
            }
            dictionary->entry[index]->deleted = true;
            free(dictionary->entry[index]->key);
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
    size_t index = hash_function(key, dictionary->cap);
    size_t start_index = index;

    while (dictionary->entry[index]) {
        if (!dictionary->entry[index]->deleted && strcmp(dictionary->entry[index]->key, key) == 0) {
            void *value = dictionary->entry[index]->value;
            free(dictionary->entry[index]->key);
            dictionary->entry[index]->deleted = true;
            dictionary->len--;
            *err = false;
            return value;
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

bool dictionary_contains(dictionary_t *dictionary, const char *key) {
    size_t index = hash_function(key, dictionary->cap);
    size_t start_index = index;

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
    for (size_t i = 0; i < dictionary->cap; i++) {
        if (dictionary->entry[i]) {
            if (!dictionary->entry[i]->deleted) {
                if (dictionary->destroy && dictionary->entry[i]->value) {
                    dictionary->destroy(dictionary->entry[i]->value);
                }
                free(dictionary->entry[i]->key);
            }
            free(dictionary->entry[i]);
        }
    }
    free(dictionary->entry);
    free(dictionary);
}
