// SPDX-License-Identifier: PMPL-1.0-or-later
#ifndef AXIOM_H
#define AXIOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t (*axiom_callback_t)(uint64_t ctx, uint32_t input);

void *axiom_init(void);
void axiom_free(void *handle);

int32_t axiom_process(void *handle, uint32_t input);
const char *axiom_get_string(void *handle);
void axiom_free_string(const char *str);
int32_t axiom_process_array(void *handle, const uint8_t *buffer, uint32_t len);

const char *axiom_last_error(void);
const char *axiom_version(void);
const char *axiom_build_info(void);

int32_t axiom_register_callback(void *handle, uint64_t callback_ptr);
int32_t axiom_invoke_callback(void *handle, uint64_t ctx, uint32_t input, uint32_t *out);

uint32_t axiom_is_initialized(void *handle);

#ifdef __cplusplus
}
#endif

#endif
