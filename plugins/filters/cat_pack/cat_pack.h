#ifndef CAT_PACK_H
#define CAT_PACK_H

#include <stdint.h>
#include "blosc2.h"

#ifdef __cplusplus
extern "C" {
#endif

int cat_pack_forward(const uint8_t *src, uint8_t *dest, int32_t length, uint8_t meta, blosc2_cparams *cparams, uint8_t id);
int cat_pack_backward(const uint8_t *src, uint8_t *dest, int32_t length, uint8_t meta, blosc2_dparams *dparams, uint8_t id);

#ifdef __cplusplus
}
#endif

#endif // CAT_PACK_H
