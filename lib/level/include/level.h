#ifndef LEVEL_H
#define LEVEL_H

#include <stdbool.h>
#include <stdint.h>

bool level_process(const int32_t *samples, uint32_t count, float *dbfs_out);
void level_init(uint32_t sample_rate_hz);

#endif /* LEVEL_H */
