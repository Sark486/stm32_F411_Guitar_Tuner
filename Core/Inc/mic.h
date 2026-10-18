#ifndef MIC_H
#define MIC_H

#include "stm32f4xx_hal.h" // IWYU pragma: keep
#include <stdint.h>

#define MIC_BLOCK_SAMPLES        256U

HAL_StatusTypeDef mic_start(I2S_HandleTypeDef *hi2s);
uint32_t mic_get_block_count(void);
uint32_t mic_get_overrun_count(void);

bool mic_read_block(int32_t *samples_out);
uint32_t mic_get_sample_rate_hz(I2S_HandleTypeDef *hi2s);

#endif /* MIC_H */
