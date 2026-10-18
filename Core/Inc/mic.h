#ifndef MIC_H
#define MIC_H

#include "stm32f4xx_hal.h" // IWYU pragma: keep
#include <stdint.h>

#define MIC_FRAMES_PER_HALF      256U
#define MIC_HALFWORDS_PER_FRAME  4U   // 2 channels × 2 half-words (24-bit in 32-bit slot)
#define MIC_HALFWORDS_PER_HALF   (MIC_FRAMES_PER_HALF * MIC_HALFWORDS_PER_FRAME)

HAL_StatusTypeDef mic_start(I2S_HandleTypeDef *hi2s);
uint32_t mic_get_callback_counter(void);
uint32_t mic_get_overrun_count(void);

bool mic_convert_samples(int32_t *out);
uint32_t mic_get_sample_rate(I2S_HandleTypeDef *hi2s);

#endif /* MIC_H */
