#ifndef MIC_H
#define MIC_H

#include "stm32f4xx_hal.h" // IWYU pragma: keep
#include <stdint.h>

#define MIC_FRAMES_PER_HALF      256U
#define MIC_HALFWORDS_PER_FRAME  4U   // 2 channels × 2 half-words (24-bit in 32-bit slot)
#define MIC_HALFWORDS_PER_HALF   (MIC_FRAMES_PER_HALF * MIC_HALFWORDS_PER_FRAME)

HAL_StatusTypeDef mic_start(I2S_HandleTypeDef *hi2s);
uint32_t mic_get_callback_counter(void);

/* Returns the half of the buffer DMA just finished filling (MIC_HALFWORDS_PER_HALF halfwords),
 * or NULL if nothing changed. Valid only until the next callback (~8 ms), will be overwritten after that. */
const uint16_t *mic_get_ready_half(void);
bool mic_convert_samples(int32_t *out);

#endif /* MIC_H */
