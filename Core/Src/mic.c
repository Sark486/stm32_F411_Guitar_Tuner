#include "mic.h"
#include <stdint.h>

#define MIC_BUF_HALFWORDS        (MIC_HALFWORDS_PER_HALF * 2U)
#define MIC_DMA_SIZE             (MIC_BUF_HALFWORDS / 2U)  // HAL counts 32-bit samples

static uint16_t i2s_dma_buffer [MIC_BUF_HALFWORDS];
static volatile uint32_t callback_counter = 0U;
static uint32_t overrun_count = 0U;

static const uint16_t *mic_get_ready_half(void)
{
  static uint32_t last_returned = 0U;
  uint32_t count = callback_counter;

  uint32_t gap = count - last_returned;
  if (gap == 0)
  {
    return NULL;
  }
  if (gap > 1)
  {
    overrun_count += gap - 1;
  }

  last_returned = count;
  if (count % 2 == 1)
  {
    return &i2s_dma_buffer[0];
  }
  else 
  {
    return &i2s_dma_buffer[MIC_HALFWORDS_PER_HALF];
  }
}

HAL_StatusTypeDef mic_start(I2S_HandleTypeDef *hi2s)
{
  return HAL_I2S_Receive_DMA(hi2s, i2s_dma_buffer, MIC_DMA_SIZE);
}

void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
  if(hi2s->Instance == SPI2)
  {
      callback_counter++;
  }
}

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
  if(hi2s->Instance == SPI2)
  {
      callback_counter++;
  }
}

uint32_t mic_get_callback_counter(void)
{
  return callback_counter;
}

uint32_t mic_get_overrun_count(void)
{
  return overrun_count;
}

bool mic_convert_samples(int32_t *out)
{
  const uint16_t *half_buffer = mic_get_ready_half();
  if (half_buffer == NULL) return false;
  for (unsigned int i=0; i < MIC_FRAMES_PER_HALF; i++) {
    uint32_t word = ((uint32_t)half_buffer[MIC_HALFWORDS_PER_FRAME * i] << 16) 
                             | half_buffer[MIC_HALFWORDS_PER_FRAME * i + 1];
    out[i] = (int32_t)word >> 8;
  }
  return true;
}

uint32_t mic_get_sample_rate(I2S_HandleTypeDef *hi2s)
{
  return hi2s->Init.AudioFreq;
}