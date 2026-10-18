#include "mic.h"
#include <stdint.h>

#define MIC_HALFWORDS_PER_FRAME  4U   // 2 channels × 2 half-words (24-bit in 32-bit slot)
#define MIC_HALFWORDS_PER_HALF   (MIC_BLOCK_SAMPLES * MIC_HALFWORDS_PER_FRAME)
#define MIC_BUF_HALFWORDS        (MIC_HALFWORDS_PER_HALF * 2U)
#define MIC_DMA_SIZE             (MIC_BUF_HALFWORDS / 2U)  // HAL counts 32-bit samples

static uint16_t i2s_dma_buffer [MIC_BUF_HALFWORDS];
static volatile uint32_t blocks_filled = 0U;
static uint32_t overrun_count = 0U;

static const uint16_t *get_ready_half(void)
{
  static uint32_t last_consumed = 0U;
  uint32_t filled = blocks_filled;

  uint32_t pending = filled - last_consumed;
  if (pending == 0)
  {
    return NULL;
  }
  if (pending > 1)
  {
    overrun_count += pending - 1;
  }

  last_consumed = filled;
  if (filled % 2 == 1)
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
      blocks_filled++;
  }
}

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
  if(hi2s->Instance == SPI2)
  {
      blocks_filled++;
  }
}

uint32_t mic_get_block_count(void)
{
  return blocks_filled;
}

uint32_t mic_get_overrun_count(void)
{
  return overrun_count;
}

bool mic_read_block(int32_t *samples_out)
{
  const uint16_t *source = get_ready_half();
  if (source == NULL) return false;
  for (unsigned int i=0; i < MIC_BLOCK_SAMPLES; i++) {
    uint32_t raw = ((uint32_t)source[MIC_HALFWORDS_PER_FRAME * i] << 16) 
                             | source[MIC_HALFWORDS_PER_FRAME * i + 1];
    samples_out[i] = (int32_t)raw >> 8;
  }
  return true;
}

uint32_t mic_get_sample_rate_hz(I2S_HandleTypeDef *hi2s)
{
  return hi2s->Init.AudioFreq;
}
