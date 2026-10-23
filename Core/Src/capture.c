#include "capture.h"
#include "mic.h"
#include <stdint.h>
#include "stm32f4xx_hal.h" 

#define CAPTURE_HALVES_COUNT  128U

static int16_t capture_buf[CAPTURE_HALVES_COUNT * MIC_BLOCK_SAMPLES];

static void convert_and_store(const int32_t *samples, unsigned int pos)
{
  for (unsigned int i = 0; i < MIC_BLOCK_SAMPLES; i++) {
    int16_t raw = (int16_t)(samples[i] >> 8);
    capture_buf[pos * MIC_BLOCK_SAMPLES + i] = raw;
  }
}

bool capture_dump(void)
{
  uint32_t rate_hz = mic_get_sample_rate_hz();
  if (rate_hz < 1000U) return false;
  uint32_t timeout_ms = 3U * MIC_BLOCK_SAMPLES / (rate_hz / 1000);
  static int32_t samples[MIC_BLOCK_SAMPLES];
  mic_resync();
  uint32_t overrun_count = mic_get_overrun_count();
  for (unsigned int i = 0; i < CAPTURE_HALVES_COUNT; i++)
  {
    uint32_t start = HAL_GetTick();
    while(1)
    {
      bool got_block = mic_read_block(samples);
      if (overrun_count != mic_get_overrun_count()) {
        return false;
      }
      if (got_block)
      {       
        break;
      }
      if (HAL_GetTick() - start >= timeout_ms)
      {
        return false;
      }
    }
    convert_and_store(samples, i);
  }
  mic_resync();
  return true;
}


