#include "capture.h"
#include "mic.h"
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "stm32f4xx_hal.h"

#define CAPTURE_TIMEOUT_BLOCKS        3U // allow 3 block periods
#define CAPTURE_TOTAL_SAMPLES_COUNT   32768U //(CAPTURE_BLOCK_COUNT * MIC_BLOCK_SAMPLES)
#define CAPTURE_BLOCK_COUNT           (CAPTURE_TOTAL_SAMPLES_COUNT / MIC_BLOCK_SAMPLES)
#define CAPTURE_TIMEOUT_MS            5000U // per half, 32 KB at 115200 takes ~3 s

static_assert(CAPTURE_TOTAL_SAMPLES_COUNT % MIC_BLOCK_SAMPLES == 0, "CAPTURE_TOTAL_SAMPLES_COUNT doesn't divide evenly by MIC_BLOCK_SAMPLES");

static UART_HandleTypeDef *uart_handle;
static int16_t capture_buf[CAPTURE_TOTAL_SAMPLES_COUNT];

static void convert_and_store(const int32_t *samples, unsigned int block_idx)
{
  for (unsigned int i = 0; i < MIC_BLOCK_SAMPLES; i++)
  {
    int16_t sample16 = (int16_t)(samples[i] >> 8); // keep the top 16 of 24 bits
    capture_buf[block_idx * MIC_BLOCK_SAMPLES + i] = sample16;
  }
}

static bool wait_for_block(int32_t *samples, uint32_t timeout_ms, uint32_t overruns_at_start)
{
  uint32_t start = HAL_GetTick();
  while (1)
  {
    bool got_block = mic_read_block(samples);
    if (overruns_at_start != mic_get_overrun_count()) 
    {
      return false;
    }
    if (got_block)
    {
      return true;
    }
    if (HAL_GetTick() - start >= timeout_ms)
    {
      return false;
    }
  }
}

static bool uart_send(const void *data, uint16_t len, uint32_t timeout_ms)
{
  return HAL_UART_Transmit(uart_handle, (const uint8_t *)data, len, timeout_ms) == HAL_OK;
}

void capture_init(UART_HandleTypeDef *huart)
{ 
  uart_handle = huart; 
}

const int16_t *capture_get_buffer()
{
  return capture_buf;
}

uint32_t capture_get_sample_count(void)
{
  return CAPTURE_TOTAL_SAMPLES_COUNT;
}

bool capture_record(void)
{
  uint32_t rate_hz = mic_get_sample_rate_hz();
  if (rate_hz < 1000U) return false;
  uint32_t samples_per_ms = rate_hz / 1000U;
  uint32_t timeout_ms = CAPTURE_TIMEOUT_BLOCKS * MIC_BLOCK_SAMPLES / samples_per_ms;
  static int32_t samples[MIC_BLOCK_SAMPLES];

  mic_resync();

  uint32_t overruns_at_start = mic_get_overrun_count();
  for (unsigned int i = 0; i < CAPTURE_BLOCK_COUNT; i++)
  {
    if (!wait_for_block(samples, timeout_ms, overruns_at_start))
    {
      return false;
    }
    convert_and_store(samples, i);
  }

  return true;
}

bool capture_send(void)
{
  if (uart_handle == NULL) return false;
  
  char line[32];   /* longest possible: "DUMP 4294967295 4294967295\r\n" = 28 chars + '\0' */
  int len = snprintf(line, sizeof line, "DUMP %" PRIu32 " %" PRIu32 "\r\n",
                     (uint32_t)CAPTURE_TOTAL_SAMPLES_COUNT, mic_get_sample_rate_hz());
  if (len < 0 || (size_t)len >= sizeof line)
  {
    return false;
  }
  if (!uart_send(line, (uint16_t)len, 100U))
  {
    return false;
  }

  bool result = true;
  uint16_t byte_chunk_size = CAPTURE_TOTAL_SAMPLES_COUNT * sizeof(int16_t) / 2;
  result = result && uart_send((const uint8_t *)capture_buf, byte_chunk_size, CAPTURE_TIMEOUT_MS);
  result = result && uart_send((const uint8_t *)capture_buf + byte_chunk_size, byte_chunk_size, CAPTURE_TIMEOUT_MS);

  static const char end_line[] = "END\r\n";
  if (!uart_send(end_line, sizeof end_line - 1U, 100U))
  {
    return false;
  }

  mic_resync();
  return result;
}

bool capture_receive(void)
{
  if (uart_handle == NULL) return false;

  // discard bytes that arrived before the command
  uint8_t discard;
  while (HAL_UART_Receive(uart_handle, &discard, 1, 0) == HAL_OK)
  {
    // ignore it
  }

  char line[32];   // longest possible: "UPLOAD OK -2147483648\r\n" = 23 chars + '\0'
  int len = snprintf(line, sizeof line, "READY %" PRIu32 "\r\n", (uint32_t)CAPTURE_TOTAL_SAMPLES_COUNT);
  if (len < 0 || (size_t)len >= sizeof line)
  {
    return false;
  }
  // PC waits for this line before sending
  if (!uart_send(line, (uint16_t)len, 100U))
  {
    return false;
  }

  // two halves like in capture_send, 65536 bytes don't fit into uint16_t
  uint16_t byte_chunk_size = CAPTURE_TOTAL_SAMPLES_COUNT * sizeof(int16_t) / 2;
  bool result = HAL_UART_Receive(uart_handle, (uint8_t *)capture_buf, byte_chunk_size, CAPTURE_TIMEOUT_MS) == HAL_OK;
  result = result && HAL_UART_Receive(uart_handle, (uint8_t *)capture_buf + byte_chunk_size, byte_chunk_size, CAPTURE_TIMEOUT_MS) == HAL_OK;

  if (result)
  {
    // sum of samples so the PC can check the upload
    int32_t sum = 0;
    for (unsigned int i = 0; i < CAPTURE_TOTAL_SAMPLES_COUNT; i++)
    {
      sum += capture_buf[i];
    }
    len = snprintf(line, sizeof line, "UPLOAD OK %" PRId32 "\r\n", sum);
  }
  else
  {
    len = snprintf(line, sizeof line, "UPLOAD FAILED\r\n");
  }
  if (len > 0 && (size_t)len < sizeof line)
  {
    uart_send(line, (uint16_t)len, 100U);
  }

  mic_resync();
  return result;
}
