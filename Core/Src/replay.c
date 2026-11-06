#include "replay.h"
#include "main.h"
#include "cycles.h"
#include "pitch.h"
#include <inttypes.h>
#include <stdio.h>

#define REPLAY_MAX_PUSHES    128U  // whole capture buffer: 32768 samples / 256 per push
#define REPLAY_MAX_RESULTS   32U   // full buffer gives 29: first one after push 16, then every 4th push

static pitch_t state;
static uint32_t push_cycles[REPLAY_MAX_PUSHES];
static bool had_result[REPLAY_MAX_PUSHES]; // true if a result for this push was saved in results
static pitch_result_t results[REPLAY_MAX_RESULTS];

// min and max cycles separately for pushes with and without result
static void print_summary(unsigned int pushes)
{
  uint32_t min_plain = UINT32_MAX;
  uint32_t max_plain = 0U;
  uint32_t min_result = UINT32_MAX;
  uint32_t max_result = 0U;

  for (unsigned int i = 0; i < pushes; i++)
  {
    uint32_t c = push_cycles[i];
    if (had_result[i])
    {
      if (c < min_result) min_result = c;
      if (c > max_result) max_result = c;
    }
    else
    {
      if (c < min_plain) min_plain = c;
      if (c > max_plain) max_plain = c;
    }
  }
  printf("S,%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 "\r\n", min_plain, max_plain, min_result, max_result);
}

void replay_run(const int16_t *samples, uint32_t count)
{
  uint32_t pushes = count / PITCH_BLOCK_SIZE;
  if (pushes > REPLAY_MAX_PUSHES)
  {
    pushes = REPLAY_MAX_PUSHES;
  }

  // fresh state for every replay
  pitch_init(&state);
  uint32_t result_count = 0U;

  for (unsigned int i = 0; i < pushes; i++)
  {
    pitch_result_t result;

    HAL_GPIO_WritePin(MARK1_GPIO_Port, MARK1_Pin, GPIO_PIN_SET);
    uint32_t start = cycles_get();
    bool got_result = pitch_push(&state, samples + i * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result);
    uint32_t end = cycles_get();
    HAL_GPIO_WritePin(MARK1_GPIO_Port, MARK1_Pin, GPIO_PIN_RESET);

    push_cycles[i] = end - start;
    had_result[i] = false;
    if (got_result && result_count < REPLAY_MAX_RESULTS)
    {
      results[result_count] = result;
      result_count++;
      had_result[i] = true;
    }
  }

  printf("REPLAY %" PRIu32 " %" PRIu32 "\r\n", pushes, result_count);
  unsigned int r = 0;
  for (unsigned int i = 0; i < pushes; i++)
  {
    printf("P,%u,%" PRIu32 "\r\n", i, push_cycles[i]);
    if (had_result[i])
    {
      const pitch_result_t *p = &results[r];
      printf("R,%u,%" PRIu32 ",%.9g,%.9g,%.9g,%d\r\n", i, p->sample_count,
             (double)p->freq_hz, (double)p->confidence, (double)p->tau, p->valid);
      r++;
    }
  }
  print_summary(pushes);
  printf("END\r\n");
}
