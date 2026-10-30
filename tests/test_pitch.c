#include <stdio.h>

#include "pitch.h"          /* public interface, via the pitch target */
#include "pitch_stages.h"   /* internal header, via lib/pitch/src */

static uint16_t failure_count = 0U;
static pitch_t state;

static void test_check(bool result, const char *description)
{
  if(!result)
  {
    failure_count++;
    printf("FAIL: %s\n", description);
  }
}

static void timing_test(void)
{
  int16_t block[PITCH_BLOCK_SIZE] = {};
  pitch_result_t result;
  bool pitch_result = false;
  pitch_init(&state);
  for (int i = 1; i <= 20; i++)
  {
    pitch_result = pitch_push(&state, block, PITCH_BLOCK_SIZE, &result);
    if (i <= 15)
    {
      test_check(!pitch_result, "returned true for the pushes 1-15");
    }
    if (i == 16)
    {
      test_check(pitch_result, "returned false for push 16");
      test_check(result.sample_count == 4096, "sample_count is not 4096 after push 16"); // 4096 = 16 * 256 (PITCH_BLOCK_SIZE) samples
    }
    if (i > 16 && i < 20)
    {
      test_check(!pitch_result, "returned true for the pushes 17-19");
    }
    if (i == 20)
    {
      test_check(pitch_result, "returned false for push 20");
      test_check(result.sample_count == 5120, "sample_count is not 5120 after push 20"); // 5120 = 20 * 256 (PITCH_BLOCK_SIZE) samples
    }
  }
 
}

static void sliding_test(void)
{
  int16_t block[PITCH_BLOCK_SIZE] = {};
  pitch_result_t result;
  pitch_init(&state);
  for (int i = 1; i <= 20; i++)
  {
    for (unsigned int j = 0; j < PITCH_BLOCK_SIZE; j++)
    {
      block[j] = (int16_t)i;
    }
    pitch_push(&state, block, PITCH_BLOCK_SIZE, &result);
  }
  test_check(state.raw_window[0] == 5, "first element of raw window is not 5");
  test_check(state.raw_window[PITCH_BLOCK_SIZE - 1] == 5, "last element of the first block of raw window is not 5");
  test_check(state.raw_window[PITCH_BLOCK_SIZE] == 6, "first element of the second block of raw window is not 6");
  test_check(state.raw_window[PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE - 1] == 19, "last element of the pre-last block of raw window is not 19");
  test_check(state.raw_window[PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE] == 20, "first element of the last block of raw window is not 20");
  test_check(state.raw_window[PITCH_RAW_WINDOW - 1] == 20, "last element of raw window is not 20");
}

static void wrong_size_test(void)
{
  int16_t block[PITCH_BLOCK_SIZE] = {};
  pitch_result_t result;
  pitch_init(&state);

  bool pitch_result = pitch_push(&state, block, PITCH_BLOCK_SIZE - 1, &result);
  test_check(!pitch_result, "pitch_push accepted a wrong block size");
  test_check(state.samples_in == 0, "samples_in is not 0");
}

int main(void)
{
  wrong_size_test();
  timing_test();
  sliding_test();

  return failure_count == 0 ? 0 : 1;
}

