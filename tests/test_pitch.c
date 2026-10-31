#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pitch.h"          /* public interface, via the pitch target */
#define VECTOR_SIZE           32768U // vector for raw input data size, comes from vectors.txt
#define VECTOR_SIZE_DECIMATED 8192U // vector for decimated data size, comes from vectors.txt
#define TOLERANCE             1e-6f // decimated values comparison tolerance, comes from vectors.txt

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

static bool read_file(const char *path, void *dest, size_t elem_size, size_t exp_cnt)
{ 
  FILE* fp = fopen(path, "rb");
  if (fp == NULL)
  {
    printf("File open error: %s\n", path);
    return false;
  }
  size_t actual_cnt = fread(dest, elem_size, exp_cnt, fp);
  if (exp_cnt != actual_cnt)
  {
    printf("File read error: %s, elements expected: %zu, actual: %zu\n", path, exp_cnt, actual_cnt);
    fclose(fp);
    return false;
  }
  fclose(fp);
  return true;
}

static void timing_test(void)
{
  int16_t block[PITCH_BLOCK_SIZE] = {};
  pitch_result_t result;
  bool pitch_result = false;
  pitch_init(&state);
  for (size_t i = 1; i <= 20; i++)
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
  for (size_t i = 1; i <= 20; i++)
  {
    for (size_t j = 0; j < PITCH_BLOCK_SIZE; j++)
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


static void decimator_test(void)
{
  static int16_t tone440_input[VECTOR_SIZE];
  if (!read_file(VECTORS_DIR "/tone440_input.i16", tone440_input, sizeof(tone440_input[0]), VECTOR_SIZE))
  {
    failure_count++;
    return;
  }

  static float tone440_decimated[VECTOR_SIZE_DECIMATED];
  if (!read_file(VECTORS_DIR "/tone440_decimated.f32", tone440_decimated, sizeof(tone440_decimated[0]), VECTOR_SIZE_DECIMATED))
  {
    failure_count++;
    return;
  }

  pitch_result_t result;
  pitch_init(&state);
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    pitch_push(&state, tone440_input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result);
    
    for (size_t k = 0; k < PITCH_BLOCK_SIZE_DECIM; k++)
    {
      float actual = state.window[PITCH_WINDOW - PITCH_BLOCK_SIZE_DECIM + k];
      float expected = tone440_decimated[block * PITCH_BLOCK_SIZE_DECIM + k];
      bool equal = fabsf(actual - expected) <= TOLERANCE;
      if (!equal)
      {
        printf("Block: %zu, index in block: %zu, actual: %.9g, expected: %.9g\n", block, k, actual, expected);
        failure_count++;
        return;
      }
    }
    
  }
  
}

int main(void)
{
  wrong_size_test();
  timing_test();
  sliding_test();
  decimator_test();

  return failure_count == 0 ? 0 : 1;
}

