#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pitch.h"            /* public interface, via the pitch target */
#define VECTOR_SIZE               32768U // vector for raw input data size, comes from vectors.txt
#define VECTOR_SIZE_DECIMATED     8192U // vector for decimated data size, comes from vectors.txt
#define VECTOR_SIZE_D             (PITCH_D_LEN * 3U) // vector for d and dprime data size, comes from vectors.txt, 3 rows
#define TOLERANCE_DECIMATED_ABS   1e-6f // decimated values absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_D_REL           1e-4f // difference values relative comparison tolerance, comes from vectors.txt
#define TOLERANCE_DPRIME_ABS      1e-4f // normalized difference values absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_TAU_ABS         1e-3f // YIN period (8 kHz samples) absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_CONFIDENCE_ABS  1e-4f // confidence (d' at the dip) absolute comparison tolerance, comes from vectors.txt
#define RESULTS_COUNT             29U   // results per case: (32768 - 4096) / 1024 + 1, rows in <case>_results.csv

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
  // reading input and golden standard file
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

  // test code begins
  pitch_result_t result;
  pitch_init(&state);
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    pitch_push(&state, tone440_input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result);
    
    for (size_t k = 0; k < PITCH_BLOCK_SIZE_DECIM; k++)
    {
      float actual = state.window[PITCH_WINDOW - PITCH_BLOCK_SIZE_DECIM + k];
      float expected = tone440_decimated[block * PITCH_BLOCK_SIZE_DECIM + k];
      bool equal = fabsf(actual - expected) <= TOLERANCE_DECIMATED_ABS;
      if (!equal)
      {
        printf("Block: %zu, index in block: %zu, actual: %.9g, expected: %.9g\n", block, k, actual, expected);
        failure_count++;
        return;
      }
    }
  }
}

static void d_cmndf_test(void)
{
  // reading input and golden standard files
  static int16_t tone440_input[VECTOR_SIZE];
  if (!read_file(VECTORS_DIR "/tone440_input.i16", tone440_input, sizeof(tone440_input[0]), VECTOR_SIZE))
  {
    failure_count++;
    return;
  }

  static float tone440_d[VECTOR_SIZE_D];
  if (!read_file(VECTORS_DIR "/tone440_d.f32", tone440_d, sizeof(tone440_d[0]), VECTOR_SIZE_D))
  {
    failure_count++;
    return;
  }

  static float tone440_dprime[VECTOR_SIZE_D];
  if (!read_file(VECTORS_DIR "/tone440_dprime.f32", tone440_dprime, sizeof(tone440_dprime[0]), VECTOR_SIZE_D))
  {
    failure_count++;
    return;
  }

  // test code begins
  pitch_result_t result;
  pitch_init(&state);
  size_t push_true_count = 0;
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    bool push_result = pitch_push(&state, tone440_input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result);
    if(push_result)
    {
      // checking frames 0, 14, 28
      if (push_true_count % 14 == 0)
      {
        size_t offset = (push_true_count / 14) * PITCH_D_LEN;
        for (size_t tau = 0; tau < PITCH_D_LEN; tau++)
        {
          float actual_d = state.d[tau];
          float expected_d = tone440_d[offset + tau];
          bool equal_d = fabsf(actual_d - expected_d) <= TOLERANCE_D_REL * fabsf(expected_d);
          if (!equal_d)
          {
            printf("D: Block: %zu, tau: %zu, actual: %.9g, expected: %.9g\n", block, tau, actual_d, expected_d);
            failure_count++;
          }

          float actual_dprime = state.dprime[tau];
          float expected_dprime = tone440_dprime[offset + tau];
          bool equal_dprime = fabsf(actual_dprime - expected_dprime) <= TOLERANCE_DPRIME_ABS;

          if (!equal_dprime)
          {
            printf("Dprime: Block: %zu, tau: %zu, actual: %.9g, expected: %.9g\n", block, tau, actual_dprime, expected_dprime);
            failure_count++;
          }
        }
      }
      push_true_count++;
    }
  }
  test_check(push_true_count == 29, "Should do 29 pushes that return true");
}

/*
  Runs one case through pitch_push and compares tau and confidence of every result with <case>_results.csv.
  CSV columns: frame,time_s,tau,freq_hz,confidence,valid; one header line.
*/
static void yin_case_test(const char *input_path, const char *results_path)
{
  // reading input and golden standard file
  static int16_t input[VECTOR_SIZE];
  if (!read_file(input_path, input, sizeof(input[0]), VECTOR_SIZE))
  {
    failure_count++;
    return;
  }

  FILE *fp = fopen(results_path, "r");
  if (fp == NULL)
  {
    printf("File open error: %s\n", results_path);
    failure_count++;
    return;
  }

  // skip the header line; the next %u skips the newline left behind
  if (fscanf(fp, "%*[^\n]") == EOF)
  {
    printf("Empty file: %s\n", results_path);
    failure_count++;
    fclose(fp);
  }

  // test code begins
  pitch_result_t result;
  pitch_init(&state);
  uint32_t result_count = 0U;
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    if (!pitch_push(&state, input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result))
    {
      continue;
    }

    uint32_t frame;
    float expected_tau;
    float expected_confidence;
    int expected_valid;
    // * in the format reads a field and discards it
    int fields = fscanf(fp, "%" SCNu32 ",%*f,%f,%*f,%f,%d", &frame, &expected_tau, &expected_confidence, &expected_valid);
    if (fields != 4)
    {
      printf("%s: bad or missing row for result %" PRIu32 " (fscanf returned %d)\n", results_path, result_count, fields);
      failure_count++;
      fclose(fp);
    }
    if (frame != result_count)
    {
      printf("%s: row frame %" PRIu32 ", expected %" PRIu32 "\n", results_path, frame, result_count);
      failure_count++;
      fclose(fp);
    }

    // tau is compared only on valid rows.
    if (expected_valid == 1 && fabsf(result.tau - expected_tau) > TOLERANCE_TAU_ABS)
    {
      printf("%s: frame %" PRIu32 ", tau actual: %.9g, expected: %.9g\n", results_path, frame, result.tau, expected_tau);
      failure_count++;
      fclose(fp);
    }
    if (fabsf(result.confidence - expected_confidence) > TOLERANCE_CONFIDENCE_ABS)
    {
      printf("%s: frame %" PRIu32 ", confidence actual: %.9g, expected: %.9g\n", results_path, frame, result.confidence, expected_confidence);
      failure_count++;
      fclose(fp);
    }
    result_count++;
  }

  test_check(result_count == RESULTS_COUNT, "yin_case_test: result count is not 29");
  test_check(fscanf(fp, "%*s") == EOF, "yin_case_test: results file has more rows than results");
}

int main(void)
{
  wrong_size_test();
  timing_test();
  sliding_test();
  decimator_test();
  d_cmndf_test();
  yin_case_test(VECTORS_DIR "/tone440_input.i16", VECTORS_DIR "/tone440_results.csv");
  yin_case_test(VECTORS_DIR "/silence_input.i16", VECTORS_DIR "/silence_results.csv");

  return failure_count == 0 ? 0 : 1;
}

