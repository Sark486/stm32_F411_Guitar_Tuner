#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pitch.h"            /* public interface, via the pitch target */
#define VECTOR_SIZE               32768U // vector for raw input data size, comes from vectors.txt
#define VECTOR_SIZE_DECIMATED     8192U // vector for decimated data size, comes from vectors.txt
#define VECTOR_SIZE_D             (PITCH_D_LEN * D_ROWS) // vector for d and dprime data size, comes from vectors.txt, 3 rows
#define TOLERANCE_DECIMATED_ABS   1e-6f // decimated values absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_D_REL           1e-4f // difference values relative comparison tolerance, comes from vectors.txt
#define TOLERANCE_DPRIME_ABS      1e-4f // normalized difference values absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_TAU_ABS         1e-3f // YIN period (8 kHz samples) absolute comparison tolerance, comes from vectors.txt
#define TOLERANCE_CONFIDENCE_ABS  1e-4f // confidence (d' at the dip) absolute comparison tolerance, comes from vectors.txt
#define RESULTS_COUNT             29U   // results per case: (32768 - 4096) / 1024 + 1, rows in <case>_results.csv
#define TOLERANCE_FREQ_CENTS      0.01f  // frequency comparison tolerance in cents, comes from vectors.txt
#define D_ROWS                    3U    // frames with d and d' saved per case: 0, 14, 28 (first, middle, last result)
#define PATH_SIZE                 512U  // buffer for one vector file path; VECTORS_DIR is absolute, so make_path checks for truncation

// golden cases from the [cases] section of vectors.txt
static const char *const cases[] = {
  "tone440", "e2_open", "e2_fret12", "e3_harmonic", "e6_fret24",
  "e1_bass", "b0_bass", "e1_missing", "silence",
};
#define CASES_COUNT               (sizeof cases / sizeof cases[0])

// result index of each saved d / d' row, in file order
static const uint32_t d_frames[D_ROWS] = {0U, 14U, 28U};

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

/*
  Builds VECTORS_DIR "/" case_name suffix into buf, for example "vectors/tone440_input.i16".
*/
static bool make_path(char *buf, size_t size, const char *case_name, const char *suffix)
{
  int needed = snprintf(buf, size, "%s/%s%s", VECTORS_DIR, case_name, suffix);
  if (needed < 0 || (size_t)needed >= size)
  {
    printf("%s: path too long or format error for %s\n", case_name, suffix);
    return false;
  }
  return true;
}

/*
  Reads <case><suffix> into dest. Counts a failure and returns false on any error.
*/
static bool load_vector(const char *case_name, const char *suffix, void *dest, size_t elem_size, size_t count)
{
  char path[PATH_SIZE];
  if (!make_path(path, sizeof path, case_name, suffix) || !read_file(path, dest, elem_size, count))
  {
    failure_count++;
    return false;
  }
  return true;
}

/*
  Decimator output against <case>_decimated.f32.
*/
static void decimator_test(const char *case_name)
{
  static int16_t input[VECTOR_SIZE];
  static float decimated[VECTOR_SIZE_DECIMATED];
  if (!load_vector(case_name, "_input.i16", input, sizeof(input[0]), VECTOR_SIZE) ||
      !load_vector(case_name, "_decimated.f32", decimated, sizeof(decimated[0]), VECTOR_SIZE_DECIMATED))
  {
    return;
  }

  pitch_result_t result;
  pitch_init(&state);
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    pitch_push(&state, input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result);

    for (size_t k = 0; k < PITCH_BLOCK_SIZE_DECIM; k++)
    {
      float actual = state.window[PITCH_WINDOW - PITCH_BLOCK_SIZE_DECIM + k];
      float expected = decimated[block * PITCH_BLOCK_SIZE_DECIM + k];
      if (fabsf(actual - expected) > TOLERANCE_DECIMATED_ABS)
      {
        printf("%s: decimated, block %zu, index in block %zu, actual: %.9g, expected: %.9g\n", case_name, block, k, actual, expected);
        failure_count++;
        return;
      }
    }
  }
}

/*
  d and d' at the saved frames (d_frames) against <case>_d.f32 / <case>_dprime.f32, one row per frame.
*/
static void d_cmndf_test(const char *case_name)
{
  static int16_t input[VECTOR_SIZE];
  static float d_expected[VECTOR_SIZE_D];
  static float dprime_expected[VECTOR_SIZE_D];
  if (!load_vector(case_name, "_input.i16", input, sizeof(input[0]), VECTOR_SIZE) ||
      !load_vector(case_name, "_d.f32", d_expected, sizeof(d_expected[0]), VECTOR_SIZE_D) ||
      !load_vector(case_name, "_dprime.f32", dprime_expected, sizeof(dprime_expected[0]), VECTOR_SIZE_D))
  {
    return;
  }

  pitch_result_t result;
  pitch_init(&state);
  uint32_t result_count = 0U;
  size_t row = 0U;                                   // next row of the d / d' files to compare
  for (size_t block = 0; block < VECTOR_SIZE / PITCH_BLOCK_SIZE; block++)
  {
    if (!pitch_push(&state, input + block * PITCH_BLOCK_SIZE, PITCH_BLOCK_SIZE, &result))
    {
      continue;
    }

    if (row < D_ROWS && result_count == d_frames[row])
    {
      size_t offset = row * PITCH_D_LEN;
      for (size_t tau = 0; tau < PITCH_D_LEN; tau++)
      {
        float actual_d = state.d[tau];
        float expected_d = d_expected[offset + tau];
        if (fabsf(actual_d - expected_d) > TOLERANCE_D_REL * fabsf(expected_d))
        {
          printf("%s: d, frame %" PRIu32 ", tau %zu, actual: %.9g, expected: %.9g\n", case_name, result_count, tau, actual_d, expected_d);
          failure_count++;
          return;
        }

        float actual_dprime = state.dprime[tau];
        float expected_dprime = dprime_expected[offset + tau];
        if (fabsf(actual_dprime - expected_dprime) > TOLERANCE_DPRIME_ABS)
        {
          printf("%s: d', frame %" PRIu32 ", tau %zu, actual: %.9g, expected: %.9g\n", case_name, result_count, tau, actual_dprime, expected_dprime);
          failure_count++;
          return;
        }
      }
      row++;
    }
    result_count++;
  }

  if (result_count != RESULTS_COUNT || row != D_ROWS)
  {
    printf("%s: d_cmndf_test: %" PRIu32 " results, %zu d rows compared; expected %u and %u\n", case_name, result_count, row, RESULTS_COUNT, D_ROWS);
    failure_count++;
  }
}

/*
  Every result against <case>_results.csv: valid on every row; tau and frequency on valid rows; confidence on every row.
  CSV columns: frame,time_s,tau,freq_hz,confidence,valid; one header line.
*/
static void yin_case_test(const char *case_name)
{
  static int16_t input[VECTOR_SIZE];
  if (!load_vector(case_name, "_input.i16", input, sizeof(input[0]), VECTOR_SIZE))
  {
    return;
  }

  char results_path[PATH_SIZE];
  if (!make_path(results_path, sizeof results_path, case_name, "_results.csv"))
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
    printf("%s: empty results file\n", case_name);
    failure_count++;
    fclose(fp);
    return;
  }

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
    float expected_freq;
    float expected_confidence;
    int expected_valid;
    // * in the format reads a field and discards it
    int fields = fscanf(fp, "%" SCNu32 ",%*f,%f,%f,%f,%d", &frame, &expected_tau, &expected_freq, &expected_confidence, &expected_valid);
    if (fields != 5)
    {
      printf("%s: bad or missing row for result %" PRIu32 " (fscanf returned %d)\n", case_name, result_count, fields);
      failure_count++;
      fclose(fp);
      return;
    }
    if (frame != result_count)
    {
      printf("%s: row frame %" PRIu32 ", expected %" PRIu32 "\n", case_name, frame, result_count);
      failure_count++;
      fclose(fp);
      return;
    }

    // valid is compared on every row: on silence every result must come out invalid
    if (result.valid != (expected_valid == 1))
    {
      printf("%s: frame %" PRIu32 ", valid actual: %d, expected: %d\n", case_name, frame, (int)result.valid, expected_valid);
      failure_count++;
      fclose(fp);
      return;
    }

    // tau and frequency only on valid rows: on invalid ones the parabola can jump far away (see 6d)
    if (expected_valid == 1 && fabsf(result.tau - expected_tau) > TOLERANCE_TAU_ABS)
    {
      printf("%s: frame %" PRIu32 ", tau actual: %.9g, expected: %.9g\n", case_name, frame, result.tau, expected_tau);
      failure_count++;
      fclose(fp);
      return;
    }

    if (expected_valid == 1 && fabsf(1200.0f * log2f(result.freq_hz / expected_freq)) > TOLERANCE_FREQ_CENTS)
    {
      printf("%s: frame %" PRIu32 ", freq actual: %.9g, expected: %.9g\n", case_name, frame, result.freq_hz, expected_freq);
      failure_count++;
      fclose(fp);
      return;
    }

    if (fabsf(result.confidence - expected_confidence) > TOLERANCE_CONFIDENCE_ABS)
    {
      printf("%s: frame %" PRIu32 ", confidence actual: %.9g, expected: %.9g\n", case_name, frame, result.confidence, expected_confidence);
      failure_count++;
      fclose(fp);
      return;
    }
    result_count++;
  }

  if (result_count != RESULTS_COUNT)
  {
    printf("%s: %" PRIu32 " results, expected %u\n", case_name, result_count, RESULTS_COUNT);
    failure_count++;
  }
  if (fscanf(fp, "%*s") != EOF)
  {
    printf("%s: results file has more rows than results\n", case_name);
    failure_count++;
  }
  fclose(fp);
}

int main(void)
{
  wrong_size_test();
  timing_test();
  sliding_test();

  for (size_t i = 0; i < CASES_COUNT; i++)
  {
    decimator_test(cases[i]);
    d_cmndf_test(cases[i]);
    yin_case_test(cases[i]);
  }

  return failure_count == 0 ? 0 : 1;
}
