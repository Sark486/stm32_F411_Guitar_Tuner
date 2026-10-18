#include "level.h"
#include <assert.h>
#include <math.h>

#define DC_BLOCKER_R  0.999f
#define WINDOW_MS     128U
#define FULL_SCALE    8388608.0f // 2^23 = 8 388 608 for 24-bit samples
#define LEVEL_FLOOR   1e-6f      // -120 dBFS: 20 * log10(1e-6) = -120

static uint32_t window_samples;
static float sum_sq;
static float x_prev;
static float y_prev;
static uint32_t n_total;

/* DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
 * https://www.dsprelated.com/freebooks/filters/DC_Blocker.html */
static float dc_blocker(float x, float x_prev, float y_prev)
{
  return x - x_prev + DC_BLOCKER_R * y_prev;
}

bool level_process(const int32_t *samples, uint32_t count, float *dbfs_out)
{
  if (window_samples == 0) return false; // uninitialized

  for (unsigned int i = 0; i < count; i++)
  {
    float x = (float)samples[i];
    y_prev = dc_blocker(x, x_prev, y_prev);
    x_prev = x;
    sum_sq += y_prev * y_prev;
    n_total++;
  }

  if (n_total >= window_samples)
  {
    float rms = sqrtf(sum_sq / (float)n_total);
    *dbfs_out = 20.0f * log10f(fmaxf(rms / FULL_SCALE, LEVEL_FLOOR)); // avoids -inf with zero input
    n_total = 0;
    sum_sq = 0.0f;
    return true;
  }
  return false;
}

void level_init(uint32_t sample_rate_hz)
{
  assert(sample_rate_hz != 0);
  window_samples = sample_rate_hz * WINDOW_MS / 1000;
  sum_sq  = 0.0f;
  x_prev  = 0.0f;
  y_prev  = 0.0f;
  n_total = 0U;
}
