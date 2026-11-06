#include "pitch.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fir_taps.h"

#define FULL_SCALE     32768.0f // 2^15 = 32 768 for 16-bit samples
#define UNROLL_FACTOR  4U // loop unrolling factor

static_assert(PITCH_FIR_TAPS == FIR_NUM_TAPS, "Number of taps is different between pitch.h and fir_taps.h");
static_assert(PITCH_BLOCK_SIZE % PITCH_DECIM == 0, "PITCH_BLOCK_SIZE doesn't divide evenly by PITCH_DECIM");
static_assert(PITCH_HOP % PITCH_BLOCK_SIZE_DECIM == 0, "PITCH_HOP doesn't divide evenly by samples count after decimation");

typedef struct {
  float  tau;       // c + parabola fraction, in 8 kHz samples
  float  confidence; // d'[tau]
  bool   in_range;  // false if the dip runs off either end of the search range
} yin_estimate_t;

/*
  Sliding the raw window to the left by PITCH_BLOCK_SIZE and adding new block to freed space on the right
*/
static void slide_raw_window(int16_t *raw_window, const int16_t *new_block)
{
  memmove(raw_window, raw_window + PITCH_BLOCK_SIZE, (PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE) * sizeof(*raw_window));
  memcpy(raw_window + (PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE), new_block, PITCH_BLOCK_SIZE * sizeof(*raw_window));
}

static void decimate(float *fir_delay, float *window, const int16_t *new_block)
{
  // filling fir_delay from (PITCH_FIR_TAPS − 1) to the end with the scaled values from the block
  for (unsigned int i = 0; i < PITCH_BLOCK_SIZE; i++)
  {
    fir_delay[PITCH_FIR_TAPS - 1 + i] = ((float)new_block[i]) / FULL_SCALE;
  }

  // sliding the window by PITCH_BLOCK_SIZE_DECIM before adding anything
  memmove(window, window + PITCH_BLOCK_SIZE_DECIM, (PITCH_WINDOW - PITCH_BLOCK_SIZE_DECIM) * sizeof(*window));

  // computing PITCH_BLOCK_SIZE_DECIM decimated values for the window
  // FIR filter https://en.wikipedia.org/wiki/Finite_impulse_response
  for (unsigned int k = 0; k < PITCH_BLOCK_SIZE_DECIM; k++)
  {
    // the last sample of the k-th group of PITCH_DECIM.
    unsigned int newest_ind = (PITCH_FIR_TAPS - 1) + PITCH_DECIM * k + (PITCH_DECIM - 1);
    float result = 0.0f;
    for (unsigned int j=0; j < PITCH_FIR_TAPS; j++)
    {
      result += fir_taps[j] * fir_delay[newest_ind - j];
    }
    // adding new decimated values to the window
    window[PITCH_WINDOW - PITCH_BLOCK_SIZE_DECIM + k] = result;
  }
  // keeping the last (PITCH_FIR_TAPS - 1) samples for the next time
  memmove(fir_delay, fir_delay + PITCH_BLOCK_SIZE, (PITCH_FIR_TAPS - 1) * sizeof(*fir_delay));
}

//calculating difference between the one and frame and another, shfted by tau
static void difference(const float *window, float *d)
{
  for (size_t tau = 0; tau < PITCH_D_LEN; tau++)
  {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (size_t i = 0; i + UNROLL_FACTOR <= PITCH_YIN_W; i += UNROLL_FACTOR)
    {
      float difference = window[i] - window[i + tau];
      sum0 += difference * difference;
      difference = window[i + 1] - window[i + 1 + tau];
      sum1 += difference * difference;
      difference = window[i + 2] - window[i + 2 + tau];
      sum2 += difference * difference;
      difference = window[i + 3] - window[i + 3 + tau];
      sum3 += difference * difference;
    }
    // remainder
    for (size_t i = PITCH_YIN_W - (PITCH_YIN_W % UNROLL_FACTOR); i < PITCH_YIN_W; i++)
    {
      float difference = window[i] - window[i + tau];
      sum0 += difference * difference;
    }
    d[tau] = sum0 + sum1 + sum2 + sum3;
  }
}

//calculating normalized difference between the one and frame and another, shfted by tau; Cumulative Mean Normalized Difference Function
static void cmndf(const float *d, float *dprime)
{
  dprime[0] = 1.0f;
  float sum = 0.0f;
  for (size_t tau = 1; tau < PITCH_D_LEN; tau++)
  {
    sum += d[tau];
    if (sum == 0.0f)
    {
      dprime[tau] = 1.0f;
    } else 
    {
      dprime[tau] = (d[tau] * ((float)tau)) / sum;
    }
  }
}
static float parabola(float a, float b, float c){
  float denominator = (a - 2.0f * b + c);
  return (denominator == 0.0f) ? 0.0f : 0.5f * (a - c) / denominator;
}

/*
  finding estimation on decimated sample rate
*/
static yin_estimate_t yin(const float *d, const float *dprime)
{

  yin_estimate_t est;
  size_t tau = PITCH_TAU_MIN;
  bool found = false;

  // first lag under threshold, then moving right until the bottom of the dip
  for (size_t t = PITCH_TAU_MIN; t <= PITCH_TAU_MAX; t++)
  {
    if(dprime[t] < PITCH_YIN_THRESHOLD)
    {
      while(t + 1U <= PITCH_TAU_MAX && dprime[t + 1] < dprime[t])
      {
        t++;
      }
      tau = t;
      found = true;
      break;
    }
  }

  // fallback, finding minimum in d'
  if(!found)
  {
    // tau = PITCH_TAU_MIN;
    for (size_t t = PITCH_TAU_MIN + 1U; t <= PITCH_TAU_MAX; t++)
    {
      if (dprime[t] < dprime[tau])
      {
        tau = t;
      }
    }
  }

  est.confidence = dprime[tau];
  
  est.in_range = !((tau == PITCH_TAU_MIN && dprime[tau - 1U] < dprime[tau]) ||
                   (tau == PITCH_TAU_MAX && dprime[tau + 1U] < dprime[tau]));

  // parabola
  // ceneter of the parabola should be on the minimum among tau-1, tau, tau+1
  size_t c = tau - 1U;
  if (d[tau] < d[c])
  {
    c = tau;
  }
  if (d[tau + 1U] < d[c])
  {
    c = tau + 1U;
  }
  if (c > PITCH_D_LEN - 2U)
  {
    c = PITCH_D_LEN - 2U;
  }

  float a = d[c - 1U];
  float b = d[c];
  float e = d[c + 1U];
  float offset = parabola(a, b, e);
  
  est.tau = (float)c + offset;
  return est;
}

/*
  finding more precise period on the full sample rate
*/
static float refine(const int16_t *raw_window, float *d_refine, float tau_coarse)
{
  // lag range
  long center = lroundf(tau_coarse);
  long first = center - (long)(PITCH_REFINE_SEARCH + 1);
  first = first >= 2 ? first : 2;
  long last = center + (long)(PITCH_REFINE_SEARCH + 1);
  last = last <= (long)PITCH_REFINE_MAX_LAG + 1 ? last : (long)PITCH_REFINE_MAX_LAG + 1;
  long count = last - first + 1;
  if (count < 3)
  {
    return tau_coarse;
  }

  size_t first_lag = (size_t)first;
  size_t n = (size_t)count;

  for (size_t i = 0; i < n; i++)
  {
    size_t t = first_lag + i;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (size_t j = 0; j + UNROLL_FACTOR <= PITCH_REFINE_W; j += UNROLL_FACTOR)
    {
      float difference = raw_window[j] - raw_window[j + t];
      sum0 += difference * difference;
      difference = raw_window[j + 1] - raw_window[j + 1 + t];
      sum1 += difference * difference;
      difference = raw_window[j + 2] - raw_window[j + 2 + t];
      sum2 += difference * difference;
      difference = raw_window[j + 3] - raw_window[j + 3 + t];
      sum3 += difference * difference;
    }
    for (size_t j = PITCH_REFINE_W - (PITCH_REFINE_W % UNROLL_FACTOR); j < PITCH_REFINE_W; j++)
    {
      float difference = raw_window[j] - raw_window[j + t];
      sum0 += difference * difference;
    }
    d_refine[i] = sum0 + sum1 + sum2 + sum3;
  }



  size_t min_ind = 1U;
  for (size_t i = 2; i < n - 1; i++)
  {
    if (d_refine[i] < d_refine[min_ind])
    {
      min_ind = i;
    } 
  }
  if ((d_refine[min_ind] > d_refine[min_ind - 1]) || (d_refine[min_ind] > d_refine[min_ind + 1]))
  {
    return tau_coarse;
  }
  return (float)(first_lag + min_ind) + parabola(d_refine[min_ind-1], d_refine[min_ind], d_refine[min_ind+1]);
}


void pitch_init(pitch_t *state)
{
  memset(state, 0, sizeof(*state));
}


/*
  Pushing block of raw samples into the raw_window, doing decimation and pushing result to the window.
*/
bool pitch_push(pitch_t *state, const int16_t *block, uint32_t count, pitch_result_t *out)
{
  if(count != PITCH_BLOCK_SIZE)
  {
    return false;
  }
  state->samples_in += PITCH_BLOCK_SIZE;
  slide_raw_window(state->raw_window, block);
  decimate(state->fir_delay, state->window, block);

  // filling until full, never more than PITCH_WINDOW
  if (state->filled < PITCH_WINDOW)
  {
    uint32_t new_filled = state->filled + PITCH_BLOCK_SIZE_DECIM;
    state->filled = new_filled > PITCH_WINDOW ? PITCH_WINDOW : new_filled;
  }

  state->since_hop += PITCH_BLOCK_SIZE_DECIM;

  if (state->filled == PITCH_WINDOW && state->since_hop >= PITCH_HOP) 
  {
    difference(state->window, state->d);
    cmndf(state->d, state->dprime);
    yin_estimate_t est = yin(state->d, state->dprime);

    if (!isfinite(est.tau) || est.tau <= 0.0f)
    {
      out->freq_hz = NAN;
    }
    else if (est.in_range)
    {
      out->freq_hz  = PITCH_FS_IN / refine(state->raw_window, state->d_refine, est.tau * PITCH_DECIM);
    } 
    else
    {
      out->freq_hz = PITCH_FS / est.tau;
    }

    state->since_hop = 0;
    out->sample_count = state->samples_in;
    out->confidence = est.confidence;
    out->tau = est.tau;
    out->valid = est.in_range && est.confidence <= PITCH_GATE && isfinite(out->freq_hz);
    return true;
  }

  return false;
}
