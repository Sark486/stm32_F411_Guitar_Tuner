#include "pitch.h"
#include <string.h>
#include "fir_taps.h"

#define FULL_SCALE    32768.0f // 2^15 = 32 768 for 16-bit samples

static_assert(PITCH_FIR_TAPS == FIR_NUM_TAPS, "Number of taps is different between pitch.h and fir_taps.h");
static_assert(PITCH_BLOCK_SIZE % PITCH_DECIM == 0, "PITCH_BLOCK_SIZE doesn't divide evenly by PITCH_DECIM");
static_assert(PITCH_HOP % PITCH_BLOCK_SIZE_DECIM == 0, "PITCH_HOP doesn't divide evenly by samples count after decimation");

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
    state->since_hop = 0;
    out->sample_count = state->samples_in;
    out->valid = false;
    out->freq_hz = 0.0f;
    out->confidence = 0.0f;
    out->tau = 0.0f;
    return true;
  }

  return false;
}
