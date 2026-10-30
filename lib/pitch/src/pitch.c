#include "pitch.h"
#include <string.h>
#include "fir_taps.h"

static_assert(PITCH_FIR_TAPS == FIR_NUM_TAPS, "Number of taps is different between pitch.h and fir_taps.h");
static_assert(PITCH_BLOCK_SIZE % PITCH_DECIM == 0, "PITCH_BLOCK_SIZE doesn't divide evenly by PITCH_DECIM");
static_assert(PITCH_HOP % PITCH_BLOCK_SIZE_DECIM == 0, "PITCH_HOP doesn't divide evenly by samples count after decimation");

static void slide_raw_window(int16_t *raw_window, const int16_t *new_block)
{
  memmove(raw_window, raw_window + PITCH_BLOCK_SIZE, (PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE) * sizeof(*raw_window));
  memcpy(raw_window + (PITCH_RAW_WINDOW - PITCH_BLOCK_SIZE), new_block, PITCH_BLOCK_SIZE * sizeof(*raw_window));
}


void pitch_init(pitch_t *state)
{
  memset(state, 0, sizeof(*state));
}

bool pitch_push(pitch_t *state, const int16_t *block, uint32_t count, pitch_result_t *out)
{
  if(count != PITCH_BLOCK_SIZE)
  {
    return false;
  }
  state->samples_in += PITCH_BLOCK_SIZE;
  slide_raw_window(state->raw_window, block);

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
