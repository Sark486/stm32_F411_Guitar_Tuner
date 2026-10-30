#ifndef PITCH_H
#define PITCH_H

#include <stdint.h>

#define PITCH_BLOCK_SIZE       256U                             // samples at 32 kHz per push: one DMA half, 8 ms
#define PITCH_HOP              256U                             // analysis runs every 256 decimated samples (8 kHz) which is 32 ms or 4 pushes
#define PITCH_DECIM            4U                               // 32kHz to 8kHz
#define PITCH_BLOCK_SIZE_DECIM (PITCH_BLOCK_SIZE / PITCH_DECIM) // samples at 8kHz per push, 8ms
#define PITCH_WINDOW           1024U                            // samples at 8 kHz: 128 ms
#define PITCH_RAW_WINDOW       (PITCH_WINDOW * PITCH_DECIM)     // the same 128 ms at 32 kHz
#define PITCH_FIR_TAPS         119U                             // filter coefficients; must match fir_taps.h
#define PITCH_TAU_MAX          276U                             // ceil(8000/29); search floor 29 Hz, about A#0, a semitone under B0, so a flat B0 still reads
#define PITCH_D_LEN            (PITCH_TAU_MAX + 2U)             // candidate periods 0 to TAU_MAX + 1; the extra one is the right-hand neighbour of the best candidate


typedef struct {
  float freq_hz; // always filled, use only if 'valid' is true
  float confidence; // lower is better
  float tau; // period in 8 kHz samples, before fine-tuning on 32kHz
  uint32_t sample_count; // total number of 32 kHz samples pushed since pitch_init, including this block
  bool valid; // true when the best period isn't at the edge of the search range and confidence <= 0.5

} pitch_result_t;


typedef struct{
  // fields are private
  float fir_delay[PITCH_FIR_TAPS - 1 + PITCH_BLOCK_SIZE]; // filter input, scaled to +-1.0. It holds the last PITCH_FIR_TAPS - 1 samples of the previous block, followed by the current block
  float window[PITCH_WINDOW]; // the latest PITCH_WINDOW decimated samples (8 kHz, 128 ms), oldest first; YIN runs on it every hop
  int16_t raw_window[PITCH_RAW_WINDOW]; // latest PITCH_RAW_WINDOW input samples (32 kHz, 128 ms), oldest first, kept as int16 exactly as pushed. The fine-tuning step uses them
  float d[PITCH_D_LEN]; // the difference function for tau from 0 to PITCH_TAU_MAX + 1
  float dprime[PITCH_D_LEN]; // normalised d
  uint32_t samples_in; // 32 kHz samples pushed since init; it becomes sample_count in the result
  uint32_t filled; // decimated samples received, counted until the window is full
  uint32_t since_hop; // decimated samples since the last analysis; analysis runs when it reaches PITCH_HOP
} pitch_t;

// inits with new state; must be executed before push
void pitch_init(pitch_t *state);

// returns true when a new result was written; *out is not changed when returned false
// returns false and doesn't change the state if count != PITCH_BLOCK_SIZE
// state, block, out must not be null
bool pitch_push(pitch_t *state, const int16_t *block, uint32_t count, pitch_result_t *out);

#endif /* PITCH_H */

