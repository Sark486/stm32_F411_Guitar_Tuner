#ifndef REPLAY_H
#define REPLAY_H

#include <stdint.h>

/*
  Replay recorded samples through the pitch module and print the results with the cycles
  of every push. Development tool: only built with TUNER_CAPTURE == ON.

  Output (one line each, parsed by tools/replay_compare.py):
    REPLAY <pushes> <results>
    P,<push>,<cycles>                                                  every push
    R,<push>,<sample_count>,<freq_hz>,<confidence>,<tau>,<valid>       after the push that produced it
    S,<min plain>,<max plain>,<min result>,<max result>                cycles; plain = push without a result
    END

  count is in samples; whole blocks of PITCH_BLOCK_SIZE are used, up to REPLAY_MAX_PUSHES.
*/
void replay_run(const int16_t *samples, uint32_t count);

#endif /* REPLAY_H */
