#ifndef CYCLES_H
#define CYCLES_H

#include <stdint.h>

/* Enable the cycle counter. */
void cycles_init(void);

/* Return the current cycle count. */
uint32_t cycles_get(void);

#endif /* CYCLES_H */
