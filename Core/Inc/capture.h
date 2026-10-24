#ifndef CAPTURE_H
#define CAPTURE_H

#include "stm32f4xx_hal.h" // IWYU pragma: keep
#include <stdbool.h>
#include <stdint.h>

void capture_init(UART_HandleTypeDef *huart);
bool capture_record(void);
bool capture_send(void);
const int16_t *capture_get_buffer(void);

#endif /* CAPTURE_H */
