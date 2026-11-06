#ifndef CAPTURE_H
#define CAPTURE_H

#include "stm32f4xx_hal.h" // IWYU pragma: keep
#include <stdint.h>

void capture_init(UART_HandleTypeDef *huart);
bool capture_record(void);
bool capture_send(void);
const int16_t *capture_get_buffer(void);
uint32_t capture_get_sample_count(void);
bool capture_receive(void);

#endif /* CAPTURE_H */
