#include <unistd.h>
#include "usart.h"

int _write(int file, char *ptr, int len)
{
  if (file != STDOUT_FILENO && file != STDERR_FILENO)
  {
    return -1;
  }

  if (HAL_UART_Transmit(&huart2, (uint8_t *)ptr, len, 0xFFFF) != HAL_OK)
  {
    return -1;
  }
  return len;
}
