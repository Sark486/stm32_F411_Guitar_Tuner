#include "cycles.h"
#include "stm32f4xx.h"

void cycles_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t cycles_get(void)
{
    return DWT->CYCCNT;
}
