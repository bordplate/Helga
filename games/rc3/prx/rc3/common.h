#ifndef RC3_COMMON_H
#define RC3_COMMON_H

#include <lib/common.h>

#define MULTI_TRACE_LOG 0
#define MULTI_LOG(msg, ...) printf("rac1multi: " msg, ##__VA_ARGS__)
//#define MULTI_LOG(msg, ...) do {} while (false)
#if MULTI_TRACE_LOG
#define MULTI_TRACE(msg, ...) printf("*rac1multi: " msg, ##__VA_ARGS__)
#else
#define MULTI_TRACE(msg, ...) do {} while (false)
#endif

#endif
