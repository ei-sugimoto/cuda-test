/*
  Useful Macros for Debugging CUDA Programs
  Copyright (C) 2017 Fujimoto Lab. @ Osaka Prefecture University, Japan
  All Rights Reserved.
  Contact: fujimoto@cs.osakafu-u.ac.jp
*/

#ifndef ERRCHK_H
#define ERRCHK_H

#ifdef _DEBUG

#define CALL(lib_call)                                \
{                                                     \
    const cudaError_t error = lib_call;               \
    if (error != cudaSuccess)                         \
    {                                                 \
       printf("Error: %s:%d,  ", __FILE__, __LINE__); \
       printf("code:%d, reason: %s\n", error,         \
            cudaGetErrorString(error));               \
       exit(1);                                       \
    }                                                 \
}

/* 
  Using C99 variadic macro, 
  the following macro LAUNCH leads the C proprocessor
  which misunderstands a single kernel call construct 
  as multiple arguments of the macro LAUNCH. 
*/

#define LAUNCH(...)                                   \
{                                                     \
	__VA_ARGS__;                                      \
    CALL(cudaPeekAtLastError());                      \
	CALL(cudaDeviceSynchronize());                    \
}

#else

#define CALL(lib_call) lib_call;
#define LAUNCH(...) __VA_ARGS__;

#endif

#endif
