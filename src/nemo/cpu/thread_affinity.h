#ifndef NEMO_CPU_THREAD_AFFINITY_H
#define NEMO_CPU_THREAD_AFFINITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Set the thread affinity for the current thread to the specified core */
void setThreadAffinity(unsigned core);

#ifdef __cplusplus
}
#endif

#endif
