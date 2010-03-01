/* Data structures which are used for communication between different
 * partitions, need to be double buffered so as to avoid race conditions.
 * These functions return the double buffer index (0 or 1) for the given cycle,
 * for either the read or write part of the buffer */


__device__
uint
readBuffer(uint cycle)
{
    return (cycle & 0x1) ^ 0x1;
}


__device__
uint
writeBuffer(uint cycle)
{
    return cycle & 0x1;
}

