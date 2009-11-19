#ifndef TIMER_HPP

#include <sys/time.h>

/*! \brief Simple timer mesauring wall-clock time */
class Timer 
{
	public:

		Timer() { reset(); }

		/*! \return elapsed time in milliseconds */
		long int elapsed();

		void reset();

	private:

		struct timeval m_start;
};



inline
long int
Timer::elapsed()
{
	struct timeval m_end;
    gettimeofday(&m_end, NULL);
    struct timeval m_res;
    timersub(&m_end, &m_start, &m_res);
    return 1000 * m_res.tv_sec + m_res.tv_usec / 1000;
}



inline
void
Timer::reset()
{
    gettimeofday(&m_start, NULL);
}

#endif
