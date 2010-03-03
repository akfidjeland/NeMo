#ifndef TIMER_HPP

#include <sys/time.h>

namespace nemo {

/*! \brief Simple timer measuring both simulated and wall-clock time.
 *
 * The main simulation code keeps track of the current cycle as well, but this
 * timer can be reset.
 */
class Timer 
{
	public:

		Timer() { reset(); }

		// call for every simulation step
		void step() { m_simCycles++ ; }

		/*! \return elapsed wall-clock time in milliseconds */
		unsigned long elapsedWallclock() const;

		/*! \return elapsed simulation time in milliseconds */
		unsigned long elapsedSimulation() const;

		void reset();

	private:

		struct timeval m_start;

		unsigned long m_simCycles;
};



inline
unsigned long
Timer::elapsedWallclock() const
{
	struct timeval m_end;
	gettimeofday(&m_end, NULL);
	struct timeval m_res;
	timersub(&m_end, &m_start, &m_res);
	return 1000 * m_res.tv_sec + m_res.tv_usec / 1000;
}



inline
unsigned long
Timer::elapsedSimulation() const
{
	return m_simCycles;
}



inline
void
Timer::reset()
{
	gettimeofday(&m_start, NULL);
	m_simCycles = 0;
}

}

#endif
