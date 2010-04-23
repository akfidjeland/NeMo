#ifndef NEMO_CPU_RNG_HPP
#define NEMO_CPU_RNG_HPP

class RNG {

	public:

		RNG();

		RNG(unsigned seeds[]);

		float gaussian();

		unsigned uniform();

	private:

		unsigned m_state[4];
};

#endif
