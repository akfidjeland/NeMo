#ifndef RNG_HPP
#define RNG_HPP

class RNG {

	public:

		RNG();

		float gaussian();

		unsigned uniform();

	private:

		unsigned int m_state[4];
};

//! \todo make this non-inlined
//#include "RNG.cpp"

#endif
