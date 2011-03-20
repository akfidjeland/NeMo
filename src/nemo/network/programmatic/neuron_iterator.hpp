#ifndef NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP
#define NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP

#include <typeinfo>
#include <nemo/network/iterator.hpp>

namespace nemo {
	namespace network {
		namespace programmatic {

//! \todo could probably use a boost iterator here
class NEMO_BASE_DLL_PUBLIC neuron_iterator : public abstract_neuron_iterator
{
	public :

		typedef std::map<nidx_t, Neuron>::const_iterator base_iterator;

		neuron_iterator(base_iterator it) : m_it(it) { }

		const value_type& operator*() const {
			//! note: returning *m_it here results in warning regarding reference to temporary
			m_data = *m_it;
			return m_data;
		}

		const value_type* operator->() const {
			m_data = *m_it;
			return &m_data;
		}

		nemo::network::abstract_neuron_iterator* clone() const {
			return new neuron_iterator(*this);
		}

		nemo::network::abstract_neuron_iterator& operator++() {
			++m_it;
			return *this;
		}

		bool operator==(const abstract_neuron_iterator& rhs) const {
			return typeid(*this) == typeid(rhs) 
				&& m_it == static_cast<const neuron_iterator&>(rhs).m_it;
		}

		bool operator!=(const abstract_neuron_iterator& rhs) const {
			return !(*this == rhs);
		}

	private :

		base_iterator m_it;

		mutable value_type m_data;
};

}	}	}

#endif
