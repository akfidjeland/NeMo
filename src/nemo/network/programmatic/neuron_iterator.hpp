#ifndef NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP
#define NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP

#include <typeinfo>
#include <vector>
#include <deque>

#include <nemo/network/iterator.hpp>
#include <nemo/NeuronType.hpp>


namespace nemo {
	namespace network {
		namespace programmatic {

//! \todo could probably use a boost iterator here
class NEMO_BASE_DLL_PUBLIC neuron_iterator : public abstract_neuron_iterator
{
	public :

		typedef std::map<nidx_t, size_t>::const_iterator base_iterator;

		neuron_iterator(base_iterator it,
			const std::vector< std::deque<float> >& param,
			const std::vector< std::deque<float> >& state,
			const NeuronType& type) :
			m_it(it), m_nt(type), m_param(param), m_state(state) {}

		void set_value() const {
			m_data.second = Neuron(m_nt);
			size_t n = m_it->second;
			m_data.first = m_it->first;
			for(size_t i=0; i < m_param.size(); ++i) {
				m_data.second.f_setParameter(i, m_param[i][n]);
			}
			for(size_t i=0; i < m_state.size(); ++i) {
				m_data.second.f_setState(i, m_state[i][n]);
			}
		}

		const value_type& operator*() const {
			set_value();
			return m_data;
		}

		const value_type* operator->() const {
			set_value();
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

		const NeuronType m_nt;

		const std::vector< std::deque<float> >& m_param;
		const std::vector< std::deque<float> >& m_state;
};

}	}	}

#endif
