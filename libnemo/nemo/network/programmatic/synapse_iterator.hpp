#ifndef NEMO_NETWORK_PROGRAMMATIC_SYNAPSE_ITERATOR_HPP
#define NEMO_NETWORK_PROGRAMMATIC_SYNAPSE_ITERATOR_HPP

#include <typeinfo>
#include <nemo/network/iterator.hpp>
#include <nemo/NetworkImpl.hpp>

namespace nemo {
	namespace network {

		class NetworkImpl;

		namespace programmatic {

class NEMO_BASE_DLL_PUBLIC synapse_iterator : public abstract_synapse_iterator
{
	public :

		synapse_iterator(
				NetworkImpl::fcm_t::const_iterator ni,
				NetworkImpl::fcm_t::const_iterator ni_end,
				NetworkImpl::axon_t::const_iterator bi,
				NetworkImpl::axon_t::const_iterator bi_end,
				NetworkImpl::bundle_t::const_iterator si,
				NetworkImpl::bundle_t::const_iterator si_end) :
			ni(ni), ni_end(ni_end), 
			bi(bi), bi_end(bi_end),
			si(si), si_end(si_end) { }

		const value_type& operator*() const {
			m_data = Synapse(ni->first, bi->first, *si);
			return m_data;
		}

		const value_type* operator->() const {
			m_data = Synapse(ni->first, bi->first, *si);
			return &m_data;
		}

		abstract_synapse_iterator* clone() const {
			return new synapse_iterator(*this);
		}

		abstract_synapse_iterator& operator++() {
			++si;
			if(si == si_end) {
				++bi;
				if(bi == bi_end) {
					++ni;
					if(ni == ni_end) {
						/* When reaching the end, all three iterators are at
						 * their respective ends. Further increments leads to
						 * undefined behaviour */
						return *this;
					}
					bi = ni->second.begin();
					bi_end = ni->second.end();
				}
				si = bi->second.begin();
				si_end = bi->second.end();
			}
			return *this;
		 }

		bool operator==(const abstract_synapse_iterator& rhs_) const {
			if(typeid(*this) != typeid(rhs_)) {
				return false;
			}
			const synapse_iterator& rhs = static_cast<const synapse_iterator&>(rhs_);
			return ni == rhs.ni && ni_end == rhs.ni_end
				&& bi == rhs.bi && bi_end == rhs.bi_end
				&& si == rhs.si && si_end == rhs.si_end;
		}

		bool operator!=(const abstract_synapse_iterator& rhs) const {
			return !(*this == rhs);
		}

	private :

		NetworkImpl::fcm_t::const_iterator ni;
		NetworkImpl::fcm_t::const_iterator ni_end;
		NetworkImpl::axon_t::const_iterator bi;
		NetworkImpl::axon_t::const_iterator bi_end;
		NetworkImpl::bundle_t::const_iterator si;
		NetworkImpl::bundle_t::const_iterator si_end;

		mutable value_type m_data;
};



}	}	}

#endif
