package nemo;

import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.thrift.TException;
import org.apache.thrift.transport.TTransportException;

import nemo.ConstructionError;
import nemo.Stimulus;
import nemo.IzhNeuron;

/**
 * Matlab interface for Nemo
 *
 * This is just an extension of the 'regular' java class. Some methods are
 * wrapped in additional marshalling and unmarshalling code to convert to and
 * from Matlab matrix format (or rather: the type of Java arrays which Matlab
 * can work with).
 *
 */

public class MatlabClient extends Client
{
	public MatlabClient(String hostname, int port)
		throws TTransportException
	{
		super(hostname, port);
	}


	/**
	 * Run simluation for a given number of cycles
	 *
	 * @param ncycles number of cycles for which simulation should run
	 * @param fstim firing stimulus as 2-by-n matrix stimulus in column-major
	 *     order. The first column contains firing cycles, while the second
	 *     column contains neuron indices. The values in this array (both for
	 *     cycles and neurons) should be in the range [0, ncycles). This is not
	 *     checked.
	 * @return 2-by-n matrix of fired neurons in the same format as fstim, i.e.
	 *     one column of cylces and one column of neuron indices.
	 *
	 */
	public int[] run(int ncycles, int[] fstim) 
		throws TException, ConstructionError 
	{
		return decodeFiring(super.run(encodeStimulus(ncycles, fstim)));
	}


	/**
	 * Convert firing from unboxed dense to boxed sparse format
	 *
	 * @see run for input format
	 */
	private List<Stimulus> encodeStimulus(int ncycles, int[] fstim)
	{
		ArrayList<Stimulus> allStimuli = new ArrayList<Stimulus>(ncycles);

		/* The input vector is a 2xn matrix in column-major order. Iterate over
		 * both columns at the same time. */
		int rowCount = 0;
		if(fstim != null) {
			rowCount = fstim.length / 2;
		}

		int row = 0;
		for(int cycle=0; cycle < ncycles; ++cycle) {

			ArrayList<Integer> cycleStimuli = new ArrayList<Integer>(); 
			while(row < rowCount && fstim[row] == cycle) {
				int neuron = fstim[rowCount+row];
				cycleStimuli.add(neuron);
				row++;
			}
			allStimuli.add(new Stimulus(cycleStimuli));
		}

		return allStimuli;
	}


	/**
	 * Convert from boxed dense to unboxed sparse format 
	 *
	 * @see run for output format
	 */
	private int[] decodeFiring(List<List<Integer>> b_firing)
	{
		int len = 0;
		for(List<Integer> cycle : b_firing) {
			len += cycle.size();
		}

		int[] u_firing = new int[len*2];

		int row = 0;
		int cycle = 0;
		Iterator<List<Integer>> iterator = b_firing.iterator();
		while (iterator.hasNext()) {
			List<Integer> cycleFiring = iterator.next();
			for(Integer neuron : cycleFiring) {
				// Fill in in column-major order
				u_firing[row] = cycle;
				u_firing[len+row] = neuron;
				row += 1;
			}
			cycle += 1;
		}

		return u_firing;
	}


	public static final short INVALID_TARGET = -1;

	public void setNetwork(
			double[] a,
			double[] b,
			double[] c,
			double[] d,
			double[] u,
			double[] v,
			int[] target,
			short[] delay,
			double[] weight,
			boolean[] plastic)
		throws ConstructionError, TException
	{	
		int ncount = a.length;

		int scount = target.length / ncount;

		//! \todo check length of the remaining neuron arrays
		//! \todo check length of the remaining synapse arrays
		//! \todo check that scount does not have a remainder
		
		// extract each neuron and send
		for(int n_idx=0; n_idx<ncount; ++n_idx) {

			ArrayList<Synapse> axon = new ArrayList<Synapse>();		

			for(int s_idx=n_idx*scount; s_idx < (n_idx+1)*scount; ++s_idx) {
				if(target[s_idx] != INVALID_TARGET) {
					Synapse synapse = new Synapse(target[s_idx],
							delay[s_idx], weight[s_idx], plastic[s_idx]);
					axon.add(synapse);
				}
			}

			IzhNeuron neuron = new IzhNeuron(n_idx, 
					a[n_idx], b[n_idx], c[n_idx], d[n_idx], u[n_idx], v[n_idx], axon);
			connection.addNeuron(neuron);
		}
	}

	

	/* It seems that for some inexlicable reason Java has no way to
	 * convert double[] to ArrayList<double>. */
	private ArrayList<Double> boxedList(double... in)
	{
		int len = 0;
		if(in != null) {
			len = in.length;
		}
		ArrayList<Double> out = new ArrayList<Double>(len);
		for (int i = 0; i < len; i++) {
			out.add(in[i]);
		}
		return out;
	}


	public void enableStdp(
			double[] pre,
			double[] post,
			double maxWeight,
			double minWeight)
		throws TException, ConstructionError
	{
		super.enableStdp(boxedList(pre), boxedList(post), maxWeight, minWeight); 
	}

}
