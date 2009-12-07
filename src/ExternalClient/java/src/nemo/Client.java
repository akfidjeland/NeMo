package nemo;

import java.io.*;
import java.util.List;
import java.util.ArrayList;

import org.apache.thrift.TException;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransportException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;

import nemo.NemoBackend;
import nemo.IzhNeuron;
import nemo.Synapse;
import nemo.ConstructionError;
import nemo.Stimulus;


/**
 * Java interface for Nemo
 *
 * Nemo runs as a separate server process with which communication takes place
 * over a socket using a thrift protocol
 */

public class Client 
{

	/** Connect to localhost */
	public Client() 
		throws TTransportException 
	{
		this("localhost", 56100);
	}

	/** Connect to remote host */
	public Client(String hostname, int port)
		throws TTransportException 
	{
		connect(hostname, port);
	}

	/* The nemo process is controlled via a socket connection using a binary
	 * thrift protocol for communication */
	protected NemoBackend.Client connection;

	//! \todo connect to remote host
	/** Connect to an existing process running on localhost */
    private void connect(String hostname, int port)
		throws TTransportException 
	{
		//! \todo remove magic
		TTransport transport = new TSocket(hostname, port);
		TProtocol protocol = new TBinaryProtocol(transport);
		connection = new NemoBackend.Client(protocol);
		transport.open();
    }

	//! \todo use unboxed type?
	public void enableStdp(
			List<Double> prefire, 
			List<Double> postfire,
			double maxWeight,
			double minWeight)
		throws TException, ConstructionError 
	{
		//! \todo note order inversion of max/min here. Propagate correct order to all
		connection.enableStdp(prefire, postfire, maxWeight, minWeight);
	}

	//! \todo provide a lower-level interface as well

	public void addNeuron(IzhNeuron neuron) 
		throws TException, ConstructionError 
	{
		connection.addNeuron(neuron);
	}

	public void addNeuron(
			int index,
			double a,
			double b,
			double c,
			double d,
			double u,
			double v,
			int[] target,
			short[] delay,
			double[] weight,
			boolean[] plastic)
		throws TException, ConstructionError 
	{
		ArrayList<Synapse> axon = new ArrayList<Synapse>();		

		try {
			int m = target.length;
			for(int i=0; i<m; ++i) {
				Synapse synapse =
					new Synapse(target[i], delay[i], weight[i], plastic[i]);
				axon.add(synapse);
			}
		} catch(ArrayIndexOutOfBoundsException e) {
			throw new ConstructionError(
				"synapse arrays (target, delay, weight) have different sizes");
		}

		addNeuron(new IzhNeuron(index, a, b, c, d, u, v, axon));
	}


	public void startSimulation()
		throws TException, ConstructionError 
	{
		connection.startSimulation();
	}


	public List<List<Integer>> run(List<Stimulus> stimulus) 
		throws TException, ConstructionError 
	{
		return connection.run(stimulus);
	}


	public void applyStdp(double reward) 
		throws TException, ConstructionError
	{
		connection.applyStdp(reward);
	}

	//! \todo getConnectivity
	
	public void stopSimulation()
		throws TException 
	{
		connection.stopSimulation();
	}
}
