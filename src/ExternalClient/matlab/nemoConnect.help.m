nemoConnect: initialize connection to nemo
------------------------------------------

::

	nemoConnect()
	nemoConnect(host)
	nemoConnect(host, port)

Connect to simulation backend. With no arguments the connection is to localhost on the default port (56100). In the other two forms the host and port can be explicitly defined.  This must be called before any other operations.

