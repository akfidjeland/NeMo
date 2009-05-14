/* 
 * Simple utility which pings a network server and prints its status to stdout.
 *
 * Author: Andreas Fidjeland 
 */

#include <stdio.h>
#include <stdlib.h>

#include <client.h>


int
main(int argc, char** argv)
{
	if(argc != 2) {
		fprintf(stderr, "Usage: hsping <hostname>\n");
		exit(EXIT_FAILURE);
	}

	const char* hostname = argv[1];
	enum server_status_t status = ping(hostname);

	switch(status) {
		case HOST_READY: printf("%s is ready\n", hostname); break;
		case HOST_BUSY: printf("%s is busy\n", hostname); break;
		case HOST_DOWN: printf("%s is down\n", hostname); break;
		default: 
			fprintf(stderr, "Error: invalid server status\n");
			exit(EXIT_FAILURE);
	}

	exit(EXIT_SUCCESS);
}
