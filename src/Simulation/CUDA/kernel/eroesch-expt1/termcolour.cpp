#include "termcolour.hpp"
#include <stdlib.h>


void 
setTextColour(FILE* outfile, int attr, int fg, int bg)
{	
	char command[13];
	sprintf(command, "%c[%d;%d;%dm", 0x1B, attr, fg+30, bg+40);
	fprintf(outfile, "%s", command);
}



/* It's useful to have the terminal width here. The sensible and portable way
 * to do this is to use ncurses. As a temporary hack we use the COLUMNS
 * environment variable, which should be available at least on some linux
 * distributions. (It's also possible to use ioctl(TIOCGWINSZ)) */ 
//! \todo use ncurses to get terminal info
//! \todo deal with changing terminal width
int
columnWidth()
{
	int columns = 80;
	char* columnsString = getenv("COLUMNS");
	if(columnsString != NULL) {
		columns = atoi(columnsString);
	}
	return columns;
}
