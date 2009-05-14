#ifndef TERM_COLOUR_HPP
#define TERM_COLOUR_HPP

#include <stdio.h>

//! \todo wrap in namespace

enum TerminalMode {
	RESET = 0,
	BRIGHT = 1,
	DIM	= 2,
	UNDERLINE =	3,
	BLINK =	4,
	REVERSE	= 7,
	HIDDEN	= 8
};

enum TerminalColour {
	BLACK =	0,
	RED	= 1,
	GREEN = 2,
	YELLOW =	3,
	BLUE = 4,
	MAGENTA = 5,
	CYAN = 6,
	WHITE =	7
};



void setTextColour(FILE* outfile, int attr, int fg, int bg);

int columnWidth();

#endif
