#!/usr/bin/perl
# Print the help comments from m-file. This script assumes the relevant comment
# block starts on the first line, which is followed by a blank line, followed
# by a synopsis.
#
# The comment block is written to stdout. 

if($#ARGV+1 != 1) {
	die "Usage: extract_doc <m-file>";
} 

open FILE, "<$ARGV[0]" or die $!;

my $lineno = 0;
while (my $line = <FILE>) {

	$lineno += 1;

	if ($line =~ m/^\%/) {

		# All-blank lines are ok
		if ($line =~ m/^%$/) {
			print "\n";
			next;
		}

		# Third line is formatted as code block
		if($lineno == 3) {
			print "Usage::\n\n";
		}

		# For body of text, make parameter names (all caps) inline literals
		#if($lineno > 3) {
		#	$line =~ s/([ -])([[:upper:]]+)([ ,.])/\1``\2``\3/g;	
		#}

		print substr($line, 2);

		# First line is formatted to header
		if($lineno == 1) {
			print '-' x (length($line)-3), "\n";
		}

	} else {
		exit; #end of comment block
	}
}
