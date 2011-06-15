#!/usr/bin/perl -w

$infile = $ARGV[0];

$refbitrate = 6000; #in Kbits/sec

@refinputsize = (720, 480);

@outputsizes = ([640, 480], [320, 240], [160, 120]);

#####
for $out (@outputsizes) {
	$sizestring = "${$out}[0]x${$out}[1]";
	$outbitrate = int( $refbitrate * (${$out}[0] * ${$out}[1]) / ($refinputsize[0] * $refinputsize[1]) );
	$cmd="ffmpeg -i ${infile}.vob -r 25 -ss 00:00:10 -s $sizestring -an -b ${outbitrate}K ${infile}${sizestring}.m2v";

	print "Executing: $cmd\n\n";
	`$cmd`;
}
