#!/bin/sh
for a in pcfg dopred 2dop
do
	echo $a
	rm -rf $a
	discodop runexp $a.prm 2>&1 | grep -v 'pcfg \|post ' | grep 'ex '
done

