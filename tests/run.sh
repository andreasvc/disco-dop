#!/bin/sh
RESULT=0
for a in pcfg dopred 2dop
do
	rm -rf $a
	discodop runexp $a.prm 2>&1 | grep -v 'pcfg \|post \|100.0' | grep 'ex '
	if [ $? = 0 ]; then
		echo "Failure"
		RESULT=1
	fi
done
if [ $RESULT = 0 ]; then
	echo "Success!"
fi
exit $RESULT
