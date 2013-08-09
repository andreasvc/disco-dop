#!/bin/sh
RESULT=0
for a in pcfg dopred 2dop
do
	rm -rf $a
	out=`discodop runexp $a.prm 2>&1 | grep -v 'pcfg \|post \|ex 100.0' | grep 'ex \|Error'`
	# FIXME: test exit code of runexp
	if [ $? = 0 ]; then
		echo "Failure in $a:"
		echo $out
		RESULT=1
	fi
done
if [ $RESULT = 0 ]; then
	echo "Success!"
fi
exit $RESULT
