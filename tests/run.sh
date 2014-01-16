#!/bin/sh
RESULT=0
for a in pcfg dopred 2dop longsent
do
	rm -rf $a
	TMP=`mktemp`
	out=`discodop runexp $a.prm 2>&1 | tee $TMP | grep -v 'pcfg \|post \|ex 100.0' | grep 'ex \|Error'`
	# FIXME: test exit code of runexp as well as grep
	if [ $? = 0 ]; then
		echo "Failure in $a:"
		echo $out
		echo
		cat $TMP
		RESULT=1
	fi
done
if [ $RESULT = 0 ]; then
	echo "Success!"
fi
rm $TMP
exit $RESULT
