#!/bin/sh
for a in pcfg 2dop dopred longsent
do
	rm -rf $a
	# run parser, if that succeeds,
	# check that we find a perfect exact match score.
	out=`discodop runexp $a.prm 2>&1`
	if [ $? -ne 0 ]; then
		echo "$out"
		echo "Nonzero exit code in $a:"
		exit $?
	fi
	line=`! grep -v 'pcfg \|post \|ex 100.0' $a/output.log | grep 'ex \|Error'`
	if [ $? -ne 0 ]; then
		cat $a/output.log
		echo "Failure in $a:"
		echo "$line"
		exit $?
	fi
done
echo "Success!"
exit 0
