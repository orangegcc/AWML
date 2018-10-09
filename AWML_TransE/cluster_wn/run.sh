for ((i=0;i<$1;i+=1)) do
	./generate $i
	./apcluster ./similarity/similarity$i.txt ./pre/pre$i.txt ./cluster/cluster$i.txt
done
#./prepare
