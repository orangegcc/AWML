for ((i=0;i<$1;i+=1)) do
    if [ $i -ne 1 -a $i -ne 3 -a $i -ne 4 ]; then
        ./generate $i
	    ./apcluster ./similarity/similarity$i.txt ./pre/pre$i.txt ./cluster/cluster$i.txt
    fi
done
#./prepare
