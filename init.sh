if [ -n "$1" ]; then
    echo "init in directry" $1
    rm -rf $1
    mkdir $1
    cp ./train.sh $1
    cp ./predict.sh $1
    cp ./accuracy.sh $1
    cp ./run.sh $1
    cp ./clear.sh $1
else
    echo "init in directry checkpoint"
    rm -rf checkpoint
    mkdir checkpoint
    cp ./train.sh checkpoint
    cp ./predict.sh checkpoint
    cp ./accuracy.sh checkpoint
    cp ./run.sh checkpoint
    cp ./clear.sh checkpoint
fi