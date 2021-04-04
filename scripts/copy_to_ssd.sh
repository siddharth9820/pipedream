if [ $USE_SSD -eq 1 ];then
    mkdir -p $SSD_DIR/imagenet/train 
    echo "copying train tar file to SSD..."
    time cp $TRAIN_TAR_FILE $SSD_DIR
    echo "extracting imagenet training data..."
    cd $SSD_DIR/imagenet/train
    time tar -xf $SSD_DIR/ILSVRC2012_img_train.tar
    echo "extracting imagenet training data subdirectories..."
    time find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}";done
    DATASET_DIR=$SSD_DIR/imagenet
      
fi