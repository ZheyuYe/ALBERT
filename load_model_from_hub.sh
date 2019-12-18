mkdir albert_model
cd albert_model
python3 ../save_hub_checkpoint.py

for i in 'base' 'large' 'xlarge' 'xxlagre'
do
    export ALBERT_DIR=$i
    for j in '1' '2'
    do
        export VERSION=$j
        mkdir temp
        cd temp
        wget https://storage.googleapis.com/tfhub-modules/google/albert_${ALBERT_DIR}/${VERSION}.tar.gz
        tar -xvzf ${VERSION}.tar.gz
        mv assets/*.* ../${ALBERT_DIR}_v${VERSION}
        cd ..
        rm -rf temp/
    done
done
