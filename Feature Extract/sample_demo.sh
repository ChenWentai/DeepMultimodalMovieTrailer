namelist=./data/trailer/demo/*
echo $namelist | sed -e 's/\s\+/\n/g' >> ../SoundNet-tensorflow/sample_demo.txt

