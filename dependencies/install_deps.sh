pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0
mim install "mmdet>=3.1.0"

echo -e "Installing mmpose..."
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..

echo -e "Installing mmdetection..."
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .