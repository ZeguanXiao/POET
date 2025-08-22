# create conda environment
conda create -n poet python=3.10 && conda activate poet

# install pytorch
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install remaining dependencies
pip install -r requirements.txt
pip install vllm==0.4.1
pip install lm-eval==0.4.7
