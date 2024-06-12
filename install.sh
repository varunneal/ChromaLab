conda create -n chromalab python=3.11

eval "$(conda shell.bash hook)"
conda activate chromalab

pip install -r requirements.txt

pip install -e "$(pwd)"

