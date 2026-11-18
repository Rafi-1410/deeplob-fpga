# Training on a shared Linux GPU server

```bash
git -c http.version=HTTP/1.1 clone https://github.com/Rafi-1410/deeplob-fpga.git
cd deeplob-fpga
conda create -n lob python=3.11 -y && conda activate lob
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA driver
pip install -r requirements.txt
curl -L -o data.zip https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
unzip -o data.zip -d data
```

Long runs go in tmux so they survive a closed SSH session:

```bash
tmux new -s train                       # detach: Ctrl+B then D; back: tmux attach -t train
~/.conda/envs/lob/bin/python train.py --model deeplob --horizon 10
```

Inside tmux, `python` may resolve to the system Python; call the environment's interpreter
directly as above (or `conda activate lob` inside the session). Check progress with
`tail -f logs/<tag>.log` or `ls runs/`.

* FI-2010 sweep, 3 seeds, both horizons: `SEEDS="0 1 2" HORIZONS="10 100" bash run_server.sh`
* LOBSTER experiment: `bash scripts/run_lobster.sh`
* More seeds for one existing run: `python rerun_seed.py <tag> <seed>`
* Colab alternative: `notebooks/deeplob_fpga_colab.ipynb`
