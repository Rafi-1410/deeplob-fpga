# From PyTorch to the PYNQ-Z2

Tested with FINN (dev branch, late 2026), Vivado / Vitis HLS 2024.2 on Ubuntu, and a PYNQ-Z2
(Zynq-7020) with the PYNQ 3.x image.

## 1. Train and export

```bash
python train.py --model light --horizon 10                       # FP32
python train.py --model light --bits 4 --horizon 10 --init runs/light_fp32_k10_s0.pt --lr 5e-4
python export_qonnx.py --ckpt runs/light_w4a4_k10_s0.pt --bits 4 --out finn/light_w4a4.onnx
python board_testset.py --ckpt runs/light_w4a4_k10_s0.pt --model light --bits 4 \
    --horizon 10 --n 2000 --out board_test_light_w4a4_k10.npz
```

Keep the exact checkpoint that went into the build: the board results are compared with it.

## 2. FINN setup (fixes needed for the PYNQ-Z2)

```bash
git -c http.version=HTTP/1.1 clone https://github.com/Xilinx/finn.git ~/finn
export FINN_XILINX_PATH=~/tools/Xilinx          # Vivado install root
export FINN_XILINX_VERSION=2024.2
export FINN_SKIP_BOARD_FILES=1                  # we install the PYNQ-Z2 board files ourselves
```

1. **"zynq7000 retired" error.** Recent FINN marks Zynq-7000 boards as retired with Vivado 2024.2.
   In `~/finn/src/finn/util/basic.py` set `retired_pynq_boards = {"Pynq-Z1"}` (remove `"Pynq-Z2"`).
2. **board_part not found.** Download the PYNQ-Z2 board files (`pynq-z2.zip`, from the TUL / PYNQ
   board-files page) and unzip them into `~/finn/deps/board_files/` so that
   `~/finn/deps/board_files/pynq-z2/A.0/board.xml` exists.
3. Export the variables above in every new shell; inside tmux run `source ~/.bashrc` first.

## 3. Build

```bash
cd ~/finn
ESTIMATE=1 MODEL=light_w4a4.onnx FPS=6000 CLK_NS=14 ./run-docker.sh build_custom ~/deeplob-fpga/finn
MODEL=light_w4a4.onnx FPS=6000 CLK_NS=14 ./run-docker.sh build_custom ~/deeplob-fpga/finn
```

A full build takes 30–90 minutes. Check timing in
`output_*/stitched_ip/.../timing_summary` (or the Vivado report): the worst negative slack (WNS)
gives the highest safe clock, f = 1000 / (CLK_NS − WNS) MHz. FINN's latency estimate is about
5× pessimistic; its throughput estimate was accurate.

## 4. Run on the board

Laptop: give the Ethernet adapter a static address in the board's subnet (Windows:
`New-NetIPAddress -InterfaceIndex <n> -IPAddress 192.168.2.1 -PrefixLength 24`), then copy the
deployment package and test set:

```bash
scp -r output_light_w4a4_Pynq-Z2_fps6000_clk14/deploy xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/lob6000
scp board_test_light_w4a4_k10.npz board_run.py xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/lob6000/
```

Board (`ssh xilinx@192.168.2.99`):

```bash
sudo -i
source /etc/profile.d/pynq_venv.sh
cd /home/xilinx/jupyter_notebooks/lob6000
cp board_run.py driver/ && cd driver
python board_run.py --bitfile ../bitfile/finn-accel.bit --fclk 66.67 \
    --npz ../board_test_light_w4a4_k10.npz --out ../board_result_66MHz.json
```

`--fclk` matters: the generated driver does not pass the build clock and defaults to 100 MHz.
Typical PL clocks are 100, 66.67, 62.5 and 50 MHz. `board_run.py` checks the outputs against
PyTorch in chunks of 250 windows (larger buffers fail to allocate), then measures batch-1 latency
200 times and throughput at batch 500.
