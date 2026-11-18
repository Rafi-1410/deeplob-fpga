import os

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

model_file = os.environ.get("MODEL", "light_w4a4.onnx")
fps = int(os.environ.get("FPS", "6000"))
clk_ns = float(os.environ.get("CLK_NS", "14"))
board = os.environ.get("BOARD", "Pynq-Z2")
estimate_only = os.environ.get("ESTIMATE", "0") == "1"

name = os.path.splitext(os.path.basename(model_file))[0]
out_dir = f"output_{name}_{board}_fps{fps}_clk{clk_ns:g}"

if estimate_only:
    outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
else:
    outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]

cfg = build_cfg.DataflowBuildConfig(
    output_dir=out_dir,
    target_fps=fps,
    synth_clk_period_ns=clk_ns,
    mvau_wwidth_max=10000,
    board=board,
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=outputs,
)
print(f"FINN build: {model_file} -> {out_dir}")
build.build_dataflow_cfg(model_file, cfg)
