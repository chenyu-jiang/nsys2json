# nsys2json

<p align="center"><img width="836" alt="nsys2json_screenshot" src="https://user-images.githubusercontent.com/22335566/206858627-e8d1e92d-d096-493a-9c94-3da2d8a79734.png"></p>


A Python script to convert the output of NVIDIA Nsight Systems (in SQLite format) to JSON in [Google Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#) for more customizable visualization and analysis. To view the resulting json file, goto ```chrome://tracing``` in Google Chrome or use [Perfetto](https://ui.perfetto.dev/).

Inspired and adapted from [nvprof2json](https://github.com/ezyang/nvprof2json).

The SQLite schema used by Nsight Systems is documented [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#exporter-sqlite-schema).

## Usage
*If you have a '.qdrep' file, you can convert it first to SQLite format through Nsight Systems [UI](https://developer.nvidia.com/nsight-systems/get-started) or [CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-export-command-switch-options).*

To extract kernel activities and NVTX annotated regions (e.g. [torch.cuda.nvtx.range](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html)):
```bash
python3 nsys2json.py <nsys_sqlite_file> -o <output_json>
```

To filter out kernel activities or NVTX annotated regions, use:
```bash
-activity-type {kernel, nvtx-cpu, nvtx-kernel}
```
* `kernel`: Raw CUDA kernel activities
* `nvtx-cpu`: NVTX annotated regions on CPU threads
* `nvtx-kernel`: NVTX annotated regions, but calculate the start and end time from CUDA kernel activities launched within the region

To filter NVTX regions based on name, use:
```bash
--nvtx-event-prefix <prefix>
```

To apply custom coloring scheme to NVTX regions, use:
```bash
--nvtx-color-scheme <dict_mapping_regex_to_chrome_colors>
```
e.g.,
```bash
--nvtx-color-scheme '{"comm": "thread_state_iowait", "Layer .* compute": "thread_state_running"}
```
For the list of available colors, see [here](https://chromium.googlesource.com/external/trace-viewer/+/bf55211014397cf0ebcd9e7090de1c4f84fc3ac0/tracing/tracing/ui/base/color_scheme.html).

## Known Issues
* This script assumes each process in the profile only executes kernel on one GPU. Process id is used to match NVTX regions to the corresponding device. Changes to process and thread naming scheme in the JSON file are needed if this assumption is violated.
