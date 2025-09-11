🚧WIP🚧

# qpm

A simple GUI to perform bacteria segmentation and qpm reconstruction.

## To launch the GUI

NOTE: [uv](https://docs.astral.sh/uv/getting-started/installation/) should be already installed.

Install [git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```bash
uvx git+https://github.com/fdrgsp/qpm
```

## Windows, GPU & CUDA

NOTE: [uv](https://docs.astral.sh/uv/getting-started/installation/) and [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed.

- download the repository (zip file)
- extract the folder (it should contain src, .gitignore, pyproject.toml and README.md)
- in your terminal, cd into this folder (e.g. `cd Path/to/qpm`)
- `uv venv qpm-env`
- `qpm-env\Scripts\activate`
- `uv pip install .`
- `uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- run the gui with: `qpm`
