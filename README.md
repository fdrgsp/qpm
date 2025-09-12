🚧WIP🚧

# qpm

A simple GUI to perform bacteria segmentation and qpm reconstruction.

## To launch the GUI

Install [git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```bash
uvx git+https://github.com/fdrgsp/qpm
```

## GPU & CUDA on Windows

To use the GPU version of `qpm` on Windows, you need to have a compatible NVIDIA GPU and the corresponding CUDA toolkit installed.

NOTE: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed.

You need to go through the following steps only once, just the first time you install `qpm`:

- download the repository (zip file)
- extract the folder (it should contain src, .gitignore, pyproject.toml and README.md)
- in your terminal, `cd` into this folder (e.g. `cd Path/to/qpm`)
- `uv venv qpm-env`
- `qpm-env\Scripts\activate`
- `uv pip install .`
- update the torch and torchvision packages to the version compatible with your CUDA version, e.g. for CUDA 12.6:

  `uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126`

  NOTE: to get the correct URL, check the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

- launch the GUI with: `qpm`

Once you have done this once, in the future, to run the `qpm` GUI, you can simply use your terminal and run:

- `cd Path/to/qpm`
- `qpm-env\Scripts\activate`
- `qpm`
