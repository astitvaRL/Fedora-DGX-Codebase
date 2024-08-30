# Animated Drawings 3D
https://github.com/user-attachments/assets/58808e46-be19-4f65-83e3-3c2de0483adf

## Installation
*This project has been tested with macOS Ventura 13.2.1 and Ubuntu 18.04. If you're installing on another operating sytem, you may encounter issues.*

We *strongly* recommend activating a Python virtual environment prior to installing Animated Drawings. 
Conda's Miniconda is a great choice. Follow [these steps](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to download and install it. Then run the following commands:

````bash
    # create and activate the virtual environment
    conda create --name animated_drawings python=3.8.13
    conda activate animated_drawings

    # clone AnimatedDrawings and use pip to install
    git clone https://github.com/hjessmith/AnimatedDrawingsRL.git
    cd AnimatedDrawings
    pip install -e .

    # visualize an example character using research visualizer
    cd animated_drawings
    python render.py ../examples/characters/d07eebd699cc4ccf9350c4570d9b273e/mvc.yaml
````
