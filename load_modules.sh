# Start fresh
module purge

# Load the base toolchain
module load foss/2022a

# Load Python if needed (usually included with foss/2022a)
module load Python/3.10.4-GCCcore-11.3.0

# Load PyTorch and CUDA support
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

# Load scientific Python libraries (SciPy, NumPy, etc.)
module load SciPy-bundle/2022.05-foss-2022a

# Load PyYAML
module load PyYAML/6.0-GCCcore-11.3.0

module load h5py/3.7.0-foss-2022a

module load matplotlib/3.5.2-foss-2022a
module load tensorboardX/2.5.1-foss-2022a

pip install --user opencv-python
pip install --user pypng
pip install --user timm
pip install --user antialiased-cnns
pip install --user kornia
pip install --user progressbar


