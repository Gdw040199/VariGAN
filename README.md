
# The Official Code of VariGAN: Enhancing Image Style Transfer by UNet Generator & Auto-encoding Discriminator via GAN

### Environmental Requirement
- Windows
- Python 3.X
- CPU or NVIDIA GPU + CUDA CuDNN
- You can refer to `requirements.txt` for specific versions used in this project

---

# Configuration Procedure

## 1. Clone Project
1. Open PyCharm, Tools -> Space -> Clone Repository
2. Repository URL: Enter the GitHub project URL

## 2. Install Necessary Packages
Create an environment using Anaconda and pip to install the necessary libraries:
1. Open the Anaconda Prompt
2. Create a virtual environment: `conda create -n pytorch_VariGAN python=3.8`
3. Enter the virtual environment: `conda activate pytorch_VariGAN`
4. Navigate to the project directory and install the required dependencies: `pip install -r requirements.txt`

## 3. Download Datasets
1. The dataset for this project is identical to *CycleGAN*. Download the dataset from the link below:
2. Extract the official dataset into the project's `datasets` directory: [CycleGAN Datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

## 4. Train the Backbone
1. Set the debugging configuration before running `train.py`.
2. Enter parameters in the file configuration interface in the format: `--dataroot [path to your training set tastA] --name [custom weight save file name] --model Varigan`
3. Modify training parameters in `options/train_options.py` as needed, such as training times, batch size, etc.
4. Start the visdom service (visual interface): `python -m visdom.server`
5. After starting the service, run the `train.py` file to start training the model.

## 5. Custom Training Using Your Own Training Set
1. Naming:
   - The training sets are named `trainA` and `trainB`
   - The test sets are named `tastA` and `tastB`
2. Set the training configuration file and set the training set path to the path of `train`.
3. Train directly, the same way as the previous step.

## Using CycleGAN Weight File After Training
1. Find the path to the `testA` or `testB` image that you want to convert.
2. Rename the weight file you want to use to `latest_net_G.pth`.
3. Before running `test.py`, set the debugging configuration in the format: `--dataroot [path to test set A] --name [weight file name used] --model test --no_dropout`
4. Run the `test.py` file for testing, and the generated converted image will be saved in the `results` directory.
