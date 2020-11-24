这个仓库记录了我学习高光谱图像像素级分类的过程。我写的大部分代码都是基于别人的代码修改而成的。

每个“excise”文件夹都包括一个高光谱图像分类方法。预计会包含CNN、RNN、GAN以及普通的机器学习方法SVM, RF, DT, RF。

我没有上传程序运行过程的产生的数据文件夹,如保存的训练集测试集和权重数据,这些需要自己创建,或者运行程序的时候自动创建.

This repository recorded my process of learning the pixel-level classification of hyperspectral images. Most of the code I wrote was based on someone else's code.

Each "excise" folder include a hyperspectral image classification method. In the future, this repository will include CNN, RNN, GAN and common machine learning methods SVM, RF, DT, RF.

I didn't upload program operation process of the data folder, such as save the training set of test data sets, and weight, the need to create your own, or automatically created when running the program.

Environment:
# Name                    Version                   Build  Channel
absl-py                   0.11.0                   pypi_0    pypi
argon2-cffi               20.1.0                   pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
async-generator           1.10                     pypi_0    pypi
attrs                     20.3.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
bleach                    3.2.1                    pypi_0    pypi
ca-certificates           2020.10.14                    0    defaults
cachetools                4.1.1                    pypi_0    pypi
certifi                   2020.6.20          pyhd3eb1b0_3    defaults
cffi                      1.14.3                   pypi_0    pypi
chardet                   3.0.4                    pypi_0    pypi
colorama                  0.4.4                    pypi_0    pypi
cudatoolkit               10.1.243             h74a9793_0    defaults
cudnn                     7.6.5                cuda10.1_0    defaults
cycler                    0.10.0                   pypi_0    pypi
decorator                 4.4.2                    pypi_0    pypi
defusedxml                0.6.0                    pypi_0    pypi
entrypoints               0.3                      pypi_0    pypi
gast                      0.3.3                    pypi_0    pypi
google-auth               1.23.0                   pypi_0    pypi
google-auth-oauthlib      0.4.2                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.33.2                   pypi_0    pypi
h5py                      2.10.0                   pypi_0    pypi
idna                      2.10                     pypi_0    pypi
ipykernel                 5.3.4                    pypi_0    pypi
ipython                   7.19.0                   pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
jedi                      0.17.2                   pypi_0    pypi
jinja2                    2.11.2                   pypi_0    pypi
joblib                    0.17.0                   pypi_0    pypi
jsonschema                3.2.0                    pypi_0    pypi
jupyter-client            6.1.7                    pypi_0    pypi
jupyter-core              4.6.3                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
kiwisolver                1.3.1                    pypi_0    pypi
markdown                  3.3.3                    pypi_0    pypi
markupsafe                1.1.1                    pypi_0    pypi
matplotlib                3.3.3                    pypi_0    pypi
mistune                   0.8.4                    pypi_0    pypi
nbclient                  0.5.1                    pypi_0    pypi
nbconvert                 6.0.7                    pypi_0    pypi
nbformat                  5.0.8                    pypi_0    pypi
nest-asyncio              1.4.3                    pypi_0    pypi
notebook                  6.1.5                    pypi_0    pypi
numpy                     1.18.5                   pypi_0    pypi
oauthlib                  3.1.0                    pypi_0    pypi
openssl                   1.1.1h               he774522_0    defaults
opt-einsum                3.3.0                    pypi_0    pypi
packaging                 20.4                     pypi_0    pypi
pandas                    1.1.4                    pypi_0    pypi
pandocfilters             1.4.3                    pypi_0    pypi
parso                     0.7.1                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    8.0.1                    pypi_0    pypi
pip                       20.2.4                     py_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
prometheus-client         0.8.0                    pypi_0    pypi
prompt-toolkit            3.0.8                    pypi_0    pypi
protobuf                  3.13.0                   pypi_0    pypi
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.20                     pypi_0    pypi
pydot                     1.4.1                    pypi_0    pypi
pydot-ng                  2.0.0                    pypi_0    pypi
pydotplus                 2.0.2                    pypi_0    pypi
pygments                  2.7.2                    pypi_0    pypi
pyparsing                 2.4.7                    pypi_0    pypi
pyrsistent                0.17.3                   pypi_0    pypi
python                    3.8.6           h60c2a47_0_cpython    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
python-dateutil           2.8.1                    pypi_0    pypi
python_abi                3.8                      1_cp38    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
pytz                      2020.4                   pypi_0    pypi
pywin32                   228                      pypi_0    pypi
pywinpty                  0.5.7                    pypi_0    pypi
pyzmq                     20.0.0                   pypi_0    pypi
requests                  2.25.0                   pypi_0    pypi
requests-oauthlib         1.3.0                    pypi_0    pypi
rsa                       4.6                      pypi_0    pypi
scikit-learn              0.23.2                   pypi_0    pypi
scipy                     1.4.1                    pypi_0    pypi
send2trash                1.5.0                    pypi_0    pypi
setuptools                49.6.0           py38h9bdc248_2    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
six                       1.15.0                   pypi_0    pypi
spectral                  0.22.1                   pypi_0    pypi
sqlite                    3.33.0               he774522_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
tensorboard               2.4.0                    pypi_0    pypi
tensorboard-plugin-wit    1.7.0                    pypi_0    pypi
tensorflow-gpu            2.3.0                    pypi_0    pypi
tensorflow-gpu-estimator  2.3.0                    pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
terminado                 0.9.1                    pypi_0    pypi
testpath                  0.4.4                    pypi_0    pypi
threadpoolctl             2.1.0                    pypi_0    pypi
tornado                   6.1                      pypi_0    pypi
traitlets                 5.0.5                    pypi_0    pypi
urllib3                   1.26.2                   pypi_0    pypi
vc                        14.1                 h869be7e_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
vs2015_runtime            14.16.27012          h30e32a0_2    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
wcwidth                   0.2.5                    pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.35.1             pyh9f0ad1d_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
wincertstore              0.2             py38h32f6830_1005    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
wrapt                     1.12.1                   pypi_0    pypi