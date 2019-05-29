# Following are the commands to run example codes

virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install tensorflow
pip install -r requirements.txt

Go to examples folders e.g. 02_xor/
then run the python code:
$ python gen_save_model.py
$ python load_n_predict.py


Other useful pip commands:

pip freeze > requirements.txt
pip install -r requirements.txt
pip install --ignore-installed --upgrade "/home/gideon/Development/tensorflow_ex/python/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl"

