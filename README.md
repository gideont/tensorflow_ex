## Tensorflow Examples

Commands to run example codes:
Note: The following steps create python virtual environment for tensorflow and keras, activate the virtual environment, and install the dependencies using pip

```bash
$ virtualenv -p /usr/bin/python3 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Go to examples folders e.g. 02_xor/
then run the python code. 

The example compile, fit the model, then save it.  Next command is to load the model to the neural network and make predictions:
```bash
$ python gen_save_model.py
$ python load_n_predict.py
```

Other useful pip commands:

```bash
$ pip freeze > requirements.txt
$ pip install -r requirements.txt
$ pip install --upgrade "tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl"
```
