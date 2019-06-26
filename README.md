# Face Features Recognition

A set of CNNs has been trained in order to classify the features of a face.
The reference dataset is CelebA, which can be downloaded from Kaggle.
 

If you're using **Pycharm**, after clone the repo use this doc to create the venv in the repository: [https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html), use `./venv/` for the folder name, because it's added on `.gitignore`.

After that, you can easly load the dependencies using: 

Remember: in order to use Tensorflow, Tensorflow-GPU you must use Python 3.6 with Pip at latest version.

Also with pycharm there are some conflict with pip. This can be a solution (after create venv)
```
python -m pip install --upgrade pip

pip install -r venv-configuration.txt
```

If you install other libs, remember to export the new configuration with the following comand: 

```
pip freeze > venv-configuration.txt
```

## Execution
For executing the project, simply

```

python Main.py

```

You will be required to insert the path of an image to process (or more), and eventually a file where the output is written. 
Otherwise, the output will be pushed to the terminal

## Note
Some warning may appear during execution.

#Models
Probably it will not be possible to load the models on Github.
In cases you do not find the .h5 file related, please contact me.

## Links

- [Configuring Virtualenv Environment Pycharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)
- [Link For Downloading Models](https://www.dropbox.com/sh/yohons41hjz9y5d/AAAFEAI3PObEWcOMG3vGjEHQa?dl=0)


## Author

- Silvio Barra (@silviobarra85)
- Fabio Narducci
- Sara Borriello
- Marco Picariello
