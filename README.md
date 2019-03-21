# An intriguing failing of convolutional neural networks and the CoordConv solution

Implementing: Liu, R., Lehman, J., Molino, P., Such, F.P., Frank, E., Sergeev, A. and Yosinski, J., 2018. [An intriguing failing of convolutional neural networks and the coordconv solution](http://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution). In Advances in Neural Information Processing Systems (pp. 9628-9639).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project was built on Python 3.5.2.

The following packages should be installed:
* ```numpy==1.14.5```
* ```tensorflow-gpu==1.12.0```


The following packages which are part of the Python standard library are used in this project:
* ```os```
* ```shutil```
* ```sys```

### Installing
You may install these packages yourself or use the [requirements.txt](requirements.txt) file like so: 
```
pip install -r requirements.txt
```

### Dataset
Before running any experiments, you'll need to create the dataset. 
Navigate to the ```src``` folder. 
Once there, run the following: 
```
python data/make_dataset.py
```

Upon completion, you should see a folder titled ```data``` in the top level of this project. 

### Path Configuration
Navigate to the file ```src/models/config.py```.
At the top of this file is a variable called ```DIR_PATH```.
Change this to match the directory path to your ```src``` folder. 

## Usage
Currently there are several scripts that will allow you to run experiments on the Not-So-Clevr dataset. 
In the future, this may extend to other datasets and there may be an inclusion of notebooks for more interactive learning. 

### Scripts
These are the 4 experiments you can run and the commands required to do so: 
* Classification using CoordConv: ```python models/train_model_coordconv_classification.py``` 
* Rendering using CoordConv: ```python models/train_model_coordconv_rendering.py```
* Classification without CoordConv: ```python models/train_model_deconv_classification.py```
* Rendering without CoordConv: ```python models/train_model_deconv_rendering.py```

All of these scripts must be run from within the ```src``` folder. 
For the classification experiments, accuracy on the test set is printed once all training epochs have completed. 
For the rendering experiments, images can be viewed using Tensorboard and the process to do this is described further below. 

### Configuration
Aside from the ```DIR_PATH``` there are other configuration options open to you. 
These can be found in the same ```src/models/config.py``` file. 
I encourage you to play around with the various parameters to examine their effects on the experiments. 

### Tensorboard
Throughout the experiments, results are recorded and stored using Tensorflow and the easiest way to view these is using Tensorboard. 

By default, these results will be stored in the following directory (from the top level):
```results/tensorboard_logs```
This can be changed by editing the ```TENSORBOARD_DIR``` parameter in the aforementioned config file. 

To view these results, navigate to a separate command window and run the following: 
```tensorboard --logdir results/tensorboard_logs```

If you change the location of your tensorboard logs, please change the above command appropriately. 

Once done, open a web browser and navigate to the following address: 
```http://localhost:6006/#scalars```

You can also navigate to this address to view the images produced (available for both classification and rendering tasks):
```http://localhost:6006/#images```

<!---## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
--->
## Versioning

I use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ashwindcruz/coord-conv/tags). 

## Authors

* **Ashwin D'Cruz** - [ashwindcruz](https://github.com/ashwindcruz)

<!---See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.--->

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
<!---
## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
--->