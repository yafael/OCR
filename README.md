# Optical Character Recognition

This program recognizes handwritten Latin characters and digits using criteria obtained from training data.

## Run the Program

### Environment Setup
Download the following packages
- Python 2.7.X
- OpenCV 2.4.X
- Numpy 1.9+
- Matplotlib 1.3+

Make sure all the versions are 32-bit.

Download the [get-pip.py](https://bootstrap.pypa.io/get-pip.py) Python script and put it in your C:/Python27/Scripts folder. From the Scripts directory, run
> python get-pip

In the Scripts folder, run the following commands:
> pip install python-dateutil

> pip install pyparsing

### Commands
Classify training data
> python train.py

Recognize characters in images
> python test.py

You can run both train and test files using
> python demo.py


## Workflow

### Project Management
The project is divided into features, or tasks. We will manage tasks using the following Trello board: [http://go.osu.edu/cse5524](http://go.osu.edu/cse5524). Each task will have its own card on the Trello board. 

### Other Guidelines
- Make sure to have good descriptions for your commits. 
- Comments are appreciated.
- Balance code readability with concision.
- Try and refer to Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html#Comments)

