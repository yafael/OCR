# Optical Character Recognition

This program recognizes handwritten Latin characters and digits using criteria obtained from training data.

## Directory Descriptions
LicensePlate-2.X
    The original code from the GitHub dude modded to fit the OpenCV 2.X library.

LicensePlate-2.X-Mod
    LicensePlate-2.X code modded to print text in all regions, not just the license plate.

LicensePlate-3.X
    The original code from the GitHub dude.

NaturalImage
    Ignore this. I need to discard it.

RefactoredProgram
- What I am working on! Kind of works...Run OCR.
- Trainer class will train data but also return a KNN that has been trained on the data
- Tester class is just for testing
- Run OCR.py for the main program


## How to Run

### Setup
Download the following packages
- Python 2.7.X
- OpenCV 2.4.X
- Numpy 1.9.X
- Matplotlib 1.3.X

Make sure all the versions are 32-bit.

Download the [get-pip.py](https://bootstrap.pypa.io/get-pip.py) Python script and put it in your C:/Python27/Scripts folder. From the Scripts directory, run
> python get-pip

In the Scripts folder, run the following commands:
> pip install python-dateutil

> pip install pyparsing

You should be ready to go.


## Workflow

### Project Management
The project is divided into features, or tasks. We will manage tasks using the following Trello board: [http://go.osu.edu/cse5524](http://go.osu.edu/cse5524). Each task will have its own card on the Trello board. 

### Team Member Contribution
1. Create a branch when working on your task. 
2. When complete, make a pull request on your branch.
3. Put the corresponding Trello card under the "Pull Request" list and tag the usernames of other team members.
4. Team members will comment on the Github pull request to suggest revisions.
5. Once approved, merge the branch with the master branch.

### Other Guidelines
- Make sure to have good descriptions for your commits. 
- Comments are appreciated.
- Balance code readability with concision.
- Try and refer to Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html#Comments)

