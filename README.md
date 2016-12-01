# Optical Character Recognition

This program recognizes handwritten Latin characters and digits using criteria obtained from training data.

## Environment Setup
Download the following packages
- Python 2.7.X
- OpenCV 2.4.X
- Numpy 1.9+
- Matplotlib 1.3+

Make sure all the versions are 32-bit.

On Windows: If you're having trouble, try installing using the pip command. If you don't have pip, run this [get-pip.py](https://bootstrap.pypa.io/get-pip.py) Python script in your C:/Python27/Scripts folder. You can now call pip from the command line in that directory.

## Running our program
Create classifications from training data
> python train.py

Run our OCR algorithm on our test images 
> python test.py

Run our OCR algorithm on our test images and compare results with expected output
> python testWithAccuracy.py

You must run the training file before running any tests.

## Presentation and Documents
Our Google Slides presentation [here](http://go.osu.edu/ocr-5524).

## Workflow

### Project Management
The project is divided into features, or tasks. We will manage tasks using the following Trello board: [http://go.osu.edu/cse5524](http://go.osu.edu/cse5524). Each task will have its own card on the Trello board. 

### Other Guidelines
- Make sure to have good descriptions for your commits. 
- Comments are appreciated.
- Try and refer to Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html#Comments)

