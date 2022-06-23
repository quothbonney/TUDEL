# TUDEL (TU Digital Electrochemistry Lab)

TUDEL is developed by Jack D.V. Carson for the Chemistry Summer Undergraduate Research Program (CSURP) at Tulsa University under an MIT License. 
It is a toolkit for photoanalysis of electrochemical films primarily intended for the synthesis of Perovskite-based solar cells. The primary feature of the
repository is quantitative measurements regarding color, uniformity, and quality of created films.

![TUDEL Examples](/imgs/sample/tudel%20banner.png)

The project will hopefully soon become an plugin to the [ImageJ](https://imagej.nih.gov/ij/) application, however due to the dependencies of the project 
and the interpreted syntax of Python, it is currently only available as a command line interface (CLI).

## Getting Started
### Dependencies
- Python >3.8
- NumPy
- OpenCV-Python
- Matplotlib

All dependency versions are listed in the `requirements.txt` file and can be easily installed with
`pip install requirements.txt` in the command line if you have Python adequately installed.

### Download
- [Git clone](https://github.com/git-guides/git-clone) the repository onto your machine
  - If you do not have git installed, the repository can be downloaded as a .zip file under the **code** button at the top of the page.
  - Unzip the folder
- `cd` into the `Electrodeposition_analysis`
  - If you installed via git, just execute `cd Electrodeposition_analysis`
  - If you installed via .zip, copy the unzipped folder's location from the source, open command line and type `cd <PASTE LOCATION HERE>`
- From here, you can execute any of the following commands

## Usage

### Calibration

Place a piece of bright red tape on a well-lit white surface and photograph. Calibrate the image to a color standard
with the following command. Do not forget the quotation marks.
```shell
python .\calibrate.py "<IMAGE DIRECTORY>"
```

### Calculate Percent Imperfection

After successful calibration, calculate the percent Imperfection of the film with
```shell
python .\main.py <TYPE> "<IMAGE DIRECTORY>"
```
A list of valid types can be viewed with `python .\main.py --help`


### Get Dimensions

The height and width of a film can be calculated with the following
```shell
python .\features\dimensions.py <TYPE> "<IMAGE_DIRECTORY>"
```


## Version History
- V0.3 Documentation and CLI interface
- V0.2 Fixed Sobel edge bug and major refactoring
- V0.1 Initial Commit