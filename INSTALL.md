# Install Instructions

In order to build TUDEL onto your machine, you must first ensure Git is installed. If you are on Windows, open Powershell from your start menu, 
or on MacOS, open the Terminal applicaiton from the startup menu.

type `git` and press enter. If you do not see a series of options, or are not redirected to an install process, you must download the Git terminal 
onto your machine. It can be downloadownloaded [here](https://git-scm.com/downloads). Follow the instructions based on your operating system.

Paste `git --version` into your terminal. If you get a version number, it has been installed correctly. If not, restart your terminal application.

Then  paste `python -V` into your terminal. If you do not see a version number, type `python`.If you are redirected to a Windows Store page, download the prompted application for Python 3.10. 
If you are not redirected, follow the download guide [here](https://www.python.org/downloads/). Once you are finished, attempt `python -V` 
again and check for a version number. If, after entering `python`, your terminal shows `>>>` before a new line. type `exit()` and continue with the download.

After this is complete, paste `git clone https://github.com/quothbonney/TUDEL.git` into your terminal. This should clone the code onto your system. Then enter the new folder with `cd TUDEL`.

Install all dependencies with `pip install -r requirements.txt`. Allow a few moments for the dependencies to be installed.

Finally, launch TUDEL with `python application.py`

--

If the program does not start and you get an ImportError, enter `pip3 install opencv-python`.


