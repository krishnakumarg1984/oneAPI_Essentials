# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: "Python 3.7 (Intel\xAE oneAPI)"
#     language: python
#     name: c009-intel_distribution_of_python_3_oneapi-beta05-python
# ---

# # Introduction to JupyterLab and Notebooks

# If you are familiar with Jupyter skip below and head to the first exercise.
#
# <video controls src="src/Introduction_to_Jupyter.mp4" width="720"/>

# __JupyterLab__ is a sequence of boxes referred to as "cells". Each cell will contain text, like this one, or C++ or Python code that may be executed as part of this tutorial. As you proceed, please note the following:
#  
# * The active cell is indicated by the blue bar on the left. Click on a cell to select it.
#  
# * Use the __"run"__ ▶ button at the top or __Shift+Enter__ to execute a selected cell, starting with this one.
#   * Note: If you mistakenly press just Enter, you will enter the editing mode for the cell. To exit editing mode and continue, press Shift+Enter.
#
#
# * Unless stated otherwise, the cells containing code within this tutorial MUST be executed in sequence.
#  
# * You may save the tutorial at any time, which will save the output, but not the state. Saved Jupyter Notebooks will save sequence numbers which may make a cell appear to have been executed when it has not been executed for the new session. Because state is not saved, re-opening or __restarting a Jupyter Notebook__ will required re-executing all the executable steps, starting in order from the beginning.
#  
# * If for any reason you need to restart the tutorial from the beginning, you may reset the state of the Jupyter Notebook and clear all output. Use the menu at the top to select __Kernel -> "Restart Kernel and Clear All Outputs"__
#  
# * Cells containing Markdown can be executed and will render. However, there is no indication of execution, and it is not necessary to explicitly execute Markdown cells.
#  
# * Cells containing executable code will have "a [ ]:" to the left of the cell:
#   * __[ ]__ blank indicates that the cell has not yet been executed.
#   * __[\*]__ indicates that the cell is currently executing.
#   * Once a cell is done executing, a number will appear in the small brackets with each cell execution to indicate where in the sequence the cell has been executed. Any output (e.g. print()'s) from the code will appear below the cell.

# ### Code editing, Compiling and Running in Jupyter Notebooks
# This code shows a simple C++ Hello world. Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.
#

# +
# %%writefile src/hello.cpp
#include <iostream>
#define RESET   "\033[0m"
#define RED     "\033[31m"    /* Red */
#define BLUE    "\033[34m"    /* Blue */

int main(){
    std::cout << RED << "Hello World" << RESET << std::endl;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 ./run_hello.sh;if [ -x "$(command -v qsub)" ]; then ./q run_hello.sh; else ./run_hello.sh; fi

# <html><body><span style="color:green"><h1>Get started on Module 1</h1></span></body></html>
#
# [Click Here](../01_oneAPI_Intro/oneAPI_Intro.ipynb)

# ### Refresh Your Jupyter Notebooks
# If it's been awhile since you started exploring the notebooks, you will likely need to update them for compatibility with Intel&reg; DevCloud for oneAPI and the Intel&reg; oneAPI DPC++ Compiler to keep up with the latest updates.
#
# Run the cell below to get latest and replace with latest version of oneAPI Essentials Modules:

# !/data/oneapi_workshop/get_jupyter_notebooks.sh


