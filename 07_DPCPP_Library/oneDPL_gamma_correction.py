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

# # oneDPL- Gamma Correction example
#
#
#
#

# #### Sections
# - [Gamma Correction](#Gamma-Correction)
# - [Why use buffer iterators?](#Why-use-buffer-iterators?)
# - _Lab Exercise:_ [Gamma Correction](#Lab-Exercise:-Gamma-Correction)
# - [Image outputs](#Image-outputs)

# ## Learning Objectives
#
# * Build a sample __DPC++ application__ to perform Image processing (gamma correction) using oneDPL.

# ## Gamma Correction
#
# Gamma correction is an image processing algorithm where we enhance the image brightness and contrast levels to have a better view of the image.
#
# Below example creates a bitmap image, and applies the gamma to the image using the DPC++ library offloading to a device. Once we run the program we can view the original image and the gamma corrected image in the corresponding cells below  
#
# In the below program we write a data parallel algorithm using the DPC++ library to leverage the computational power in __heterogenous computers__. The DPC++ platform model includes a host computer and a device. The host offloads computation to the device, which could be a __GPU, FPGA, or a multi-core CPU__.
#
#  We create a buffer, being responsible for moving data around and counting dependencies. DPC++ Library provides `oneapi::dpl::begin()` and `oneapi::dpl::end()` interfaces for getting buffer iterators and we implemented as below.
#  
#  
#  
# ### Why use buffer iterators?
#
# Using buffer iterators will ensure that memory is not copied back and forth in between each algorithm execution on device. The code example below shows how the same example above is implemented using buffer iterators which make sure the memory stays on device until the buffer is destructed.
#  
# Pass the policy object to the `std::for_each` Parallel STL algorithm, which is defined in the oneapi::dpl::execution namespace  and pass the __'begin'__ and __'end'__  buffer iterators as the second and third arguments. 
#
# The `oneapi::dpl::execution::dpcpp_default` object is a predefined object of the device_policy class, created with a default kernel name and a default queue. Use it to create customized policy objects, or to pass directly when invoking an algorithm.
# The Parallel STL API handles the data transfer and compute.
#
# ### Lab Exercise: Gamma Correction
# * In this example the student will learn how to use oneDPL library to perform the gamma correction.
# * Follow the __Steps 1 to 3__ in the below code to create a SYCL buffer, create buffer iterators, and then call the std::for each function with DPC++ support. 
#
# 1. Select the code cell below, __follow the STEPS 1 to 3__ in the code comments, click run ▶ to save the code to file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile gamma-correction/src/main.cpp
//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iomanip>
#include <iostream>
#include <CL/sycl.hpp>

#include "utils.hpp"

using namespace sycl;
using namespace std;

int main() {
  // Image size is width x height
  int width = 1440;
  int height = 960;

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  // Lambda to process image with gamma = 2
  auto gamma_f = [](ImgPixel &pixel) {
    auto v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0f;

    auto gamma_pixel = static_cast<uint8_t>(255 * v * v);
    if (gamma_pixel > 255) gamma_pixel = 255;
    pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
  };

  // fill image with created fractal
  int index = 0;
  image.fill([&index, width, &fractal](ImgPixel &pixel) {
    int x = index % width;
    int y = index / width;

    auto fractal_pixel = fractal(x, y);
    if (fractal_pixel < 0) fractal_pixel = 0;
    if (fractal_pixel > 255) fractal_pixel = 255;
    pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

    ++index;
  });

  string original_image = "fractal_original.png";
  string processed_image = "fractal_gamma.png";
  Img<ImgFormat::BMP> image2 = image;
  image.write(original_image);

  // call standard serial function for correctness check
  image.fill(gamma_f);

  // use default policy for algorithms execution
  auto policy = oneapi::dpl::execution::dpcpp_default;
  // We need to have the scope to have data in image2 after buffer's destruction
  {
    // ****Step 1: Uncomment the below line to create a buffer, being responsible for moving data around and counting dependencies    
    //buffer<ImgPixel> b(image2.data(), image2.width() * image2.height());

    // create iterator to pass buffer to the algorithm
    // **********Step 2: Uncomment the below lines to create buffer iterators. These are passed to the algorithm
    //auto b_begin = oneapi::dpl::begin(b);
    //auto b_end = oneapi::dpl::end(b);

    //*****Step 3: Uncomment the below line to call std::for_each with DPC++ support    
    //std::for_each(policy, b_begin, b_end, gamma_f);
  }

  image2.write(processed_image);
  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    cout << "success\n";
  } else {
    cout << "fail\n";
    return 1;
  }
  cout << "Run on "
       << policy.queue().get_device().template get_info<info::device::name>()
       << "\n";
  cout << "Original image is in " << original_image << "\n";
  cout << "Image after applying gamma correction on the device is in "
       << processed_image << "\n";

  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_gamma_correction.sh; if [ -x "$(command -v qsub)" ]; then ./q run_gamma_correction.sh; else ./run_gamma_correction.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ### Image outputs
# once you run the program sucessfuly it creates gamma corrected image and the original image. You can see the difference by running the two cells below and visually compare it.  

# ##### View the gamma corrected Image
# Select the cell below and click run ▶ to view the generated image using gamma correction:

from IPython.display import display, Image
display(Image(filename='gamma-correction/build/src/fractal_gamma.png'))

# ##### View the original Image
# Select the cell below and click run ▶ to view the generated image using gamma correction:

from IPython.display import display, Image
display(Image(filename='gamma-correction/build/src/fractal_original.png'))

# # Summary
# In this module you will have learned how to apply gamma correction to Images using Data Parallel C++ Library

# <html><body><span style="color:Red"><h1>Reset Notebook</h1></span></body></html>
#
# ##### Should you be experiencing any issues with your notebook or just want to start fresh run the below cell.
#
#

# + jupyter={"source_hidden": true}
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
button = widgets.Button(
    description='Reset Notebook',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='This will update this notebook, overwriting any changes.',
    icon='check' # (FontAwesome names without the `fa-` prefix)
)
out = widgets.Output()
def on_button_clicked(_):
      # "linking function with output"
      with out:
          # what happens when we press the button
          clear_output()
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/07_DPCPP_Library/ ~/oneAPI_Essentials/07_DPCPP_Library
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
