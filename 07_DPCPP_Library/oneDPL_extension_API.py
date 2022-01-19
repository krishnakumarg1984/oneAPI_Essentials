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

# # oneDPL - Extension APIs

# ## Learning Objectives
#
# * Explain oneDPL Extension API Algorithms with examples
# * Explain oneDPL Extension API Iterators with examples
# * Explain oneDPL Extension API Utility classes with examples
#
#
# oneDPL consists of an additional set of library classes and functions called the __Extension API__. The Extension API currently includes six algorithms and three functional utility classes and three Iterators.
#
# In this notebook we introduce you to the above __Extension APIs__ and how we can use these in DPC++ program to perform some of the parallel STL operations in heterogenous environments.

# ## List of oneDPL Extension APIs
#
# | Type | function call | Description |
# |:---|:---|:---|
# | Algorithm | [__reduce_by_segment__](#reduce_by_segment) | Performs partial reductions on a sequence's values and keys |
# | Algorithm | [__inclusive_scan_by_segment__](#inclusive_scan_by_segment)  | Performs partial prefix scans on a sequence's values |
# | Algorithm | [__exclusive_scan_by_segment__](#exclusive_scan_by_segment) | Performs partial prefix scans on a sequence's values. sets the first element to the initial value provided |
# | Algorithm | [__binary_search__](#Binary-Search) | Performs a binary search of the input sequence for each of the values in the search sequence provided |
# | Algorithm | [__upper_bound__](#upper_bound) | Performs a binary search of the input sequence for each of the values in the search sequence provided to identify the highest index in the input sequence |
# | Algorithm | [__lower_bound__](#lower_bound) | Performs a binary search of the input sequence for each of the values in the search sequence provided to identify the lowest index in the input sequence |
# | Iterators | [__Zip Iterators__](#Zip-Iterators) | Used for iterating over several containers simultaneously in STL algorithms  |
# | Iterators |  [__Counting Iterators__](#Counting-Iterators) | Iterator for STL algorithms that changes the counter according to arithmetic operations of the iterator type |
# | Iterators | [__Transform Iterators__](#Transform-Iterators) | Iterator that applies a transformation to a sequence |
# | Iterators |  [__discard Iterators__](#discard_iterator) | Random access iterator-like type that provides write-only dereference operations that discard values passed |
# | Iterators | [__Permutation Iterators__](#permutation_iterator) | An iterator whose dereferenced value set is defined by the source iterator provided|
# | Utility Classes | [__identity__](#Functional-Utility-classes) | A C++11 implementation of the C++20 std::identity function object type |
# | Utility Classes | [__minimum__](#Minimum) | A function object type where the operator() applies std::less to its arguments, |
# | Utility Classes | [__maximum__](#Maximum) | A function object type where the operator() applies std::greater to its arguments |
#
#
#
# - _Code:_ [zip iterators and stable sort example](#zip-iterators-and-stable-sort-example)
#

# ## oneDPL Extension API Algorithms
# Below section talks about the list of the six Extension API algorithms

# ### reduce_by_segment
# The reduce_by_segment algorithm performs partial reductions on a sequence's values and keys. Each reduction is computed with a given reduction operation for a contiguous subsequence of values, which are determined by keys being equal according to a predicate. A return value is a pair of iterators holding the end of the output sequences for keys and values.
#
# For correct computation, the reduction operation should be associative. If no operation is specified, the default operation for the reduction is std::plus, and the default predicate is std::equal_to. The algorithm requires that the type of the elements used for values be default constructible.
#
# __Example__:
#
# keys   [0,0,0,1,1,1]
#
# values [1,2,3,4,5,6]
#
# output_keys   [0,1]
#
# output_values [1+2+3=6,4+5+6=15]

# The code below shows the usage of reduce_by_segment. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/reduce_segment.cpp

//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
        
    const int num_elements = 6;    
    auto R = range(num_elements);    
    
    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for Keys
    std::vector<int> input_keys{ 0,0,0,1,1,1 };
    //Initialize the input vector for Values
    std::vector<int> input_values{ 1,2,3,4,5,6 };
    //Output vectors where we get the results back
    std::vector<int> output_keys(num_elements, 0);
    std::vector<int> output_values(num_elements, 0);    
    
    //Create buffers for the above vectors    
    buffer buf_in(input_keys);
    buffer buf_seq(input_values);    
    buffer buf_out_keys(output_keys.data(),R);
    buffer buf_out_vals(output_values.data(),R);


    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto result_key_begin = oneapi::dpl::begin(buf_out_keys);
    auto result_vals_begin = oneapi::dpl::begin(buf_out_vals);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);
    //auto pair_iters = make_pair <std::vector::iterator, std::vector::iterator>

    //Calling the dpstd reduce by search algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here.
    // dpstd::reduce_by_segment returns a pair of iterators to the result_key_begin and result_vals_begin respectively
    int count_keys,count_vals = 0;    
    
    auto pair_iters = oneapi::dpl::reduce_by_segment(make_device_policy(q), keys_begin, keys_end, vals_begin, result_key_begin, result_vals_begin);
    auto iter_keys = std::get<0>(pair_iters);    
    // get the count of the items in the result_keys using std::distance
    count_keys = std::distance(result_key_begin,iter_keys);    
    //get the second iterator
    auto iter_vals = std::get<1>(pair_iters);    
    count_vals = std::distance(result_vals_begin,iter_vals);    

    // 3.Checking results by creating the host accessors    
    host_accessor result_keys(buf_out_keys,read_only);
    host_accessor result_vals(buf_out_vals,read_only); 
    

    std::cout<< "Keys = [ ";    
    std::copy(input_keys.begin(),input_keys.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Values = [ ";     
    std::copy(input_values.begin(),input_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Output Keys = [ ";    
    std::copy(output_keys.begin(),output_keys.begin() + count_keys,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";
    
    std::cout<< "Output Values = [ ";    
    std::copy(output_values.begin(),output_values.begin() + count_vals,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_reduce_segment.sh;if [ -x "$(command -v qsub)" ]; then ./q run_reduce_segment.sh; else ./run_reduce_segment.sh; fi

# ### inclusive_scan_by_segment
# The inclusive_scan_by_segment algorithm performs partial prefix scans on a sequence's values. Each scan applies to a contiguous subsequence of values, which are determined by the keys associated with the values being equal. The return value is an iterator targeting the end of the result sequence.
#
# For correct computation, the prefix scan operation should be associative. If no operation is specified, the default operation is std::plus, and the default predicate is std::equal_to. The algorithm requires that the type of the elements used for values be default constructible.
#
# __Example__:
#
# keys   [0,0,0,1,1,1]
#
# values [1,2,3,4,5,6]
#
# result [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

# The code below shows the usage of inclusive_scan_by_segment. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/inclusive_scan.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
    const int num_elements = 6;    
    auto R = range(num_elements);

    //Initialize the input vector for Keys
    std::vector<int> input_keys{ 0,0,0,1,1,1 };
    //Initialize the input vector for Values
    std::vector<int> input_values{ 1,2,3,4,5,6 };
    //Output vectors where we get the results back
    std::vector<int> output_values(num_elements, 0);

    //Create buffers for the above vectors   
    
    buffer buf_in(input_keys);
    buffer buf_seq(input_values);    
    buffer buf_out(output_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(queue(default_selector{}));
    std::cout << "Run on " << policy.queue().get_device().get_info<info::device::name>() << "\n";

    auto iter_res = oneapi::dpl::inclusive_scan_by_segment(policy, keys_begin, keys_end, vals_begin, result_begin);
    auto count_res = std::distance(result_begin,iter_res);

    // 3.Checking results    
    host_accessor result_vals(buf_out,read_only);
        
    std::cout<< "Keys = [ ";    
    std::copy(input_keys.begin(),input_keys.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Values = [ ";     
    std::copy(input_values.begin(),input_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Output Values = [ ";    
    std::copy(output_values.begin(),output_values.begin() + count_res,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_in_scan.sh;if [ -x "$(command -v qsub)" ]; then ./q run_in_scan.sh; else ./run_in_scan.sh; fi

# ### exclusive_scan_by_segment
# The exclusive_scan_by_segment algorithm performs partial prefix scans on a sequence's values. Each scan applies to a contiguous subsequence of values that are determined by the keys associated with the values being equal, and sets the first element to the initial value provided. The return value is an iterator targeting the end of the result sequence.
#
# For correct computation, the prefix scan operation should be associative. If no operation is specified, the default operation is std::plus, and the default predicate is std::equal_to.
#
# __Example__:
#
# keys:   [0,0,0,1,1,1]
#
# values: [1,2,3,4,5,6]
#
# initial value: [0]
#
# result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

# The code below shows the usage of exclusive_scan_by_segment. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/exclusive_scan.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;



int main() {
    using T = int;
    const int num_elements = 6;    
    auto R = range(num_elements);     

    //Initialize the input vector for Keys
    std::vector<int> input_keys{ 0,0,0,1,1,1 };
    //Initialize the input vector for Values
    std::vector<int> input_values{ 1,2,3,4,5,6 };
    //Output vectors where we get the results back
    std::vector<int> output_values(num_elements, 0);

    //Create buffers for the above vectors    
    
    buffer buf_in(input_keys);
    buffer buf_seq(input_values);
    //buffer buf_out(output_values);
    buffer buf_out(output_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(queue(default_selector{}));
    std::cout << "Run on " << policy.queue().get_device().get_info<info::device::name>() << "\n";

    auto iter_res = oneapi::dpl::exclusive_scan_by_segment(policy, keys_begin, keys_end, vals_begin, result_begin,T(0));
    auto count_res = std::distance(result_begin,iter_res);    

    // 3.Checking results    
    host_accessor result_vals(buf_out,read_only);

    std::cout<< "Keys = [ ";    
    std::copy(input_keys.begin(),input_keys.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Values = [ ";     
    std::copy(input_values.begin(),input_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Output Values = [ ";    
    std::copy(output_values.begin(),output_values.begin() + count_res,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_ex_scan.sh;if [ -x "$(command -v qsub)" ]; then ./q run_ex_scan.sh; else ./run_ex_scan.sh; fi

# ### Binary Search
#
# The __binary_search__ algorithm performs a binary search of the input sequence for each of the values in the search sequence provided. The result of a search for the i-th element of the search sequence, a boolean value indicating whether the search value was found in the input sequence, is assigned to the ith element of the result sequence.
#
# The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. The algorithm assumes the input sequence has been sorted by the comparator provided. If no comparator is provided then a function object that uses operator< to compare the elements will be used.
#
# __Example__:
#
# input sequence: [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
#
# search sequence: [0, 2, 4, 7, 6]
#
# result sequence: [true, true, false, false, true]
#

# The code below shows the usage of Binary search. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/binary_search.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
    //const int n = 10;
    //const int k = 5;

    const int num_elements = 5;    
    auto R = range(num_elements);    
    
    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for search
    std::vector<int> input_seq{0, 2, 2, 2, 3, 3, 3, 3, 6, 6};
    //Initialize the input vector for search pattern
    std::vector<int> input_pattern{0, 2, 4, 7, 6};
    //Output vector where we get the results back
    std::vector<int> output_values(num_elements,0); 
 
  
    //Create buffers for the above vectors    

    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);    
    buffer buf_out(output_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);  

    //function object to be passed to sort function  

    //Calling the dpstd binary search algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here. 
    const auto i =  oneapi::dpl::binary_search(policy,keys_begin,keys_end,vals_begin,vals_end,result_begin);
   

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);  

    std::cout<< "Input sequence = [";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search sequence = [";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search results = [";    
    std::copy(output_values.begin(),output_values.end(),std::ostream_iterator<bool>(std::cout," "));
    std::cout <<"]"<< "\n";  
  
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_binary_search.sh;if [ -x "$(command -v qsub)" ]; then ./q run_binary_search.sh; else ./run_binary_search.sh; fi

# ### lower_bound
#
# The lower_bound algorithm performs a binary search of the input sequence for each of the values in the search sequence provided to identify the lowest index in the input sequence where the search value could be inserted without violating the ordering provided by the comparator used to sort the input sequence. 
#
# The result of a search for the i-th element of the search sequence, the ﬁrst index in the input sequence where the search value could be inserted without violating the ordering of the input sequence, is assigned to the i-th element of the result sequence. The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. If no comparator is provided then a function object that uses operator< to compare the elements will be used.
#
# __Example__:
#
# input sequence: [0, 2, 2, 2, 3, 3, 3, 3, 6, 6] 
#
# search sequence: [0, 2, 4, 7, 6] 
#
# result sequence: [0, 1, 8, 10, 8]

# The code below shows the usage of lower bound. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/lower_bound.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
  
    const int num_elements = 5;
    auto R = range(num_elements); 

    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for search
    std::vector<int> input_seq{0, 2, 2, 2, 3, 3, 3, 3, 6, 6};
    //Initialize the input vector for search pattern
    std::vector<int> input_pattern{0, 2, 4, 7, 6};
    //Output vector where we get the results back
    std::vector<int> out_values(num_elements,0);    
 
      
    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);    
    buffer buf_out(out_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);  

    //Calling the dpstd upper_bound algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here.
    
    oneapi::dpl::lower_bound(policy,keys_begin,keys_end,vals_begin,vals_end,result_begin);   

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);
    
    std::cout<< "Input Sequence = [ ";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Sequence = [ ";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Results = [ ";    
    std::copy(out_values.begin(),out_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_lower_bound.sh;if [ -x "$(command -v qsub)" ]; then ./q run_lower_bound.sh; else ./run_lower_bound.sh; fi

# ### upper_bound
# The upper_bound algorithm performs a binary search of the input sequence for each of the values in the search sequence provided to identify the highest index in the input sequence where the search value could be inserted without violating the ordering provided by the comparator used to sort the input sequence. 
#
# The result of a search for the i-th element of the search sequence, the last index in the input sequence where the search value could be inserted without violating the ordering of the input sequence, is assigned to the i-th element of the result sequence. The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. If no comparator is provided then a function object that uses operator< to compare the elements will be used.
#
# __Example__:
#
# input sequence: [0, 2, 2, 2, 3, 3, 3, 3, 6, 6] 
#
# search sequence: [0, 2, 4, 7, 6] 
#
# result sequence: [1, 4, 8, 10, 10]

# The code below shows the usage of upper bound. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/upper_bound.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;


int main() {
  
    const int num_elements = 5;
    auto R = range(num_elements); 

    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for search
    std::vector<int> input_seq{0, 2, 2, 2, 3, 3, 3, 3, 6, 6};
    //Initialize the input vector for search pattern
    std::vector<int> input_pattern{0, 2, 4, 7, 6};
    //Output vector where we get the results back
    std::vector<int> out_values(num_elements,0);    
 
      
    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);    
    buffer buf_out(out_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);  

    //Calling the dpstd upper_bound algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here.
    
    oneapi::dpl::upper_bound(make_device_policy(q),keys_begin,keys_end,vals_begin,vals_end,result_begin);    

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);

    std::cout<< "Input Sequence = [ ";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Sequence = [ ";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Results = [ ";    
    std::copy(out_values.begin(),out_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_upper_bound.sh;if [ -x "$(command -v qsub)" ]; then ./q run_upper_bound.sh; else ./run_upper_bound.sh; fi

# ## oneDPL Custom Iterators
#
# The definitions of the iterators provided in the Extension API are available through the `oneapi/dpl/iterator` header. All iterators are implemented in the oneapi::dpl namespace.
# * __Zip Iterators__: zip_iterator is an iterator used for iterating over several containers simultaneously in STL algorithms
# * __Counting Iterators__: counting_iterator is a iterator for STL algorithms that changes the counter according to arithmetic operations of the iterator type
# * __Transform Iterators__: Iterator that applies a given function to an element of a sequence, and returns the result of the function.

# ### Zip Iterators
#
# #### What are Zip Iterators
# Zip iterators are part of the DPC++ library extensions API. zip_iterator is an iterator used for iterating over several containers simultaneously in STL algorithms.
#
# #### Why use Zip Iterators
# STL algorithms has limitation on the number of data sources it can operate on. This limitation comes the number of iterators we can provide as argument to a STL algorithm. Zip iterators enables relax this limitation.
#  we can create zip_iterator by calling the `oneapi::dpl::make_zip_iterator` function and the object is created based upon the types of parameters we pass to the function.
#
# Include the header `#include <oneapi/dpl/iterator>` to use zip Iterators.
#
#

# The code below shows the usage of Zip Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/zip_iterator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    constexpr int num_elements = 16;
    std::vector<int> input_v1(num_elements, 2), input_v2(num_elements, 5), input_v3(num_elements, 0);
    //Zip Iterator zips up the iterators of individual containers of interest.
    auto start = oneapi::dpl::make_zip_iterator(input_v1.begin(), input_v2.begin(), input_v3.begin());
    auto end = oneapi::dpl::make_zip_iterator(input_v1.end(), input_v2.end(), input_v3.end());
    //create device policy
    auto exec_policy = make_device_policy(q);
    std::for_each(exec_policy, start, end, [](auto t) {
        //The zip iterator is used for expressing bounds in PSTL algorithms.
        using std::get;
        get<2>(t) = get<1>(t) * get<0>(t);
        });
    for (auto it = input_v3.begin(); it < input_v3.end(); it++)
    std::cout << (*it) <<" ";
    std::cout << "\n";
    return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_zip_iterator.sh;if [ -x "$(command -v qsub)" ]; then ./q run_zip_iterator.sh; else ./run_zip_iterator.sh; fi

# ### Counting Iterators
#
# #### What are Counting Iterators
# __Counting Iterator__ is a iterator for STL algorithms that changes the counter according to arithmetic operations of the iterator type. 
#
# #### Why use Counting Iterators
# We use the iterator to get an index of a container element to make some calculations. The counter changes according to arithmetics of the random access iterator type. 
# we can create counting iterator by calling the `oneapi::dpl::counting_iterator` function and the object is created based upon the types of parameters we pass to the function.
#
# Include the header `#include <oneapi/dpl/iterator` to use counting Iterators. 
#

# The code below shows the usage of counting Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/counting_iterator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>


using namespace sycl;
using namespace oneapi::dpl::execution;
int main() {
    
    oneapi::dpl::counting_iterator<int> count_a(0);
    oneapi::dpl::counting_iterator<int> count_b = count_a + 100;
    int init = count_a[0]; // OK: init == 0
    //*count_b = 7; // ERROR: counting_iterator doesn't provide write operations
    auto sum = std::reduce(dpl::execution::dpcpp_default,
     count_a, count_b, init); // sum is (0 + 0 + 1 + ... + 99) = 4950
    std::cout << "The Sum is: " <<sum<<"\n";
    
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_counting_iterator.sh;if [ -x "$(command -v qsub)" ]; then ./q run_counting_iterator.sh; else ./run_counting_iterator.sh; fi

# ### Transform Iterators
#
# #### What are Transform Iterators
# Transform Iterator applies a given function to an element of a sequence, and returns the result of the function.
#
# #### Why use Transform Iterators
# A transform_iterator is a random-access iterator that applies a transformation to a sequence. The transformation, a given function, is applied upon dereferencing of the iterator itself to the dereferenced value of an underlying iterator. Expressing a pattern this way can be efficient since the transformed sequence can be consumed for example by an algorithm without storing temporary values in memory. 
#
# Function `dpl::make_transform_iterator` returns a transform_iterator object with underlying iterator and custom functor set.
#
# Include the header `#include <oneapi/dpl/iterator` to use Transform Iterators.

# The code below shows the usage of Transform Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/transform_iterator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

using namespace sycl;
using namespace oneapi::dpl::execution;
        
int main() {
    
    dpl::counting_iterator<int> first(0);
    dpl::counting_iterator<int> last(100);
    auto func = [](const auto &x){ return x * 2; };
    auto transform_first = dpl::make_transform_iterator(first, func);
    auto transform_last = transform_first + (last - first);
    auto sum = std::reduce(dpl::execution::dpcpp_default,
         transform_first, transform_last); // sum is (0 + -1 + ... + -9) = -45   
    std::cout <<"Reduce output using Transform Iterator: "<<sum << "\n";
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_transform_iterator.sh;if [ -x "$(command -v qsub)" ]; then ./q run_transform_iterator.sh; else ./run_counting_iterator.sh; fi

# ### discard_iterator
#
# #### What are Discard Iterators
# The discard_iterator is a random access iterator-like type that provides write-only dereference operations that discard values passed.
#
# #### Why use Discard Iterators
# The iterator is useful in the implementation of stencil algorithms, where the stencil is not part of the desired output. An example of this would be a copy_if algorithm that receives an input iterator range and a stencil iterator range, then copies the elements of the input whose corresponding stencil value is 1. You must use the discard_iterator to avoid declaring a temporary allocation to store a copy of the stencil.

# The code below shows the usage of Transform Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# +
# %%writefile lab/discard_iterator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>

#include <tuple>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;
using std::get;


int main() {

    const int num_elements = 10;

    //Initialize the input vector for search
    std::vector<int> input_seq{2, 4, 12, 24, 34, 48, 143, 63, 76, 69};
    //Initialize the stencil values
    std::vector<int> input_pattern{1, 2, 4, 1, 6, 1, 2, 1, 7, 1};
    //Output vector where we get the results back
    std::vector<int> out_values(num_elements,0);


    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);
    buffer buf_out(out_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = oneapi::dpl::execution::dpcpp_default;

    auto zipped_first = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);

    auto iter_res = std::copy_if(dpl::execution::dpcpp_default,zipped_first, zipped_first + num_elements,
                 dpl::make_zip_iterator(result_begin, dpl::discard_iterator()),
                 [](auto t){return get<1>(t) == 1;});    
    

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);

    std::cout<< "Input Sequence = [ ";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Sequence to search = [ ";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Results with stencil value of 1 = [ ";    
    std::copy(out_values.begin(),out_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}

# -

# ! chmod 755 q; chmod 755 run_discard_iterator.sh;if [ -x "$(command -v qsub)" ]; then ./q run_discard_iterator.sh; else ./run_discard_iterator.sh; fi

# ### permutation_iterator
#
# #### What are permutation Iterators
# The permutation_iterator is an iterator whose dereferenced value set is defined by the source iterator provided. Its iteration order over the dereferenced value set is defined by either another iterator, or a functor whose index operator defines the mapping from the permutation_iterator index to the index of the source iterator
#
# #### Why use permutation Iterators
# permutation_iterator is useful in implementing applications where noncontiguous elements of data, represented by an iterator, need to be processed by an algorithm as though they were contiguous. An example is copying every other element to an output iterator.
# The make_permutation_iterator is provided to simplify the construction of iterator instances. The function receives the source iterator and the iterator, or function object, representing the index map.

# The code below shows the usage of Transform Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/permutation_iterator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace std;

struct multiply_index_by_two {
    template <typename Index>
    Index operator[](const Index& i) const
    {
        return i * 2;
    }
};

int main() {
    //queue q;
    const int num_elelemts = 100;
    std::vector<float> result(num_elelemts, 0);
    oneapi::dpl::counting_iterator<int> first(0);
    oneapi::dpl::counting_iterator<int> last(20);

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    using namespace oneapi;
    auto permutation_first = oneapi::dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;
    auto it = ::std::copy(oneapi::dpl::execution::dpcpp_default, permutation_first, permutation_last, result.begin());
    auto count = ::std::distance(result.begin(),it);
    
    for(int i = 0; i < count; i++) ::std::cout << result[i] << " ";
    
   // for (auto it = result.begin(); it < result.end(); it++)     
    //   ::std::cout << (*it) <<" "; 
        
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_permutation_iterator.sh;if [ -x "$(command -v qsub)" ]; then ./q run_permutation_iterator.sh; else ./permutation_iterator.sh; fi

# ### Range-Based API
#
# C++20 introduces the Ranges library. The С++20 standard splits ranges into two categories: factories and adaptors. A range factory does not have underlying data. An element is generated on success by an index, or by dereferencing an iterator. A range adaptor, from a oneDPL perspective, is an utility that transforms base range, or another adapted range into a view with custom behavior. 
#
# oneDPL supports an iota_view range factory. A sycl::buffer wrapped with all_view can be used as the range. oneDPl considers the supported factories and all_view as base ranges. The range adaptors may be combined into a pipeline with a base range at the beginning.
#
# The code below shows the usage of Ranges. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/ranges.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include<oneapi/dpl/execution>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/ranges>
#include<iostream>
#include<vector>
#include<CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::experimental::ranges;

int main()
{
    std::vector<int> v(20);

    {
        buffer A(v);
        auto view = iota_view(0, 20);
        auto rev_view = views::reverse(view);
        auto range_res = all_view<int, cl::sycl::access::mode::write>(A);

        copy(oneapi::dpl::execution::dpcpp_default, rev_view, range_res);
    }

    for (auto x : v)
        std::cout << x << " ";
    std::cout << "\n";
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_ranges.sh;if [ -x "$(command -v qsub)" ]; then ./q run_ranges.sh ; else ./run_ranges.sh; fi

# ## Functional Utility classes
# identity: A C++11 implementation of the C++20 std::identity function object type, where the operator() returns the argument unchanged.
#
# minimum: A function object type where the operator() applies std::less to its arguments, and then returns the lesser argument unchanged.
#
# maximum: A function object type where the operator() applies std::greater to its arguments, and then returns the greater argument unchanged.

# ### Minimum
# * minimum: A function object type where the operator() applies std::less to its arguments, and then returns the lesser argument unchanged.
#
# Below code uses the dpstd::minimum when performing the std::exclusive_scan function.

# The code below shows the usage of __Minimum__ utility class. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/minimum_function.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

using namespace sycl;
using namespace oneapi::dpl::execution;
        
int main() {
    
    queue q;
    constexpr int N = 8;
    //Input vector
    std::vector<int> v{3,-1,-4,1,-5,-9,2,6};
    //create a separate scope for buffer destruction
    std::vector<int>result(N);
    {
        buffer buf(v);
        buffer buf_res(result);

        //dpstd buffer iterators for both the input and the result vectors
        auto start_v = oneapi::dpl::begin(buf);
        auto end_v = oneapi::dpl::end(buf);
        auto start_res = oneapi::dpl::begin(buf_res);
        auto end_res = oneapi::dpl::end(buf_res);
        
        //use std::fill to initialize the result vector
        std::fill(oneapi::dpl::execution::dpcpp_default,start_res, end_res, 0);  
        //usage of dpl::minimum<> function call within the std::exclusive_scan function
        std::exclusive_scan(oneapi::dpl::execution::dpcpp_default, start_v, end_v, start_res, int(0), oneapi::dpl::minimum<int>() );        
    }    
    
    for(int i = 0; i < result.size(); i++) std::cout << result[i] << "\n";
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_min.sh;if [ -x "$(command -v qsub)" ]; then ./q run_min.sh; else ./run_min.sh; fi

# ### Maximum
# * maximum: A function object type where the operator() applies std::greater to its arguments, and then returns the greater argument unchanged.
# Below code uses the dpstd::maximum when performing the std::exclusive_scan function.

# The code below shows the usage of __Maximum__ utility class. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/maximum_function.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

using namespace sycl;
using namespace oneapi::dpl::execution;
        
int main() {
    
    queue q;
    constexpr int N = 8;
    
    std::vector<int> v{-3,1,4,-1,5,9,-2,6}; 
    //create a separate scope for buffer destruction
    std::vector<int>result(N);
    {
        buffer<int,1> buf(v.data(), range<1>(N));
        buffer<int,1> buf_res(result.data(), range<1>(N));
        
        //dpstd buffer iterators for both the input and the result vectors
        auto start_v = oneapi::dpl::begin(buf);
        auto end_v = oneapi::dpl::end(buf);
        auto start_res = oneapi::dpl::begin(buf_res);
        auto end_res = oneapi::dpl::end(buf_res);
        
        //use std::fill to initialize the result vector
        std::fill(oneapi::dpl::execution::dpcpp_default,start_res, end_res, 0);  
        //usage of dpstd::maximum<> function call within the std::exclusive_scan function
        std::exclusive_scan(oneapi::dpl::execution::dpcpp_default, start_v, end_v, start_res, int(0), oneapi::dpl::maximum<int>() );        
    }
    
    
    for(int i = 0; i < result.size(); i++) std::cout << result[i] << "\n";
    return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_max.sh;if [ -x "$(command -v qsub)" ]; then ./q run_max.sh; else ./run_max.sh; fi

# ## zip iterators and stable sort example
#
# The below example walks us through performing a `std::stablesort` using the zip iterators and counting iterator.
#
# counting_iterator is a iterator for STL algorithms that changes the counter according to arithmetic operations of the iterator type. We use the iterator to get an index of a container element to make some calculation.
#
# We create two SYCL buffers for holding __keys__ and __values__. We create the iterators using the `oneapi::dpl::begin` for these buffers. We create the `counting_iterator` to actually iterate through the buffer to get the index space for each element. We initialize the buffer holding the keys value using std::tranform function. Note the execution policy object we as passing as parameter to the `std::transform` function. Then we fill vals_buf with the analogous of std::iota using counting_iterator where we are assigning successive values in that range.
#
# The second part is the sorting. We are creating the zip iteartor using the `make_zip_iterator()` function where we are passing the buffers for the keys and values. Finally we call the `std::stable_sort()` passing the policy object and the zip iterator object as parameter to sort by keys. Run the program and observe the output.
#
#

# The code below shows the stable sort algorithm using Zip Iterators. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile stable_sort_by_key/src/main.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <CL/sycl.hpp>


using namespace sycl;
using namespace oneapi::dpl::execution;


using namespace std;

int main() {
  const int n = 1000000;
  buffer<int> keys_buf{n};  // buffer with keys
  buffer<int> vals_buf{n};  // buffer with values

  // create objects to iterate over buffers
  auto keys_begin = oneapi::dpl::begin(keys_buf);
  auto vals_begin = oneapi::dpl::begin(vals_buf);

  auto counting_begin = oneapi::dpl::counting_iterator<int>{0};
  // use default policy for algorithms execution
  auto policy = oneapi::dpl::execution::dpcpp_default;

  // 1. Initialization of buffers
  // let keys_buf contain {n, n, n-2, n-2, ..., 4, 4, 2, 2}
  transform(policy, counting_begin, counting_begin + n, keys_begin,
            [n](int i) { return n - (i / 2) * 2; });
  // fill vals_buf with the analogue of std::iota using counting_iterator
  copy(policy, counting_begin, counting_begin + n, vals_begin);

  // 2. Sorting
  auto zipped_begin = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);
  // stable sort by keys
  stable_sort(
      policy, zipped_begin, zipped_begin + n,
      // Generic lambda is needed because type of lhs and rhs is unspecified.
      [](auto lhs, auto rhs) { return get<0>(lhs) < get<0>(rhs); });

  // 3.Checking results
  //host_accessor host_keys(keys_buf,read_only);
  //host_accessor host_vals(vals_buf,read_only);
  auto host_keys = keys_buf.get_access<access::mode::read>();
  auto host_vals = vals_buf.get_access<access::mode::read>();

  // expected output:
  // keys: {2, 2, 4, 4, ..., n - 2, n - 2, n, n}
  // vals: {n - 2, n - 1, n - 4, n - 3, ..., 2, 3, 0, 1}
  for (int i = 0; i < n; ++i) {
    if (host_keys[i] != (i / 2) * 2 &&
        host_vals[i] != n - (i / 2) * 2 - (i % 2 == 0 ? 2 : 1)) {
      cout << "fail: i = " << i << ", host_keys[i] = " << host_keys[i]
           << ", host_vals[i] = " << host_vals[i] << "\n";
      return 1;
    }
  }

  cout << "success\nRun on "
       << policy.queue().get_device().template get_info<info::device::name>()
       << "\n";
  return 0;
}


# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_stable_sort.sh;if [ -x "$(command -v qsub)" ]; then ./q run_stable_sort.sh; else ./run_stable_sort.sh; fi

# # Summary
# In this module you will have learned the following:
# * The oneDPL extension API algorithms, iterators and the utility classes 
#
#
#

# <html><body><span style="color:Red"><h1>Reset Notebook</h1></span></body></html>
#
# ##### Should you be experiencing any issues with your notebook or just want to start fresh run the below cell.
#
#

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
