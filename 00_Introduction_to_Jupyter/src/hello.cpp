#include <iostream>
#define RESET   "\033[0m"
#define RED     "\033[31m"    /* Red */
#define BLUE    "\033[34m"    /* Blue */

int main(){
    std::cout << RED << "Hello World" << RESET << std::endl;
}
