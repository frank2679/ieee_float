#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

int main() {
    float a = std::numeric_limits<float>::denorm_min();
    float b = std::numeric_limits<float>::denorm_min();
    float sum = a + b;
    std::cout << "a = " << std::hexfloat << a << std::endl;
    std::cout << "b = " << std::hexfloat << b << std::endl;
    std::cout << "sum = " << std::hexfloat << sum << std::endl;

    float f = 3.14159;
    std::cout << std::hexfloat << f << std::endl;
    return 0;
}
