#include <iostream>
#include "LookupTable.hpp"

int main() {
    // Initialize lookup table from binary file
    std::string filepath = "data/Strategic_value_table.bin";
    LookupTable table(filepath);

    // Query a target state using interpolation
    float x_rel = 2.34f;
    float y_rel = -3.0f;
    float v_p = 0.4f;
    float v_v = 0.6f;

    float value = table.get_value_interp(x_rel, y_rel, v_p, v_v);

    // Display the result
    std::cout << "Interpolated value at ("
              << "x_rel = " << x_rel << ", "
              << "y_rel = " << y_rel << ", "
              << "v_p = "   << v_p << ", "
              << "v_v = "   << v_v << ") is: "
              << value << std::endl;

    // Pause to view output (useful when double-clicking .exe)
    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}
