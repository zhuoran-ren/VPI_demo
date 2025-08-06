#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

class LookupTable {
public:
    // Constructor: load grid and value table from binary file
    LookupTable(const std::string& filename) {
        load_from_file(filename);
    }

    // Perform 4D linear interpolation on the lookup table
    float get_value_interp(float x, float y, float vp, float vv) const {
        int ix = find_interval(X_REL_GRID, x);
        int iy = find_interval(Y_REL_GRID, y);
        int ip = find_interval(VEL_P_GRID, vp);
        int iv = find_interval(VEL_V_GRID, vv);

        float x1 = X_REL_GRID[ix],     x2 = X_REL_GRID[ix + 1];
        float y1 = Y_REL_GRID[iy],     y2 = Y_REL_GRID[iy + 1];
        float p1 = VEL_P_GRID[ip],     p2 = VEL_P_GRID[ip + 1];
        float v1 = VEL_V_GRID[iv],     v2 = VEL_V_GRID[iv + 1];

        float xd = (x2 - x1 > 0) ? (x - x1) / (x2 - x1) : 0;
        float yd = (y2 - y1 > 0) ? (y - y1) / (y2 - y1) : 0;
        float pd = (p2 - p1 > 0) ? (vp - p1) / (p2 - p1) : 0;
        float vd = (v2 - v1 > 0) ? (vv - v1) / (v2 - v1) : 0;

        float result = 0.0f;

        // Trilinear interpolation over 4 dimensions
        for (int dx = 0; dx <= 1; ++dx)
        for (int dy = 0; dy <= 1; ++dy)
        for (int dp = 0; dp <= 1; ++dp)
        for (int dv = 0; dv <= 1; ++dv) {
            float w = ((dx ? xd : 1 - xd) *
                       (dy ? yd : 1 - yd) *
                       (dp ? pd : 1 - pd) *
                       (dv ? vd : 1 - vd));

            int idx = flatten(ix + dx, iy + dy, ip + dp, iv + dv);
            result += w * value_table[idx];
        }

        return result;
    }

private:
    int NX, NY, NP, NV;
    std::vector<float> X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID;
    std::vector<float> value_table;

    void load_from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: cannot open file " << filename << "\n";
            exit(1);
        }

        int dims[4];
        file.read(reinterpret_cast<char*>(dims), sizeof(dims));
        NX = dims[0], NY = dims[1], NP = dims[2], NV = dims[3];

        X_REL_GRID.resize(NX);
        Y_REL_GRID.resize(NY);
        VEL_P_GRID.resize(NP);
        VEL_V_GRID.resize(NV);

        file.read(reinterpret_cast<char*>(X_REL_GRID.data()), NX * sizeof(float));
        file.read(reinterpret_cast<char*>(Y_REL_GRID.data()), NY * sizeof(float));
        file.read(reinterpret_cast<char*>(VEL_P_GRID.data()), NP * sizeof(float));
        file.read(reinterpret_cast<char*>(VEL_V_GRID.data()), NV * sizeof(float));

        size_t total = static_cast<size_t>(NX) * NY * NP * NV;
        value_table.resize(total);
        file.read(reinterpret_cast<char*>(value_table.data()), total * sizeof(float));

        if (!file) {
            std::cerr << "Error while reading value table.\n";
            exit(1);
        }

        file.close();
    }

    int flatten(int i, int j, int p, int v) const {
        return ((i * NY + j) * NP + p) * NV + v;
    }

    int find_interval(const std::vector<float>& grid, float val) const {
        if (val <= grid.front()) return 0;
        if (val >= grid.back()) return grid.size() - 2;
        for (size_t i = 0; i < grid.size() - 1; ++i) {
            if (val >= grid[i] && val <= grid[i + 1]) return i;
        }
        return grid.size() - 2;
    }
};
