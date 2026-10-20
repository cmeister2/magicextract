// Timing-only adapter; the pinned author's ELSED source is compiled unchanged.
#include "ELSED.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
std::string token(std::istream& stream) {
    std::string value;
    while (stream >> value) {
        if (value[0] != '#') return value;
        std::getline(stream, value);
    }
    throw std::runtime_error("truncated PGM header");
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: timed_cli image.pgm iterations");
        std::ifstream stream(argv[1], std::ios::binary);
        if (token(stream) != "P5") throw std::runtime_error("binary PGM required");
        const int width = std::stoi(token(stream)), height = std::stoi(token(stream));
        if (std::stoi(token(stream)) != 255) throw std::runtime_error("8-bit PGM required");
        stream.get();
        cv::Mat image(height, width, CV_8UC1);
        stream.read(reinterpret_cast<char*>(image.data), width * height);
        if (stream.gcount() != width * height) throw std::runtime_error("truncated PGM pixels");
        cv::setNumThreads(1);
        upm::ELSEDParams parameters;
        parameters.ksize = 7;
        upm::ELSED detector(parameters);
        for (int i = 0; i < 5; ++i) detector.detectSalient(image);
        const int iterations = std::stoi(argv[2]);
        std::vector<double> milliseconds;
        size_t count = 0;
        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            const auto segments = detector.detectSalient(image);
            const auto end = std::chrono::steady_clock::now();
            count = segments.size();
            milliseconds.push_back(std::chrono::duration<double, std::milli>(end-start).count());
        }
        std::sort(milliseconds.begin(), milliseconds.end());
        std::cout << std::setprecision(9) << "{\"segments\":" << count
                  << ",\"detector_ms_median\":" << milliseconds[milliseconds.size()/2]
                  << ",\"detector_ms_p95\":" << milliseconds[(milliseconds.size()-1)*95/100]
                  << ",\"iterations\":" << iterations << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
