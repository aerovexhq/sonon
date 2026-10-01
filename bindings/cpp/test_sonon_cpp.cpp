/**
 * Automated test for Sonon C++20 header-only shared memory client.
 */

#include "include/sonon.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    const std::string test_path = "/tmp/test_sonon_cpp.bin";
    ::unlink(test_path.c_str());

    sonon::SononClient client(test_path, 16000.0f, 1024);
    assert(client.is_valid());

    // Write samples
    std::vector<float> test_samples(512);
    for (size_t i = 0; i < test_samples.size(); ++i) {
        test_samples[i] = static_cast<float>(i) * 0.002f;
    }

    size_t written = client.write_samples(test_samples);
    assert(written == 512);

    // Read back samples
    std::vector<float> read_back = client.read_samples();
    assert(read_back.size() == 512);
    for (size_t i = 0; i < 512; ++i) {
        assert(std::fabs(read_back[i] - test_samples[i]) < 1e-5f);
    }

    // Health check
    auto health = client.poll_health();
    assert(health.has_value());
    assert(health->health_score == 1.0f);
    assert(health->worst_severity == 0);

    ::unlink(test_path.c_str());
    std::cout << "[PASS] Sonon C++20 Client test passed.\n";
    return 0;
}
