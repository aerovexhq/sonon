/**
 * @file sonon.hpp
 * @brief Modern C++20 Header-Only Client for Sonon Acoustic Engine.
 *
 * Provides zero-copy POSIX shared memory (/dev/shm/sonon_audio) audio streaming,
 * keyword spotting event polling, and drone acoustic health monitoring with RAII.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace sonon {

constexpr uint32_t SHM_MAGIC = 0x534F4E4F; // "SONO"
constexpr uint32_t SHM_VERSION = 1;
constexpr const char* DEFAULT_SHM_PATH = "/dev/shm/sonon_audio";
constexpr size_t DEFAULT_CAPACITY = 65536;
constexpr size_t HEADER_SIZE = 64;

#pragma pack(push, 1)
struct ShmHeaderRaw {
    uint32_t magic;
    uint32_t version;
    float sample_rate;
    uint32_t channels;
    uint32_t capacity;
    uint64_t write_head;
    uint64_t read_head;
    float health_score;
    uint32_t worst_severity;
    float last_keyword_confidence;
    uint32_t last_keyword_len;
    char last_keyword[12];
    uint8_t reserved[4];
};
#pragma pack(pop)

struct KeywordEvent {
    std::string keyword;
    float confidence;
};

struct HealthStatus {
    float health_score;
    uint32_t worst_severity;
};

class SononClient {
public:
    explicit SononClient(const std::string& path = DEFAULT_SHM_PATH,
                         float sample_rate = 16000.0f,
                         size_t capacity = DEFAULT_CAPACITY)
        : path_(path), sample_rate_(sample_rate), capacity_(capacity), fd_(-1) {
        init_channel();
    }

    ~SononClient() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    SononClient(const SononClient&) = delete;
    SononClient& operator=(const SononClient&) = delete;

    SononClient(SononClient&& other) noexcept
        : path_(std::move(other.path_)),
          sample_rate_(other.sample_rate_),
          capacity_(other.capacity_),
          fd_(other.fd_) {
        other.fd_ = -1;
    }

    SononClient& operator=(SononClient&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            path_ = std::move(other.path_);
            sample_rate_ = other.sample_rate_;
            capacity_ = other.capacity_;
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    bool is_valid() const noexcept { return fd_ >= 0; }

    size_t write_samples(const float* samples, size_t count) {
        if (fd_ < 0 || !samples || count == 0) return 0;

        uint64_t write_head = 0;
        ::lseek(fd_, 20, SEEK_SET);
        if (::read(fd_, &write_head, sizeof(write_head)) != sizeof(write_head)) {
            return 0;
        }

        size_t start_slot = write_head % capacity_;
        size_t end_slot = start_slot + count;

        if (end_slot <= capacity_) {
            ::lseek(fd_, HEADER_SIZE + start_slot * sizeof(float), SEEK_SET);
            ::write(fd_, samples, count * sizeof(float));
        } else {
            size_t first_count = capacity_ - start_slot;
            size_t second_count = count - first_count;

            ::lseek(fd_, HEADER_SIZE + start_slot * sizeof(float), SEEK_SET);
            ::write(fd_, samples, first_count * sizeof(float));

            ::lseek(fd_, HEADER_SIZE, SEEK_SET);
            ::write(fd_, samples + first_count, second_count * sizeof(float));
        }

        uint64_t new_write_head = write_head + count;
        ::lseek(fd_, 20, SEEK_SET);
        ::write(fd_, &new_write_head, sizeof(new_write_head));

        return count;
    }

    size_t write_samples(const std::vector<float>& samples) {
        return write_samples(samples.data(), samples.size());
    }

    std::vector<float> read_samples() {
        if (fd_ < 0) return {};

        uint64_t write_head = 0, read_head = 0;
        ::lseek(fd_, 20, SEEK_SET);
        if (::read(fd_, &write_head, sizeof(write_head)) != sizeof(write_head)) return {};
        if (::read(fd_, &read_head, sizeof(read_head)) != sizeof(read_head)) return {};

        if (write_head <= read_head) return {};

        uint64_t unread = write_head - read_head;
        if (unread > capacity_) {
            read_head = write_head - capacity_;
            unread = capacity_;
        }

        size_t count = static_cast<size_t>(unread);
        std::vector<float> result(count);

        size_t start_slot = read_head % capacity_;
        size_t end_slot = start_slot + count;

        if (end_slot <= capacity_) {
            ::lseek(fd_, HEADER_SIZE + start_slot * sizeof(float), SEEK_SET);
            ::read(fd_, result.data(), count * sizeof(float));
        } else {
            size_t first_count = capacity_ - start_slot;
            size_t second_count = count - first_count;

            ::lseek(fd_, HEADER_SIZE + start_slot * sizeof(float), SEEK_SET);
            ::read(fd_, result.data(), first_count * sizeof(float));

            ::lseek(fd_, HEADER_SIZE, SEEK_SET);
            ::read(fd_, result.data() + first_count, second_count * sizeof(float));
        }

        ::lseek(fd_, 28, SEEK_SET);
        ::write(fd_, &write_head, sizeof(write_head));

        return result;
    }

    std::optional<HealthStatus> poll_health() {
        if (fd_ < 0) return std::nullopt;

        float score = 1.0f;
        uint32_t severity = 0;
        ::lseek(fd_, 36, SEEK_SET);
        if (::read(fd_, &score, sizeof(score)) != sizeof(score)) return std::nullopt;
        if (::read(fd_, &severity, sizeof(severity)) != sizeof(severity)) return std::nullopt;

        return HealthStatus{score, severity};
    }

    std::optional<KeywordEvent> poll_detection() {
        if (fd_ < 0) return std::nullopt;

        float confidence = 0.0f;
        uint32_t kw_len = 0;
        char kw_buf[12] = {0};

        ::lseek(fd_, 44, SEEK_SET);
        if (::read(fd_, &confidence, sizeof(confidence)) != sizeof(confidence)) return std::nullopt;
        if (::read(fd_, &kw_len, sizeof(kw_len)) != sizeof(kw_len)) return std::nullopt;
        if (::read(fd_, kw_buf, sizeof(kw_buf)) != sizeof(kw_buf)) return std::nullopt;

        if (kw_len == 0 || confidence <= 0.0f) return std::nullopt;

        size_t len = std::min(static_cast<size_t>(kw_len), sizeof(kw_buf));
        return KeywordEvent{std::string(kw_buf, len), confidence};
    }

private:
    void init_channel() {
        size_t total_size = HEADER_SIZE + capacity_ * sizeof(float);
        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) return;

        struct stat st;
        if (::fstat(fd_, &st) == 0 && static_cast<size_t>(st.st_size) < total_size) {
            ::ftruncate(fd_, total_size);

            ShmHeaderRaw header{};
            header.magic = SHM_MAGIC;
            header.version = SHM_VERSION;
            header.sample_rate = sample_rate_;
            header.channels = 1;
            header.capacity = static_cast<uint32_t>(capacity_);
            header.health_score = 1.0f;

            ::lseek(fd_, 0, SEEK_SET);
            ::write(fd_, &header, sizeof(header));
        }
    }

    std::string path_;
    float sample_rate_;
    size_t capacity_;
    int fd_;
};

} // namespace sonon
