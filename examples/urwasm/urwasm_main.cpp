#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <time.h>

extern "C" {
    void  llama_cpp_init_from_gguf_bytes(uint8_t *bytes, uint32_t len);
    char* infer_c_string(const char *prompt, int *token_count);
}

[[noreturn]] static void
bail(const char *msg)
{
    perror(msg);
    std::exit(1);
}

static inline uint64_t now_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        bail("clock_gettime");
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void print_elapsed_ms(const char *label, uint64_t start_ns, uint64_t end_ns) {
    const uint64_t dt = end_ns - start_ns;
    const double ms = (double)dt / 1e6;
    std::fprintf(stderr, "%s: %.3f ms\n", label, ms);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/model.gguf\n", argv[0]);
        return 2;
    }

    const char *path = argv[1];

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) bail("open");

    struct stat st;
    if (::fstat(fd, &st) != 0) bail("fstat");

    const size_t len = (size_t) st.st_size;
    void *map = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) bail("mmap");

    uint8_t *buf = (uint8_t*) std::malloc(len);
    if (!buf) bail("malloc");
    std::memcpy(buf, map, len);

    ::munmap(map, len);
    ::close(fd);

    std::fprintf(stderr, "init begin\n");
    const uint64_t t0 = now_ns();
    llama_cpp_init_from_gguf_bytes(buf, (uint32_t)len);
    const uint64_t t1 = now_ns();
    print_elapsed_ms("init done:", t0, t1);

    const char *prompt = "Q: What is the capital of the United Kingdom?\nA: ";

    std::fprintf(stderr, "prompt:\r\n%s\r\nend of prompt\r\n", prompt);
    
    int token_count = 0;

    std::fprintf(stderr, "infer begin\n");
    const uint64_t t2 = now_ns();
    char *ans = infer_c_string(prompt, &token_count);
    const uint64_t t3 = now_ns();
    print_elapsed_ms("infer done:", t2, t3);

    if (!ans) {
        std::fprintf(stderr, "infer failed\n");
        return 1;
    }

    const uint64_t infer_ns = t3 - t2;
    const double infer_s = (double) infer_ns / 1e9;

    if (infer_s > 0.0) {
        const double tps = (double) token_count / infer_s;
        std::fprintf(stderr, "tokens: %d\n", token_count);
        std::fprintf(stderr, "tokens/sec: %.3f\n", tps);
    } else {
        std::fprintf(stderr, "tokens: %d\n", token_count);
        std::fprintf(stderr, "tokens/sec: inf\n");
    }

    std::puts(">>>");
    std::puts(ans);
    std::free(ans);

    return 0;
}