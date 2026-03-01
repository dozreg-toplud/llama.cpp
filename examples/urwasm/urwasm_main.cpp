#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" {
    void  load_gguf_bytes(uint8_t *bytes, uint32_t len);
    char* infer_c_string(const char *prompt);
}

[[noreturn]] static void
bail(const char *msg)
{
    perror(msg);
    std::exit(1);
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

    size_t len = (size_t)st.st_size;
    void *map = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) bail("mmap");

    uint8_t *buf = (uint8_t*)std::malloc(len);
    if (!buf) bail("malloc");
    std::memcpy(buf, map, len);

    ::munmap(map, len);
    ::close(fd);

    load_gguf_bytes(buf, (uint32_t)len);

    const char *prompt = "What is the capital of the United Kingdom?";
    char *ans = infer_c_string(prompt);
    if (!ans) {
        std::fprintf(stderr, "infer failed\n");
        return 1;
    }

    std::puts(ans);
    std::free(ans);

    return 0;
}