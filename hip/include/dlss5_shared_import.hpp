// D3D12 shared handle -> dma-buf fd -> HIP device pointer, inside a Proton game (2026-09-29).
//
// Ported from guentra/dlss5-amd-hip-linux hip/src/gpu_interop.hip (cea4e91b, 3d1b3956, b0e2a21; MIT):
//  - Proton 10 D3DKMT NT handles are not file handles (wine_server_handle_to_fd says
//    STATUS_BAD_DEVICE_TYPE); the 0x23400C dxgk ioctl that winevulkan uses turns one into a file-type
//    handle first, and wine_server_handle_to_fd unwraps that with FILE_READ_DATA;
//  - the fd is imported with hipImportExternalMemory (OPAQUE_FD) and mapped as one buffer.
// Left out on purpose: upstream's win32u .symtab scanner fallback (it crashed a game once).
//
// Every step's result is collected into `why`, because a log file written here lands in the container's
// private /tmp: the error string is the one channel that reaches OptiScaler.log.
//
// Imports are cached for the process lifetime keyed by (handle, size, generation): handle values are
// reused after CloseHandle, so the caller bumps the generation whenever it recreates its buffers.
#pragma once
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace dlss5 {
namespace shared {

typedef int (*wine_h2fd_t)(void*, unsigned, int*, unsigned*);
typedef int (*nt_ioctl_fn)(void*, void*, void*, void*, void*, unsigned, void*, unsigned, void*, unsigned);
typedef int (*nt_close_fn)(void*);

inline void note(std::string& why, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
inline void note(std::string& why, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!why.empty()) why += "; ";
    why += buf;
}

// A non-exported function of a loaded library, from its .symtab (absent if the file is stripped).
// Every offset is checked against the file size before it is read.
inline void* elf_local_sym(void* handle, const char* name) {
    struct link_map* lm = nullptr;
    if (!handle || dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm || !lm->l_name || !lm->l_name[0]) return nullptr;
    const int fd = open(lm->l_name, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(ElfW(Ehdr))) {
        close(fd);
        return nullptr;
    }
    const size_t size = size_t(st.st_size);
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return nullptr;
    const char* bytes = (const char*)map;
    void* found = nullptr;
    const ElfW(Ehdr)* eh = (const ElfW(Ehdr)*)map;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) == 0 && eh->e_shentsize == sizeof(ElfW(Shdr)) && eh->e_shoff <= size &&
        size_t(eh->e_shnum) * sizeof(ElfW(Shdr)) <= size - eh->e_shoff) {
        const ElfW(Shdr)* sh = (const ElfW(Shdr)*)(bytes + eh->e_shoff);
        const ElfW(Shdr)* symtab = nullptr;
        const ElfW(Shdr)* strtab = nullptr;
        for (int i = 0; i < eh->e_shnum; ++i)
            if (sh[i].sh_type == SHT_SYMTAB) {
                symtab = &sh[i];
                if (sh[i].sh_link < eh->e_shnum) strtab = &sh[sh[i].sh_link];
                break;
            }
        if (symtab && strtab && symtab->sh_entsize == sizeof(ElfW(Sym)) && symtab->sh_offset <= size &&
            symtab->sh_size <= size - symtab->sh_offset && strtab->sh_offset <= size &&
            strtab->sh_size <= size - strtab->sh_offset) {
            const ElfW(Sym)* sym = (const ElfW(Sym)*)(bytes + symtab->sh_offset); // byte offset, not index
            const char* strs = bytes + strtab->sh_offset;
            const size_t n = symtab->sh_size / sizeof(ElfW(Sym));
            for (size_t i = 0; i < n; ++i) {
                if (!sym[i].st_name || ELF64_ST_TYPE(sym[i].st_info) != STT_FUNC || !sym[i].st_value) continue;
                if (sym[i].st_name >= strtab->sh_size) continue;
                if (strnlen(strs + sym[i].st_name, strtab->sh_size - sym[i].st_name) == strtab->sh_size - sym[i].st_name)
                    continue;
                if (strcmp(strs + sym[i].st_name, name) == 0) {
                    found = (char*)lm->l_addr + sym[i].st_value;
                    break;
                }
            }
        }
    }
    munmap(map, size);
    return found;
}

// Wine 11 (GE-Proton 11): win32u's own route from a D3DKMT shared NT handle to the host fd.
inline int d3dkmt_handle_to_fd(void* handle, std::string& why) {
    typedef unsigned (*open_fn)(unsigned, void*, unsigned*, unsigned*);
    typedef int (*get_fd_fn)(unsigned);
    typedef int (*destroy_fn)(unsigned);
    static std::once_flag once;
    static open_fn open_res;
    static get_fd_fn get_fd;
    static destroy_fn destroy_res;
    static bool have_win32u;
    std::call_once(once, [] {
        void* h = dlopen("win32u.so", RTLD_NOW | RTLD_NOLOAD);
        have_win32u = h != nullptr;
        if (!h) return;
        open_res = (open_fn)elf_local_sym(h, "d3dkmt_open_resource");
        get_fd = (get_fd_fn)elf_local_sym(h, "d3dkmt_object_get_fd");
        destroy_res = (destroy_fn)elf_local_sym(h, "d3dkmt_destroy_resource");
    });
    if (!have_win32u) {
        note(why, "win32u.so not loaded");
        return -1;
    }
    if (!open_res || !get_fd) {
        note(why, "win32u.so has no d3dkmt_open_resource/d3dkmt_object_get_fd in .symtab (stripped build?)");
        return -1;
    }
    unsigned mutex = 0, sync = 0;
    const unsigned local = open_res(0, handle, &mutex, &sync);
    if (!local) {
        note(why, "d3dkmt_open_resource failed");
        return -1;
    }
    const int fd = get_fd(local);
    if (destroy_res) destroy_res(local);
    if (fd < 0) note(why, "d3dkmt_object_get_fd failed");
    return fd;
}

inline void* ntdll_unix() {
    void* h = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD);
    return h ? h : dlopen("ntdll.so", RTLD_NOW);
}

inline int handle_to_fd(void* handle, std::string& why) {
    static std::once_flag once;
    static wine_h2fd_t h2fd;
    static nt_ioctl_fn ioctl;
    static nt_close_fn nt_close;
    std::call_once(once, [] {
        h2fd = (wine_h2fd_t)dlsym(RTLD_DEFAULT, "wine_server_handle_to_fd");
        void* nt = ntdll_unix();
        if (nt) {
            if (!h2fd) h2fd = (wine_h2fd_t)dlsym(nt, "wine_server_handle_to_fd");
            ioctl = (nt_ioctl_fn)dlsym(nt, "NtDeviceIoControlFile");
            nt_close = (nt_close_fn)dlsym(nt, "NtClose");
        }
    });
    if (!h2fd) {
        note(why, "wine_server_handle_to_fd not found (not a Wine process?)");
        return -1;
    }
    // 1. The dxgk ioctl (Proton 10 D3DKMT shared handles).
    if (ioctl && nt_close) {
        struct {
            long status;
            unsigned long information;
        } iosb{};
        int file32 = -1;
        const int st = ioctl(handle, nullptr, nullptr, nullptr, &iosb, 0x23400Cu, nullptr, 0, &file32, 4);
        if (st == 0 && file32 != -1 && file32 != 0) {
            void* file = (void*)(intptr_t)file32;
            int fd = -1;
            const int st2 = h2fd(file, 0x0001u, &fd, nullptr);
            nt_close(file);
            if (st2 == 0 && fd >= 0) return fd;
            note(why, "dxgk ioctl ok but handle_to_fd st=0x%x", (unsigned)st2);
        } else {
            note(why, "dxgk ioctl st=0x%x", (unsigned)st);
        }
    } else {
        note(why, "NtDeviceIoControlFile not found");
    }
    // 2. A handle that already is a file handle (older Wine), with the access winevulkan asks for first.
    static const unsigned accesses[] = {0x0001u, 0x0003u, 0x00100000u, 0x0u, 0xC0000000u, 0x80000000u};
    for (unsigned a : accesses) {
        int fd = -1;
        unsigned options = 0;
        const int st = h2fd(handle, a, &fd, &options);
        if (st == 0 && fd >= 0) return fd;
        if (a == accesses[0]) note(why, "handle_to_fd(FILE_READ_DATA) st=0x%x", (unsigned)st);
    }
    // 3. Wine 11 D3DKMT objects (GE-Proton 11): win32u's internal open + get_fd (hip-43).
    return d3dkmt_handle_to_fd(handle, why);
}

struct Import {
    void* handle;
    size_t size;
    uint32_t gen;
    hipExternalMemory_t ext;
    void* dev;
};

inline std::mutex& import_mutex() {
    static std::mutex m;
    return m;
}
inline std::vector<Import>& imports() {
    static std::vector<Import> v;
    return v;
}

// dev: the handle's memory as a HIP device pointer. False with `why` filled on any failure.
inline bool import_handle(void* handle, size_t size, uint32_t gen, void** dev, std::string& why) {
    if (!handle || !size || !dev) {
        note(why, "null handle/size");
        return false;
    }
    std::lock_guard<std::mutex> lock(import_mutex());
    for (const auto& e : imports())
        if (e.handle == handle && e.size == size && e.gen == gen) {
            *dev = e.dev;
            return true;
        }
    const int fd = handle_to_fd(handle, why);
    if (fd < 0) return false;
    const off_t fd_size = lseek(fd, 0, SEEK_END);
    if (fd_size > 0 && size_t(fd_size) < size) {
        note(why, "shared memory is %lld bytes, need %zu", (long long)fd_size, size);
        close(fd);
        return false;
    }
    hipExternalMemoryHandleDesc d{};
    d.type = hipExternalMemoryHandleTypeOpaqueFd;
    d.handle.fd = fd; // on success the HIP runtime owns the fd
    d.size = fd_size > 0 && size_t(fd_size) > size ? size_t(fd_size) : size;
    hipExternalMemory_t ext = nullptr;
    hipError_t err = hipImportExternalMemory(&ext, &d);
    if (err != hipSuccess) {
        note(why, "hipImportExternalMemory(%zu bytes): %s", size_t(d.size), hipGetErrorString(err));
        close(fd);
        return false;
    }
    hipExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = size;
    void* ptr = nullptr;
    err = hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd);
    if (err != hipSuccess || !ptr) {
        note(why, "hipExternalMemoryGetMappedBuffer: %s", hipGetErrorString(err));
        (void)hipDestroyExternalMemory(ext);
        return false;
    }
    imports().push_back({handle, size, gen, ext, ptr});
    *dev = ptr;
    return true;
}

} // namespace shared
} // namespace dlss5
