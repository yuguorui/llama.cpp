#include <time.h>
#include <sys/types.h>

#ifdef __SGX_ENCLAVE__

#ifndef __SGX_USER_TYPES_H__
#define __SGX_USER_TYPES_H__

# define CLOCK_MONOTONIC 1
#define CLOCKS_PER_SEC 1000000

typedef int clockid_t;

struct timespec {
    time_t tv_sec;      // nombre de secondes
    long tv_nsec;       // nombre de nanosecondes
};

#define stdout (void *)1
#define stderr (void *)2

#define O_RDONLY        00000000

#ifndef SEEK_SET
#define	SEEK_SET	0	/* set file offset to offset */
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#define	SEEK_END	2	/* set file offset to EOF plus offset */
#endif	/* !SEEK_SET */

#ifdef __cplusplus
extern "C" {
#endif

int clock_gettime(clockid_t clockid, struct timespec *tp);
clock_t clock(void);
int printf(const char* fmt, ...);
int fprintf(void *f, const char* fmt, ...);
int open(const char *pathname, int flags);
int close(int fd);
off_t lseek(int fd, off_t offset, int whence);
ssize_t read(int fd, void *buf, size_t count);
time_t time(time_t *tloc);
unsigned int sleep(unsigned int seconds);

#ifdef __cplusplus
}
#endif

#endif // __SGX_USER_TYPES_H__
#endif // __SGX_ENCLAVE__

