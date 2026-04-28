/*
 * comm_servlet.cpp
 *
 *      Author: xshthkr
 */

#include "comm_servlet.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#include <hwloc.h>

namespace async_rbruck_alltoallv {

/*
INTERNAL APIS
*/

void* servlet_malloc(size_t size, bool use_hugepages) {
    void *ptr = nullptr;
    // 2MB alignment required for Transparent Huge Pages (THP)
    if (posix_memalign(&ptr, 2 * 1024 * 1024, size) != 0) {
        ptr = malloc(size); // fallback
    }
    if (ptr && use_hugepages) {
        // Advise kernel to use hugepages for this mapping
        madvise(ptr, size, MADV_HUGEPAGE);
    }
    return ptr;
}

void servlet_free(void *ptr) {
    free(ptr);
}

static void pin_to_core(int core_id) {
    if (core_id < 0) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static double monotonic_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void execute_transfers(ServletSlot *slot, const ServletConfig *config) {

    CommDescriptor *desc = &slot->desc;

    int ngroup { desc->ngroup };
    int n { desc->n };
    int gid { desc->gid };
    int grank { desc->grank };
    int bblock { desc->bblock };
    MPI_Comm comm { desc->comm };

    if (bblock <= 0 || bblock > ngroup) bblock = ngroup;

    MPI_Request *reqs = (MPI_Request *) malloc(2 * bblock * sizeof(MPI_Request));
    MPI_Status *stats = (MPI_Status *) malloc(2 * bblock * sizeof(MPI_Status));

    double post_start { MPI_Wtime() };

    for (int ii { 0 }; ii < ngroup; ii += bblock) {
        int req_cnt { 0 };
        int ss { (ngroup - ii < bblock) ? (ngroup - ii) : bblock };

        /* post receives */
        for (int i { 0 }; i < ss; i++) {
            int nsrc { (gid + i + ii) % ngroup };
            int src { nsrc * n + grank };

            MPI_Irecv(&desc->recv_buf[desc->recv_displs[nsrc]], desc->recv_sizes[nsrc], MPI_CHAR, src, 0, comm, &reqs[req_cnt++]);
        }

        /* post sends */
        for (int i { 0 }; i < ss; i++) {
            int ndst { (gid - i - ii + ngroup) % ngroup };
            int dst { ndst * n + grank };

            MPI_Isend(&desc->send_buf[desc->send_displs[ndst]], desc->send_sizes[ndst], MPI_CHAR, dst, 0, comm, &reqs[req_cnt++]);
        }

        double post_end { MPI_Wtime() };
        slot->post_time += (post_end - post_start);

        /* drive progress with MPI_Testsome */
        int completed_total { 0 };
        double deadline { monotonic_seconds() + config->deadlock_timeout_s };

        while (completed_total < req_cnt) {
            int outcount { 0 };
            int indices[2 * bblock];
            MPI_Status tst[2 * bblock];

            int ret { MPI_Testsome(req_cnt, reqs, &outcount, indices, tst) };

            if (ret != MPI_SUCCESS) {
                /* fallback: force completion */
                MPI_Waitall(req_cnt, reqs, stats);
                completed_total = req_cnt;
                break;
            }

            if (outcount == MPI_UNDEFINED) {
                /* all requests already completed */
                break;
            }

            if (outcount > 0) {
                completed_total += outcount;
                deadline = monotonic_seconds() + config->deadlock_timeout_s;
            } else {
                /* no progress, check deadlock timeout */
                if (monotonic_seconds() > deadline) {
                    fprintf(stderr, "[comm_servlet] WARN: no progress for %ds, falling back to MPI_Waitall\n", config->deadlock_timeout_s);
                    MPI_Waitall(req_cnt, reqs, stats);
                    completed_total = req_cnt;
                    break;
                }
            }
        }

        post_start = MPI_Wtime();
    }

    // todo: only captures time after last batch finishes
    slot->progress_time = MPI_Wtime() - post_start;

    free(reqs);
    free(stats);
}

static void *servlet_thread_fn(void *arg) {

    ServletContext *ctx = (ServletContext *)arg;

    pin_to_core(ctx->config.servlet_core_id);

    int backoff_us { 0 };

    while (!ctx->shutdown.load(std::memory_order_acquire)) {

        bool did_work { false };

        for (int s { 0 }; s < NUM_SLOTS; s++) {
            int state { ctx->slots[s].state.load(std::memory_order_acquire) };

            if (state == static_cast<int>(ServletState::READY)) {
                did_work = true;
                backoff_us = 0;

                double t0 { MPI_Wtime() };
                ctx->slots[s].post_time = 0;
                ctx->slots[s].progress_time = 0;

                execute_transfers(&ctx->slots[s], &ctx->config);

                ctx->slots[s].total_time = MPI_Wtime() - t0;

                /* signal completion */
                ctx->slots[s].state.store(static_cast<int>(ServletState::DONE), std::memory_order_release);
            }
        }

        if (!did_work) {
            /* adaptive backoff when idle */
            if (backoff_us < ctx->config.backoff_max_us) {
                backoff_us++;
            }
            if (backoff_us > 0) {
                usleep(backoff_us);
            }
        }
    }

    return nullptr;
}

/*
PUBLIC APIS
*/

// Forward declarations for utils used in servlet_init
static int detect_nic_affine_core() {
    hwloc_topology_t topology;
    if (hwloc_topology_init(&topology) < 0) return -1;
    
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
    if (hwloc_topology_load(topology) < 0) {
        hwloc_topology_destroy(topology);
        return -1;
    }

    int target_pu { -1 };
    hwloc_obj_t osdev = nullptr;
    
    // Find first OpenFabrics (InfiniBand) device
    while ((osdev = hwloc_get_next_osdev(topology, osdev)) != nullptr) {
        if (osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
            hwloc_obj_t non_io = hwloc_get_non_io_ancestor_obj(topology, osdev);
            if (non_io && non_io->cpuset) {
                // Find first PU in the CPUSet closest to the NIC
                hwloc_obj_t pu = hwloc_get_obj_inside_cpuset_by_type(topology, non_io->cpuset, HWLOC_OBJ_PU, 0);
                if (pu) {
                    target_pu = pu->os_index;
                    break;
                }
            }
        }
    }

    hwloc_topology_destroy(topology);
    return target_pu;
}

int servlet_init(ServletContext *ctx, const ServletConfig *config) {

    ctx->config = *config;

    // Auto-detect NIC affinity via hwloc if requested
    if (ctx->config.servlet_core_id == -2) {
        int core { detect_nic_affine_core() };
        if (core >= 0) {
            ctx->config.servlet_core_id = core;
        } else {
            // Fallback to no pinning if hwloc fails or no IB device found
            ctx->config.servlet_core_id = -1;
        }
    }

    ctx->shutdown.store(false, std::memory_order_relaxed);
    ctx->producer_idx = 0;

    for (int i { 0 }; i < NUM_SLOTS; i++) {
        ctx->slots[i].state.store(static_cast<int>(ServletState::IDLE), std::memory_order_relaxed);
        ctx->slots[i].send_buffer = nullptr;
        ctx->slots[i].send_buffer_capacity = 0;
        ctx->slots[i].sizes_storage = nullptr;
        ctx->slots[i].sizes_ngroup = 0;
        ctx->slots[i].extra_buffer = nullptr;
        ctx->slots[i].extra_buffer_capacity = 0;
        ctx->slots[i].temp_recv_buffer = nullptr;
        ctx->slots[i].temp_recv_buffer_capacity = 0;
        ctx->slots[i].chunk_recv_buffer = nullptr;
        ctx->slots[i].chunk_recv_buffer_capacity = 0;
        ctx->slots[i].post_time = 0;
        ctx->slots[i].progress_time = 0;
        ctx->slots[i].total_time = 0;
        memset(&ctx->slots[i].desc, 0, sizeof(CommDescriptor));
    }

    int rc = pthread_create(&ctx->thread, nullptr, servlet_thread_fn, ctx);
    if (rc != 0) {
        fprintf(stderr, "[comm_servlet] ERROR: pthread_create failed (%d)\n", rc);
        return -1;
    }

    ctx->thread_active = true;
    return 0;
}

int servlet_shutdown(ServletContext *ctx) {
    if (!ctx->thread_active) return 0;

    /* wait for any in-flight work first */
    servlet_wait(ctx);

    ctx->shutdown.store(true, std::memory_order_release);
    pthread_join(ctx->thread, nullptr);
    ctx->thread_active = false;

    /* free slot-owned buffers */
    for (int i { 0 }; i < NUM_SLOTS; i++) {
        if (ctx->slots[i].send_buffer) {
            servlet_free(ctx->slots[i].send_buffer);
            ctx->slots[i].send_buffer = nullptr;
        }
        if (ctx->slots[i].sizes_storage) {
            free(ctx->slots[i].sizes_storage); // sizes_storage doesn't use THP
            ctx->slots[i].sizes_storage = nullptr;
        }
        if (ctx->slots[i].extra_buffer) {
            servlet_free(ctx->slots[i].extra_buffer);
            ctx->slots[i].extra_buffer = nullptr;
        }
        if (ctx->slots[i].temp_recv_buffer) {
            servlet_free(ctx->slots[i].temp_recv_buffer);
            ctx->slots[i].temp_recv_buffer = nullptr;
        }
        if (ctx->slots[i].chunk_recv_buffer) {
            servlet_free(ctx->slots[i].chunk_recv_buffer);
            ctx->slots[i].chunk_recv_buffer = nullptr;
        }
    }

    return 0;
}

void servlet_submit(ServletContext *ctx) {
    int slot { ctx->producer_idx };
    ctx->slots[slot].state.store(static_cast<int>(ServletState::READY), std::memory_order_release);
    ctx->producer_idx = 1 - slot;
}

void servlet_wait(ServletContext *ctx) {
    for (int i { 0 }; i < NUM_SLOTS; i++) {
        while (true) {
            int s { ctx->slots[i].state.load(std::memory_order_acquire) };
            if (s == static_cast<int>(ServletState::IDLE)) break;
            if (s == static_cast<int>(ServletState::DONE)) {
                ctx->slots[i].state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
                break;
            }
            /* READY: still in-flight, spin */
        }
    }
}

bool servlet_test(ServletContext *ctx) {
    for (int i { 0 }; i < NUM_SLOTS; i++) {
        int s { ctx->slots[i].state.load(std::memory_order_acquire) };
        if (s == static_cast<int>(ServletState::READY)) return false;
        if (s == static_cast<int>(ServletState::DONE)) {
            ctx->slots[i].state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
        }
    }
    return true;
}

} /* namespace async_rbruck_alltoallv */
