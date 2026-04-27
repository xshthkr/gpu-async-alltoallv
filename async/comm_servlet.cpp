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

namespace async_rbruck_alltoallv {

/*
INTERNAL APIS
*/

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

static void execute_transfers(ServletContext *ctx) {

    CommDescriptor *desc = &ctx->desc;

    int ngroup { desc->ngroup };
    int n { desc->n };
    int gid { desc->gid };
    int grank { desc->grank };
    int bblock { desc->bblock };
    MPI_Comm comm {desc->comm };

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
        ctx->post_time += (post_end - post_start);

        /* drive progress with MPI_Testsome */
        int completed_total { 0 };
        double deadline { monotonic_seconds() + ctx->config.deadlock_timeout_s };

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
                deadline = monotonic_seconds() + ctx->config.deadlock_timeout_s;
            } else {
                /* no progress, check deadlock timeout */
                if (monotonic_seconds() > deadline) {
                    fprintf(stderr, "[comm_servlet] WARN: no progress for %ds, falling back to MPI_Waitall\n", ctx->config.deadlock_timeout_s);
                    MPI_Waitall(req_cnt, reqs, stats);
                    completed_total = req_cnt;
                    break;
                }
            }
        }

        post_start = MPI_Wtime();
    }

    ctx->progress_time = MPI_Wtime() - post_start;

    free(reqs);
    free(stats);
}

static void *servlet_thread_fn(void *arg) {

    ServletContext *ctx = (ServletContext *)arg;

    pin_to_core(ctx->config.servlet_core_id);

    int backoff_us { 0 };

    while (!ctx->shutdown.load(std::memory_order_acquire)) {

        int state { ctx->state.load(std::memory_order_acquire) };

        if (state == static_cast<int>(ServletState::READY)) {
            backoff_us = 0;

            double t0 { MPI_Wtime() };
            ctx->post_time = 0;
            ctx->progress_time = 0;

            execute_transfers(ctx);

            ctx->total_time = MPI_Wtime() - t0;

            /* signal completion */
            ctx->state.store(static_cast<int>(ServletState::DONE), std::memory_order_release);
        } else {
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

int servlet_init(ServletContext *ctx, const ServletConfig *config) {

    ctx->config = *config;
    ctx->shutdown.store(false, std::memory_order_relaxed);
    ctx->state.store(static_cast<int>(ServletState::IDLE), std::memory_order_relaxed);
    ctx->post_time = 0;
    ctx->progress_time = 0;
    ctx->total_time = 0;

    memset(&ctx->desc, 0, sizeof(CommDescriptor));

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

    ctx->shutdown.store(true, std::memory_order_release);
    pthread_join(ctx->thread, nullptr);
    ctx->thread_active = false;
    return 0;
}

void servlet_submit(ServletContext *ctx) {
    ctx->state.store(static_cast<int>(ServletState::READY), std::memory_order_release);
}

void servlet_wait(ServletContext *ctx) {
    while (ctx->state.load(std::memory_order_acquire) != static_cast<int>(ServletState::DONE)) {
        // next iteration of phase 1 to pipeline across iterations?
        // useful computation?
        // spin for now
    }
    ctx->state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
}

bool servlet_test(ServletContext *ctx) {
    if (ctx->state.load(std::memory_order_acquire) == static_cast<int>(ServletState::DONE)) {
        ctx->state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
        return true;
    }
    return false;
}

} /* namespace async_rbruck_alltoallv */
