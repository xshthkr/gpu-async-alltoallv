/*
 * comm_servlet.h
 *
 * dedicated communication servlet for ParLinNa
 *
 * each mpi process (rank) spawns one comm pthread that handles
 * inter-node MPI_Isend/MPI_Irecv on a dedicated NIC-affine core 
 * while the main thread performs computation 
 *
 * all servlet threads on a node share one physical core
 *
 * main thread runs phase 1 intra-node bruck, enqueues phase 2 
 * work, then does useful computation (overlap)
 * 
 * servlet thread dequeues work, posts mpi transfers, drives
 * MPI_Testsome progress loop, signals completion
 *
 *      Author: xshthkr
 */

#ifndef COMM_SERVLET_H_
#define COMM_SERVLET_H_

#include <mpi.h>
#include <pthread.h>
#include <atomic>

namespace async_rbruck_alltoallv {

/*
servlet work states (main thread <-> comm thread handoff)

protocol:
1. main thread fills desc, stores READY
2. servlet sees READY, executes transfers, stores DONE
3. main thread sees DONE, consumes results, stores IDLE
*/
enum class ServletState : int {
    IDLE    = 0,
    READY   = 1,
    DONE    = 2
};

/*
describes one inter-node scatter operation.

filled by the main thread after phase 1 completes
the servlet reads these fields and posts MPI_Isend/MPI_Irecv accordingly

sizes and displacements in BYTES
*/
struct CommDescriptor {
    // send side 
    // packed by phase 1 into temp_send_buffer
    char *send_buf;
    int  *send_sizes;       // [ngroup] byte count per destination node 
    int  *send_displs;      // [ngroup] byte offset per destination node

    // recv side 
    // servlet writes directly into recvbuf
    char *recv_buf;
    int  *recv_sizes;       // [ngroup] byte count per source node  
    int  *recv_displs;      // [ngroup] byte offset per source node 

    // routing 
    int   ngroup;           // number of node-groups 
    int   n;                // ranks per node 
    int   gid;              // this rank's group id
    int   grank;            // this rank's intra-node rank
    int   bblock;           // batching block size

    MPI_Comm comm;
};

/*
config knobs passed to servlet_init()
*/
struct ServletConfig {
    int servlet_core_id;        // physical core to pin to (-1 = no pinning)
    int backoff_max_us;         // max idle backoff in microseconds
    int deadlock_timeout_s;     // seconds before fallback MPI_Waitall
};

/*
returns ServletConfig with sensible defaults
*/
inline ServletConfig servlet_default_config() {
    ServletConfig c;
    c.servlet_core_id    = -1;
    c.backoff_max_us     = 100;
    c.deadlock_timeout_s = 10;
    return c;
}

/*
per-process servlet context
owns the comm thread and the work descriptor

cacheline-aligned to prevent false sharing between the atomic state flag
(written by both threads) and the descriptor fields (written only by main)
*/
struct alignas(64) ServletContext {
    // shutdown signal
    std::atomic<bool> shutdown{false};

    // work handoff flag — own cacheline
    alignas(64) std::atomic<int> state{static_cast<int>(ServletState::IDLE)};

    // the work descriptor
    // written by main before READY, read by servlet
    CommDescriptor desc;

    // config
    ServletConfig config;

    // thread handle
    pthread_t thread;
    bool      thread_active{false};

    // timing
    // written by servlet, read by main after DONE
    double post_time;           // time to post all MPI requests
    double progress_time;       // time in Testsome loop
    double total_time;          // READY -> DONE wall time
};


/*
spawn the servlet thread 
call once per mpi process after MPI_Init_thread(MPI_THREAD_MULTIPLE)

returns 0 on success -1 on error
*/
int servlet_init(ServletContext *ctx, const ServletConfig *config);

/*
signal the servlet to exit and join the thread
call once per process, before MPI_Finalize()
*/
int servlet_shutdown(ServletContext *ctx);


/*
submit the pre-filled descriptor for execution
caller must have filled ctx -> desc before calling this
returns immediately, servlet begins work asynchronously
*/
void servlet_submit(ServletContext *ctx);

/*
block until the servlet signals completion (state == DONE)
after return, received data is valid in desc.recv_buf
resets state to IDLE
*/
void servlet_wait(ServletContext *ctx);

/*
non-blocking completion test
returns true if servlet has finished (and resets state to IDLE)
*/
bool servlet_test(ServletContext *ctx);

} /* namespace async_rbruck_alltoallv */

#endif /* COMM_SERVLET_H_ */
