/*
 * comm_servlet.h
 *
 * dedicated communication servlet for ParLinNa
 *
 * each mpi process (rank) spawns one comm pthread that handles
 * inter-node MPI_Isend/MPI_Irecv on a dedicated NIC-affine core 
 * while the main thread performs computation 
 *
 * double-buffered slot design enables pipelining:
 * phase 2 of iteration K runs on slot A while
 * phase 1 of iteration K+1 prepares slot B
 *
 * servlet thread dequeues work from either slot, posts mpi transfers,
 * drives MPI_Testsome progress loop, signals completion
 *
 *      Author: xshthkr
 */

#ifndef COMM_SERVLET_H_
#define COMM_SERVLET_H_

#include <mpi.h>
#include <pthread.h>
#include <atomic>
#include <cstddef>

namespace async_rbruck_alltoallv {

/* number of double-buffer slots for pipelining */
static const int NUM_SLOTS = 2;

/*
servlet work states (main thread <-> comm thread handoff)

protocol per slot:
1. main thread fills slot desc, stores READY
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
    char *send_buf;
    int  *send_sizes;       // [ngroup] byte count per destination node 
    int  *send_displs;      // [ngroup] byte offset per destination node

    // recv side 
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
    int servlet_core_id;        // physical core to pin to (-1 = no pinning, -2 = auto hwloc NIC-affine)
    int backoff_max_us;         // max idle backoff in microseconds
    int deadlock_timeout_s;     // seconds before fallback MPI_Waitall
    bool use_hugepages;         // attempt to allocate buffers using MAP_HUGETLB
};

/*
returns ServletConfig with sensible defaults
*/
inline ServletConfig servlet_default_config() {
    ServletConfig c;
    c.servlet_core_id    = -2; // auto detect NIC affinity via hwloc by default
    c.backoff_max_us     = 100;
    c.deadlock_timeout_s = 10;
    c.use_hugepages      = true;
    return c;
}

/*
one double-buffer slot

owns the send buffer and sizes/displs arrays on the heap
so they persist while the servlet reads them asynchronously

also owns phase 1 workspace buffers (extra_buffer, temp_recv_buffer)
and phase 2 chunk receive buffer (chunk_recv_buffer)
so they are allocated once and reused across iterations

sizes_storage layout: [send_sizes | send_displs | recv_sizes | recv_displs]
                       4 * ngroup ints total
*/
struct alignas(64) ServletSlot {
    std::atomic<int> state{static_cast<int>(ServletState::IDLE)};

    CommDescriptor desc;

    // owned send buffer for phase 2 (heap, persists across calls)
    char  *send_buffer{nullptr};
    size_t send_buffer_capacity{0};

    // owned sizes/displs storage (heap, 4 * ngroup ints)
    int   *sizes_storage{nullptr};
    int    sizes_ngroup{0};

    // phase 1 workspace buffers (heap, reused across calls)
    char  *extra_buffer{nullptr};
    size_t extra_buffer_capacity{0};

    char  *temp_recv_buffer{nullptr};
    size_t temp_recv_buffer_capacity{0};

    // phase 2 contiguous recv buffer for chunking
    char  *chunk_recv_buffer{nullptr};
    size_t chunk_recv_buffer_capacity{0};

    // per-slot timing
    double post_time{0};
    double progress_time{0};
    double total_time{0};
};

/*
per-process servlet context
owns the comm thread and the double-buffered work slots
*/
struct alignas(64) ServletContext {
    // shutdown signal
    std::atomic<bool> shutdown{false};

    // double-buffered work slots
    ServletSlot slots[NUM_SLOTS];

    // which slot the producer (main thread) writes to next
    int producer_idx{0};

    // config
    ServletConfig config;

    // thread handle
    pthread_t thread;
    bool      thread_active{false};
};


/*
spawn the servlet thread 
call once per mpi process after MPI_Init_thread(MPI_THREAD_MULTIPLE)

returns 0 on success -1 on error
*/
int servlet_init(ServletContext *ctx, const ServletConfig *config);

/*
signal the servlet to exit, join the thread, free slot buffers
call once per process, before MPI_Finalize()
*/
int servlet_shutdown(ServletContext *ctx);

/*
submit the current producer slot for execution
sets state to READY, toggles producer_idx
returns immediately, servlet begins work asynchronously
*/
void servlet_submit(ServletContext *ctx);

/*
block until ALL in-flight slots complete
resets completed slots to IDLE
*/
void servlet_wait(ServletContext *ctx);

/*
non-blocking completion test
returns true if ALL slots are idle or done (and resets DONE to IDLE)
*/
bool servlet_test(ServletContext *ctx);

} /* namespace async_rbruck_alltoallv */

#endif /* COMM_SERVLET_H_ */
