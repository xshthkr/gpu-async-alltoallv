/*
 * servlet_test.cpp
 *
 * (terrible?) minimal test: runs ParLinNa_coalesced and ParLinNa_servlet
 * on the same data and compares results for correctness
 *
 * usage: mpirun -n <nprocs> ./servlet_test <n> <r> <bblock> <msg_size>
 *  n        = ranks per node (group size)
 *  r        = bruck radix
 *  bblock   = batching block size
 *  msg_size = elements per destination
 *
 *      Author: xshthkr
 */

#include "../async/async.h"
#include "../async/comm_servlet.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

int main(int argc, char **argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: MPI_THREAD_MULTIPLE not supported" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 5) {
        if (rank == 0)
            std::cout << "Usage: mpirun -n <nprocs> " << argv[0]
                      << " <n> <r> <bblock> <msg_size>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int n { atoi(argv[1]) };
    int r { atoi(argv[2]) };
    int bblock { atoi(argv[3]) };
    int msg_size { atoi(argv[4]) };

    if (nprocs % n != 0) {
        if (rank == 0)
            std::cerr << "ERROR: nprocs (" << nprocs << ") must be divisible by n (" << n << ")" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // build uniform sendcounts (msg_size elements to each rank)
    int sendcounts[nprocs], recvcounts[nprocs];
    int sdispls[nprocs], rdispls[nprocs];

    for (int i { 0 }; i < nprocs; i++) { sendcounts[i] = msg_size; } 

    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int soffset { 0 };
    int roffset { 0 };
    for (int i { 0 }; i < nprocs; i++) {
        sdispls[i] = soffset;
        rdispls[i] = roffset;
        soffset += sendcounts[i];
        roffset += recvcounts[i];
    }

    // allocate multiple sendbufs and recvbufs for multiple iterations
    int num_iters { 10 };
    long long **sendbufs = new long long*[num_iters];
    long long **recvbufs_seq = new long long*[num_iters];
    long long **recvbufs_pipe = new long long*[num_iters];

    for (int i { 0 }; i<num_iters; i++) {
        sendbufs[i] = new long long[soffset];
        recvbufs_seq[i] = new long long[roffset];
        recvbufs_pipe[i] = new long long[roffset];
        
        memset(recvbufs_seq[i], 0, roffset * sizeof(long long));
        memset(recvbufs_pipe[i], 0, roffset * sizeof(long long));
        
        int idx { 0 };
        for (int p { 0 }; p < nprocs; p++) {
            for (int j { 0 }; j < sendcounts[p]; j++) {
                sendbufs[i][idx++] = rank * 1000 + p;
            }
        }
    }

    // init servlet
    async_rbruck_alltoallv::ServletConfig cfg = async_rbruck_alltoallv::servlet_default_config();
    async_rbruck_alltoallv::ServletContext servlet_ctx;
    async_rbruck_alltoallv::servlet_init(&servlet_ctx, &cfg);

    int typesize;
    MPI_Type_size(MPI_LONG_LONG, &typesize);
    async_rbruck_alltoallv::ParLinNa_Handle* handle = async_rbruck_alltoallv::ParLinNa_Init_handle(MPI_COMM_WORLD, n, typesize);

    /* SEQUENTIAL LOOP */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_seq_start { MPI_Wtime() };
    for (int i { 0 }; i < num_iters; i++) {
        async_rbruck_alltoallv::ParLinNa_Phase1(handle, r, (char*)sendbufs[i], sendcounts, sdispls, MPI_LONG_LONG, recvcounts, MPI_LONG_LONG, MPI_COMM_WORLD);
        async_rbruck_alltoallv::ParLinNa_Phase2_submit(handle, bblock, (char*)recvbufs_seq[i], rdispls, MPI_LONG_LONG, MPI_COMM_WORLD, &servlet_ctx);
        async_rbruck_alltoallv::servlet_wait(&servlet_ctx);
    }
    double t_seq { MPI_Wtime() - t_seq_start };

    /* PIPELINED LOOP (Real Overlap) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_pipe_start { MPI_Wtime() };
    
    // Iteration 0 Phase 1
    async_rbruck_alltoallv::ParLinNa_Phase1(handle, r, (char*)sendbufs[0], sendcounts, sdispls, MPI_LONG_LONG, recvcounts, MPI_LONG_LONG, MPI_COMM_WORLD);
    
    for (int i { 0 }; i < num_iters; i++) {
        // Submit current iteration
        async_rbruck_alltoallv::ParLinNa_Phase2_submit(handle, bblock, (char*)recvbufs_pipe[i], rdispls, MPI_LONG_LONG, MPI_COMM_WORLD, &servlet_ctx);
        
        // OVERLAP WINDOW: compute next iteration's Phase 1 concurrently!
        if (i + 1 < num_iters) {
            async_rbruck_alltoallv::ParLinNa_Phase1(handle, r, (char*)sendbufs[i+1], sendcounts, sdispls, MPI_LONG_LONG, recvcounts, MPI_LONG_LONG, MPI_COMM_WORLD);
        }
        
        // Wait for current iteration's Phase 2
        async_rbruck_alltoallv::servlet_wait(&servlet_ctx);
    }
    double t_pipe { MPI_Wtime() - t_pipe_start };

    async_rbruck_alltoallv::servlet_shutdown(&servlet_ctx);
    async_rbruck_alltoallv::ParLinNa_Free_handle(handle);

    // check results
    int errors { 0 };
    for (int i { 0 }; i < num_iters; i++) {
        for (int j { 0 }; j < roffset; j++) {
            if (recvbufs_seq[i][j] != recvbufs_pipe[i][j]) {
                errors++;
            }
        }
    }

    int total_errors { 0 };
    MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_t_seq { 0 };
    int max_t_pipe { 0 };
    MPI_Reduce(&t_seq, &max_t_seq, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_pipe, &max_t_pipe, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sasha's beloved ParLinNa Overlap Test (" << num_iters << " iters)" << std::endl;
        std::cout << "procs=" << nprocs << " n=" << n << " r=" << r << " bblock=" << bblock << " msg_size=" << msg_size << std::endl;
        std::cout << "sequential loop: " << max_t_seq << "s" << std::endl;
        std::cout << "pipelined loop:  " << max_t_pipe << "s" << std::endl;
        
        if (max_t_pipe < max_t_seq) {
            double speedup { (max_t_seq - max_t_pipe) / max_t_seq * 100.0 };
            std::cout << "overlap speedup: " << speedup << "%" << std::endl;
        }

        if (total_errors == 0) {
            std::cout << "PASS: pipelined results match sequential exactly (yipee)" << std::endl;
        } else {
            std::cout << "FAIL: " << total_errors << " mismatches (womp womp)" << std::endl;
        }
    }

    for (int i { 0 }; i < num_iters; i++) {
        delete[] sendbufs[i];
        delete[] recvbufs_seq[i];
        delete[] recvbufs_pipe[i];
    }
    delete[] sendbufs;
    delete[] recvbufs_seq;
    delete[] recvbufs_pipe;

    MPI_Finalize();
    return (total_errors > 0) ? 1 : 0;
}
