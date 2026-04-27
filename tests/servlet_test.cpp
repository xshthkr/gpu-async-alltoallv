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

    // fill send buffer with identifiable pattern: rank * 1000 + dest
    long long *sendbuf = new long long[soffset];
    int idx { 0 };
    for (int i { 0 }; i < nprocs; i++) {
        for (int j { 0 }; j < sendcounts[i]; j++) {
            sendbuf[idx++] = rank * 1000 + i;
        }
    }

    /* RUN MPI_Alltoallv */
    long long *recv_mpi = new long long[roffset];
    memset(recv_mpi, 0, roffset * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 { MPI_Wtime() };
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
                  recv_mpi, recvcounts, rdispls, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    double t_mpi { MPI_Wtime() - t0 };

    /* RUN ParLinNa_coalesced */
    long long *recv_ref = new long long[roffset];
    memset(recv_ref, 0, roffset * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    async_rbruck_alltoallv::ParLinNa_coalesced(
        n, r, bblock,
        (char*)sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
        (char*)recv_ref, recvcounts, rdispls, MPI_LONG_LONG,
        MPI_COMM_WORLD);

    double t_coalesced { MPI_Wtime() - t0 };

    /* RUN ParLinNa_servlet */

    long long *recv_srv = new long long[roffset];
    memset(recv_srv, 0, roffset * sizeof(long long));

    // init servlet
    async_rbruck_alltoallv::ServletConfig cfg = async_rbruck_alltoallv::servlet_default_config();
    async_rbruck_alltoallv::ServletContext servlet_ctx;
    async_rbruck_alltoallv::servlet_init(&servlet_ctx, &cfg);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    async_rbruck_alltoallv::ParLinNa_servlet(
        n, r, bblock,
        (char*)sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
        (char*)recv_srv, recvcounts, rdispls, MPI_LONG_LONG,
        MPI_COMM_WORLD, &servlet_ctx);

    double t_servlet { MPI_Wtime() - t0 };

    async_rbruck_alltoallv::servlet_shutdown(&servlet_ctx);

    // compare results
    int errors_srv { 0 };
    int errors_coal { 0 };
    for (int i { 0 }; i < roffset; i++) {
        if (recv_mpi[i] != recv_srv[i]) {
            errors_srv++;
            if (errors_srv <= 5) {
                std::cerr << "[rank " << rank << "] servlet mismatch at index " << i << ": mpi=" << recv_mpi[i] << " srv=" << recv_srv[i] << std::endl;
            }
        }
        if (recv_mpi[i] != recv_ref[i]) {
            errors_coal++;
            if (errors_coal <= 5) {
                std::cerr << "[rank " << rank << "] coalesced mismatch at index " << i << ": mpi=" << recv_mpi[i] << " coal=" << recv_ref[i] << std::endl;
            }
        }
    }

    // gather results
    int total_errors_srv { 0 };
    int total_errors_coal { 0 };
    MPI_Reduce(&errors_srv, &total_errors_srv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&errors_coal, &total_errors_coal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_t_mpi { 0 };
    double max_t_coal { 0 };
    double max_t_srv { 0 };
    MPI_Reduce(&t_mpi, &max_t_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_coalesced, &max_t_coal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_servlet, &max_t_srv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sasha's beloved ParLinNa Servlet Test" << std::endl;
        std::cout << "procs=" << nprocs << " n=" << n << " r=" << r << " bblock=" << bblock << " msg_size=" << msg_size << std::endl;
        std::cout << "default mpi:  " << max_t_mpi << "s" << std::endl;
        std::cout << "coalesced:    " << max_t_coal << "s" << std::endl;
        std::cout << "servlet:      " << max_t_srv << "s" << std::endl;

        if (total_errors_srv == 0 && total_errors_coal == 0) {
            std::cout << "PASS: all results match MPI_Alltoallv (yipee)" << std::endl;
        } else {
            if (total_errors_srv > 0) std::cout << "FAIL: servlet has " << total_errors_srv << " mismatches" << std::endl;
            if (total_errors_coal > 0) std::cout << "FAIL: coalesced has " << total_errors_coal << " mismatches" << std::endl;
            std::cout << "(womp womp)" << std::endl;
        }
    }

    delete[] sendbuf;
    delete[] recv_mpi;
    delete[] recv_ref;
    delete[] recv_srv;

    MPI_Finalize();
    return (total_errors_srv > 0 || total_errors_coal > 0) ? 1 : 0;
}
