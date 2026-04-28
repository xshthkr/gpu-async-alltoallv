/*
 * servlet_test_v2.cpp
 *
 * tests ParLinNa_servlet_v2 (chunked single-call pipeline)
 *
 * 1. MPI_Alltoallv          (baseline)
 * 2. ParLinNa_coalesced     (reference)
 * 3. ParLinNa_servlet       (v1, no chunking)
 * 4. ParLinNa_servlet_v2    (chunked pipeline, varying chunk counts)
 *
 * correctness: all results compared against MPI_Alltoallv
 *
 * usage: mpirun -n <nprocs> ./servlet_test_v2 <n> <r> <bblock> <msg_size>
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

    // build uniform sendcounts
    int sendcounts[nprocs], recvcounts[nprocs];
    int sdispls[nprocs], rdispls[nprocs];

    for (int i { 0 }; i < nprocs; i++) { sendcounts[i] = msg_size; }

    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int soffset { 0 }, roffset { 0 };
    for (int i { 0 }; i < nprocs; i++) {
        sdispls[i] = soffset;
        rdispls[i] = roffset;
        soffset += sendcounts[i];
        roffset += recvcounts[i];
    }

    // fill send buffer: rank * 1000 + dest
    long long *sendbuf { new long long[soffset] };
    int idx { 0 };
    for (int i { 0 }; i < nprocs; i++) {
        for (int j { 0 }; j < sendcounts[i]; j++) {
            sendbuf[idx++] = rank * 1000 + i;
        }
    }

    /* 1. MPI_Alltoallv baseline */
    long long *recv_mpi { new long long[roffset] };
    memset(recv_mpi, 0, roffset * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 { MPI_Wtime() };
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
                  recv_mpi, recvcounts, rdispls, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    double t_mpi { MPI_Wtime() - t0 };

    /* 2. ParLinNa_coalesced */
    long long *recv_coal { new long long[roffset] };
    memset(recv_coal, 0, roffset * sizeof(long long));

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    async_rbruck_alltoallv::ParLinNa_coalesced(
        n, r, bblock,
        (char*)sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
        (char*)recv_coal, recvcounts, rdispls, MPI_LONG_LONG,
        MPI_COMM_WORLD);
    double t_coal { MPI_Wtime() - t0 };

    /* 3. ParLinNa_servlet v1 (single-shot, no chunking) */
    long long *recv_v1 { new long long[roffset] };
    memset(recv_v1, 0, roffset * sizeof(long long));

    async_rbruck_alltoallv::ServletConfig cfg { async_rbruck_alltoallv::servlet_default_config() };
    async_rbruck_alltoallv::ServletContext servlet_ctx;
    async_rbruck_alltoallv::servlet_init(&servlet_ctx, &cfg);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    async_rbruck_alltoallv::ParLinNa_servlet(
        n, r, bblock,
        (char*)sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
        (char*)recv_v1, recvcounts, rdispls, MPI_LONG_LONG,
        MPI_COMM_WORLD, &servlet_ctx);
    async_rbruck_alltoallv::servlet_wait(&servlet_ctx);
    double t_v1 { MPI_Wtime() - t0 };

    /* 4. ParLinNa_servlet_v2 with varying chunk counts */
    int chunk_counts[] = { 2, 4, 8 };
    int num_configs { 3 };

    // skip chunk counts that exceed msg_size (can't split 1 element into 4 chunks)
    // find how many are valid
    int valid_configs { 0 };
    for (int c { 0 }; c < num_configs; c++) {
        if (chunk_counts[c] <= msg_size) valid_configs++;
    }

    long long **recv_v2 { new long long*[valid_configs] };
    double *t_v2 { new double[valid_configs] };

    int vi { 0 };
    for (int c { 0 }; c < num_configs; c++) {
        if (chunk_counts[c] > msg_size) continue;

        recv_v2[vi] = new long long[roffset];
        memset(recv_v2[vi], 0, roffset * sizeof(long long));

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        async_rbruck_alltoallv::ParLinNa_servlet_v2(
            n, r, bblock, chunk_counts[c],
            (char*)sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
            (char*)recv_v2[vi], recvcounts, rdispls, MPI_LONG_LONG,
            MPI_COMM_WORLD, &servlet_ctx);
        t_v2[vi] = MPI_Wtime() - t0;
        vi++;
    }

    async_rbruck_alltoallv::servlet_shutdown(&servlet_ctx);

    /* correctness checks */
    int errors_coal { 0 }, errors_v1 { 0 };
    for (int i { 0 }; i < roffset; i++) {
        if (recv_mpi[i] != recv_coal[i]) errors_coal++;
        if (recv_mpi[i] != recv_v1[i]) errors_v1++;
    }

    int *errors_v2 { new int[valid_configs] };
    for (int c { 0 }; c < valid_configs; c++) {
        errors_v2[c] = 0;
        for (int i { 0 }; i < roffset; i++) {
            if (recv_mpi[i] != recv_v2[c][i]) errors_v2[c]++;
        }
    }

    /* reduce results to rank 0 */
    int total_coal { 0 }, total_v1 { 0 };
    MPI_Reduce(&errors_coal, &total_coal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&errors_v1, &total_v1, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int *total_v2 { new int[valid_configs] };
    for (int c { 0 }; c < valid_configs; c++) {
        total_v2[c] = 0;
        MPI_Reduce(&errors_v2[c], &total_v2[c], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    double max_t_mpi { 0 }, max_t_coal { 0 }, max_t_v1 { 0 };
    MPI_Reduce(&t_mpi, &max_t_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_coal, &max_t_coal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_v1, &max_t_v1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *max_t_v2 { new double[valid_configs] };
    for (int c { 0 }; c < valid_configs; c++) {
        max_t_v2[c] = 0;
        MPI_Reduce(&t_v2[c], &max_t_v2[c], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "Sasha's beloved ParLinNa Servlet v2 Test" << std::endl;
        std::cout << "procs=" << nprocs << " n=" << n << " r=" << r
                  << " bblock=" << bblock << " msg_size=" << msg_size << std::endl;
        std::cout << std::endl;

        std::cout << "MPI_Alltoallv:       " << max_t_mpi << "s" << std::endl;
        std::cout << "coalesced:           " << max_t_coal << "s" << std::endl;
        std::cout << "servlet v1:          " << max_t_v1 << "s" << std::endl;

        vi = 0;
        for (int c { 0 }; c < num_configs; c++) {
            if (chunk_counts[c] > msg_size) continue;
            std::cout << "servlet v2 (C=" << chunk_counts[c] << "):    "
                      << max_t_v2[vi] << "s" << std::endl;
            vi++;
        }

        std::cout << std::endl;

        bool all_pass { total_coal == 0 && total_v1 == 0 };
        vi = 0;
        for (int c { 0 }; c < num_configs; c++) {
            if (chunk_counts[c] > msg_size) continue;
            if (total_v2[vi] > 0) all_pass = false;
            vi++;
        }

        if (all_pass) {
            std::cout << "PASS: all results match MPI_Alltoallv (yipee)" << std::endl;
        } else {
            if (total_coal > 0) std::cout << "FAIL: coalesced " << total_coal << " mismatches" << std::endl;
            if (total_v1 > 0) std::cout << "FAIL: servlet v1 " << total_v1 << " mismatches" << std::endl;
            vi = 0;
            for (int c { 0 }; c < num_configs; c++) {
                if (chunk_counts[c] > msg_size) continue;
                if (total_v2[vi] > 0)
                    std::cout << "FAIL: servlet v2 (C=" << chunk_counts[c] << ") "
                              << total_v2[vi] << " mismatches" << std::endl;
                vi++;
            }
            std::cout << "(womp womp)" << std::endl;
        }
    }

    delete[] sendbuf;
    delete[] recv_mpi;
    delete[] recv_coal;
    delete[] recv_v1;
    for (int c { 0 }; c < valid_configs; c++) { delete[] recv_v2[c]; }
    delete[] recv_v2;
    delete[] t_v2;
    delete[] errors_v2;
    delete[] total_v2;
    delete[] max_t_v2;

    MPI_Finalize();
    return 0;
}
