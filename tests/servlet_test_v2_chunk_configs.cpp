/*
 * servlet_test_v2_configs.cpp
 *
 * Run ParLinNa_servlet_v2 across a sweep of message sizes, radix values,
 * block sizes, and chunk counts. The test builds a random send-count
 * distribution for each message size, compares to MPI_Alltoallv, and repeats
 * for the requested number of iterations.
 *
 * usage: mpirun -n <nprocs> ./servlet_test_v2_configs <loop-count> <n> <bblock> <radix-list...>
 *
 *      Author: xshthkr
 */

// TODO: benchmark across multiple servlet configuration combinations

#include "../async/async.h"
#include "../async/comm_servlet.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
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
        if (rank == 0) {
            std::cout << "Usage: mpirun -n <nprocs> " << argv[0]
                      << " <loop-count> <ncores-per-node> <bblock> <radix-list...>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int loopcount { std::atoi(argv[1]) };
    int ncores { std::atoi(argv[2]) };
    int bblock { std::atoi(argv[3]) };
    std::vector<int> radix_list;
    for (int arg { 4 }; arg < argc; ++arg) {
        radix_list.push_back(std::atoi(argv[arg]));
    }

    if (radix_list.empty()) {
        if (rank == 0) {
            std::cerr << "ERROR: radix-list must contain at least one entry" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (nprocs % ncores != 0) {
        if (rank == 0) {
            std::cerr << "ERROR: nprocs (" << nprocs << ") must be divisible by n (" << ncores << ")" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int ngroup { nprocs / ncores };
    std::mt19937_64 rng(static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + rank);

    for (int msg_size { 2 }; msg_size <= 1024; msg_size *= 2) {
        int sendcounts[nprocs];
        int sdispls[nprocs];
        int recvcounts[nprocs];
        int rdispls[nprocs];

        std::uniform_int_distribution<int> dist(1, msg_size);
        for (int i { 0 }; i < nprocs; ++i) {
            sendcounts[i] = dist(rng);
        }
        std::shuffle(sendcounts, sendcounts + nprocs, rng);

        MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

        int soffset { 0 };
        int roffset { 0 };
        for (int i { 0 }; i < nprocs; ++i) {
            sdispls[i] = soffset;
            rdispls[i] = roffset;
            soffset += sendcounts[i];
            roffset += recvcounts[i];
        }

        long long *sendbuf = new long long[soffset];
        long long *recv_mpi = new long long[roffset];

        int idx { 0 };
        for (int i { 0 }; i < nprocs; ++i) {
            for (int j { 0 }; j < sendcounts[i]; ++j) {
                sendbuf[idx++] = static_cast<long long>(rank) * 1000000LL + static_cast<long long>(i) * 1000LL + j;
            }
        }
        std::memset(recv_mpi, 0, roffset * sizeof(long long));

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 { MPI_Wtime() };
        MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
                      recv_mpi, recvcounts, rdispls, MPI_LONG_LONG,
                      MPI_COMM_WORLD);
        double t_mpi { MPI_Wtime() - t0 };

        // if (rank == 0) {
        //     std::cout << "\n[msg_size=" << msg_size << "] MPI_Alltoallv baseline: " << t_mpi << "s" << std::endl;
        // }

        for (int r_value : radix_list) {
            int radix { r_value };
            for (int bsize { 1 }; bsize <= ngroup; bsize *= 2) {
                std::vector<int> chunk_counts;
                int chunk { 2 };
                while (chunk <= ncores ) {
                    chunk_counts.push_back(chunk);
                    chunk *= 2;
                }
                if (chunk_counts.empty()) {
                    chunk_counts.push_back(1);
                }

                async_rbruck_alltoallv::ServletConfig cfg { async_rbruck_alltoallv::servlet_default_config() };
                async_rbruck_alltoallv::ServletContext servlet_ctx;
                if (async_rbruck_alltoallv::servlet_init(&servlet_ctx, &cfg) != 0) {
                    if (rank == 0) {
                        std::cerr << "ERROR: servlet_init failed for radix=" << radix << " bsize=" << bsize << std::endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                int total_errors { 0 };
                double max_time_for_config { 0.0 };

                for (int it { 0 }; it < loopcount; ++it) {
                    for (int num_chunks : chunk_counts) {
                        long long *recv_srv = new long long[roffset];
                        std::memset(recv_srv, 0, roffset * sizeof(long long));

                        MPI_Barrier(MPI_COMM_WORLD);
                        t0 = MPI_Wtime();
                        async_rbruck_alltoallv::ParLinNa_servlet_v2(
                            ncores, radix, bblock, num_chunks,
                            reinterpret_cast<char*>(sendbuf), sendcounts, sdispls, MPI_LONG_LONG,
                            reinterpret_cast<char*>(recv_srv), recvcounts, rdispls, MPI_LONG_LONG,
                            MPI_COMM_WORLD, &servlet_ctx);
                        async_rbruck_alltoallv::servlet_wait(&servlet_ctx);
                        double elapsed { MPI_Wtime() - t0 };

                        double max_elapsed { 0.0 };
                        MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                        if (max_elapsed > max_time_for_config) {
                            max_time_for_config = max_elapsed;
                        }

                        int local_errors { 0 };
                        for (int i { 0 }; i < roffset; ++i) {
                            if (recv_srv[i] != recv_mpi[i]) {
                                local_errors += 1;
                            }
                        }
                        int global_errors { 0 };
                        MPI_Allreduce(&local_errors, &global_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                        total_errors += global_errors;

                        if (rank == 0) {
                            std::cout << "[ServletV2Chunk] " << nprocs << ", " << msg_size << ", " << num_chunks << ", " << bsize << ", " << radix << ", " << max_elapsed << std::endl;
                            // std::cout << "[ServletV2Chunk] msg=" << msg_size
                            //           << " radix=" << radix
                            //           << " b=" << bsize
                            //           << " chunks=" << num_chunks
                            //           << " iter=" << it
                            //           << " max_time=" << max_elapsed
                            //           << "s errors=" << global_errors << std::endl;
                        }

                        delete[] recv_srv;
                    }
                }

                async_rbruck_alltoallv::servlet_shutdown(&servlet_ctx);

                if (rank == 0 && total_errors != 0) {
                    // std::cout << "[ServletV2Chunk] msg=" << msg_size
                    //           << " radix=" << radix
                    //           << " b=" << bsize
                    //           << " total_errors=" << total_errors
                    //           << std::endl;
                    if (total_errors != 0) {
                        std::cout << " FAIL (total_errors=" << total_errors << ")" << std::endl;
                    }
                }
            }
        }

        delete[] sendbuf;
        delete[] recv_mpi;
    }

    MPI_Finalize();
    return 0;
}
