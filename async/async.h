/*
 * async.h
 *
 * Author: xshthkr
 */

#ifndef ASYNC_H_
#define ASYNC_H_

#include "comm_servlet.h"

#include <mpi.h>

namespace async_rbruck_alltoallv {

int ParLogNa(
    int r, int b, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
	char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
    MPI_Comm comm);

int ParLinNa_coalesced(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm);

int ParLinNa_staggered(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm);

struct ParLinNa_BufferState {
    char *temp_send_buffer;
    int temp_send_capacity; // BYTES
    int *send_sizes;
    int *send_displs;
    int *recv_sizes;
    int *recv_displs;
};

struct ParLinNa_Handle {
    int current_idx;
    int nprocs;
    int ngroup;
    int n;
    int typesize;
    ParLinNa_BufferState buffers[2];
};

ParLinNa_Handle* ParLinNa_Init_handle(MPI_Comm comm, int n, int typesize);

void ParLinNa_Free_handle(ParLinNa_Handle* handle);

int ParLinNa_Phase1(
    ParLinNa_Handle *handle,
    int r,
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
    int *recvcounts, MPI_Datatype recvtype,
    MPI_Comm comm);

int ParLinNa_Phase2_submit(
    ParLinNa_Handle *handle,
    int bblock,
    char *recvbuf, int *rdispls, MPI_Datatype recvtype,
    MPI_Comm comm, ServletContext *servlet_ctx);

int ParLinNa_servlet(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm, ServletContext *servlet_ctx);

}

#endif /* ASYNC_H_ */