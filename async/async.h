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

int ParLinNa_servlet(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm, ServletContext *servlet_ctx);

int ParLinNa_servlet_v2(
    int n, int r, int bblock, int num_chunks,
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
    MPI_Comm comm, ServletContext *servlet_ctx);

}

#endif /* ASYNC_H_ */