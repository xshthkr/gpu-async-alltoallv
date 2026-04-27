/*
 * parlinna_servlet.cpp
 *
 * ParLinNa_coalesced with comm servlet for phase 2
 *
 * phase 1 (intra-node bruck) is identical to ParLinNa_coalesced
 * phase 2 (inter-node scatter) is offloaded to the comm servlet
 *
 *      Author: xshthkr
 */

#include "async.h"

#include "comm_servlet.h"
#include "mpi.h"
#include "utils.h"

#include <iostream>
#include <math.h>
#include <memory.h>

static double init_time        { 0 };
static double findMax_time     { 0 };
static double rotateIndex_time { 0 };
static double alcCopy_time     { 0 };
static double getBlock_time    { 0 };
static double prepData_time    { 0 };
static double excgMeta_time    { 0 };
static double excgData_time    { 0 };
static double replace_time     { 0 };
static double orgData_time     { 0 };
static double prepSP_time      { 0 };
static double SP_time          { 0 };

namespace async_rbruck_alltoallv {

int ParLinNa_servlet(
    int n, int r, int bblock,
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
    MPI_Comm comm, ServletContext *servlet_ctx)
{
	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw;
	int grank, gid, imax, max_sd;
	int local_max_count { 0 };
    int max_send_count { 0 };
    int id { 0 };
	int updated_sentcounts[nprocs];
    int rotate_index_array[nprocs];
    int pos_status[nprocs];
	char *temp_send_buffer;
    char *extra_buffer;
    char *temp_recv_buffer;


	ngroup = nprocs / float(n); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = rbruck_alltoallv_utils::pow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	int sent_blocks[max_sd];
	double et { MPI_Wtime() };
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	st = MPI_Wtime();
	// 2. create local index array after rotation
	for (int i { 0 }; i < ngroup; i++) {
		int gsp { i * n };
		for (int j { 0 }; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time = et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs * sizeof(int));
	memcpy(updated_sentcounts, sendcounts, nprocs * sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count * typesize * nprocs);
	extra_buffer = (char*) malloc(max_send_count * typesize * nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count * typesize * max_sd);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	/*
	PHASE 1: intra-node bruck (identical to ParLinNa_coalesced)

	synchronous MPI_Sendrecv between ranks within the same node
	metadata exchange -> data exchange -> replace, per bruck round
	*/

	int spoint { 1 };
    int distance { 1 };
    int next_distance { r };
    int di { 0 };
	for (int x { 0 }; x < sw; x++) {
		for (int z { 1 }; z < r; z++) {
			di = 0;
            spoint = z * distance;
			if (spoint > n - 1) {break;}

			st = MPI_Wtime();
			// get the sent data-blocks
			for (int g { 0 }; g < ngroup; g++) {
				for (int i { spoint }; i < n; i += next_distance) {
					for (int j { i }; j < (i + distance); j++) {
						if (j > n - 1 ) { break; }
						int id = g * n + (j + grank) % n;
						sent_blocks[di++] = id;
					}
				}
			}
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount { 0 };
            int offset { 0 };
			for (int i { 0 }; i < di; i++) {
				int send_index { rotate_index_array[sent_blocks[i]] };
				metadata_send[i] = updated_sentcounts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index] * typesize], updated_sentcounts[send_index] * typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i] * max_send_count * typesize], updated_sentcounts[send_index] * typesize);
				offset += updated_sentcounts[send_index] * typesize;
			}

			int recv_proc { gid * n + (grank + spoint) % n };       // receive data from rank + 2^step process
			int send_proc { gid * n + (grank - spoint + n) % n };   // send data from rank - 2^k process

			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i { 0 }; i < di; i++) { sendCount += metadata_recv[i]; }
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();
			// 5) replace
			offset = 0;
			for (int i { 0 }; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i] * max_send_count * typesize], &temp_recv_buffer[offset], metadata_recv[i] * typesize);

				offset += metadata_recv[i] * typesize;
				pos_status[send_index] = 1;
				updated_sentcounts[send_index] = metadata_recv[i];
			}
			et = MPI_Wtime();
			replace_time += et - st;

		}
		distance *= r;
		next_distance *= r;
	}

	st = MPI_Wtime();
	// organize data into temp_send_buffer for inter-node scatter
	int index = 0;
	for (int i { 0 }; i < nprocs; i++) {
		int d { updated_sentcounts[rotate_index_array[i]] * typesize };
		if (grank == (i % n) ) {
			memcpy(&temp_send_buffer[index], &sendbuf[sdispls[i] * typesize], d);
		}
		else {
			memcpy(&temp_send_buffer[index], &extra_buffer[i * max_send_count * typesize], d);
		}
		index += d;
	}
	et = MPI_Wtime();
	orgData_time = et - st;

	free(temp_recv_buffer);
	free(extra_buffer);


	/*
	PHASE 2: inter-node scatter via comm servlet

	compute per-node send/recv sizes and displacements (in bytes),
	fill the servlet's CommDescriptor, submit, and wait
	*/

	st = MPI_Wtime();

	int nsend[ngroup], nrecv[ngroup], nsdisp[ngroup], nrdisp[ngroup];
	int soffset = 0, roffset = 0;
	for (int i { 0 }; i < ngroup; i++) {
		nsend[i] = 0;
        nrecv[i] = 0;
		for (int j { 0 }; j < n; j++) {
			int id { i * n + j };
			int sn { updated_sentcounts[rotate_index_array[id]] };
			nsend[i] += sn;
			nrecv[i] += recvcounts[id];
		}
		nsdisp[i] = soffset, nrdisp[i] = roffset;
		soffset += nsend[i] * typesize, roffset += nrecv[i] * typesize;
	}

	et = MPI_Wtime();
	prepSP_time = et - st;

	st = MPI_Wtime();

	// convert element counts to byte counts for the descriptor
	int nsend_bytes[ngroup], nrecv_bytes[ngroup];
	for (int i { 0 }; i < ngroup; i++) {
		nsend_bytes[i] = nsend[i] * typesize;
		nrecv_bytes[i] = nrecv[i] * typesize;
	}

	// DONT FORGET THIS SHIT
	// HELLA IMPORTANT
	// TODO: nsend_bytes, nsdisp, nrecv_bytes, nrdisp are stack-allocated
	// safe now because servlet_wait() blocks before this frame returns
	// if real overlap is added later, heap-allocate these or copy into desc
	
	// fill descriptor and submit to servlet
	CommDescriptor *desc = &servlet_ctx->desc;
	desc->send_buf   	= temp_send_buffer;
	desc->send_sizes 	= nsend_bytes;
	desc->send_displs	= nsdisp;
	desc->recv_buf   	= recvbuf;
	desc->recv_sizes 	= nrecv_bytes;
	desc->recv_displs	= nrdisp;
	desc->ngroup     	= ngroup;
	desc->n          	= n;
	desc->gid        	= gid;
	desc->grank      	= grank;
	desc->bblock     	= bblock;
	desc->comm       	= comm;

	servlet_submit(servlet_ctx);

	// OVERLAP WINDOW
	// compute ranks are free to do useful work here
	// for now spin-wait

	servlet_wait(servlet_ctx);

	et = MPI_Wtime();
	SP_time = et - st;

	free(temp_send_buffer);

	return 0;
}

}
