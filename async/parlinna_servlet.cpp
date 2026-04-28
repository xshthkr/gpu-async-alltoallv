/*
 * parlinna_servlet.cpp
 *
 * ParLinNa_coalesced with comm servlet for phase 2
 *
 * phase 1 (intra-node bruck) runs on main thread
 * phase 2 (inter-node scatter) is offloaded to comm servlet
 *
 * double-buffered slots enable pipelining:
 * calling ParLinNa_servlet repeatedly overlaps phase 2 of the
 * current call with phase 1 of the next call automatically
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

extern void* servlet_malloc(size_t size, bool use_hugepages);
extern void servlet_free(void *ptr);

/*
wait until a slot is available for new work
transitions DONE -> IDLE, spins on READY
*/
static void wait_slot_available(ServletSlot *slot) {
	while (true) {
		int s { slot->state.load(std::memory_order_acquire) };
		if (s == static_cast<int>(ServletState::IDLE)) return;
		if (s == static_cast<int>(ServletState::DONE)) {
			slot->state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
			return;
		}
		/* READY: still in-flight, spin */
	}
}

/*
ensure the slot has enough capacity for all heap-owned buffers:
- send_buffer: phase 2 send payload
- sizes_storage: per-node send/recv sizes and displacements
- extra_buffer: phase 1 bruck intermediate storage
- temp_recv_buffer: phase 1 bruck receive scratch

only reallocates when current capacity is insufficient
in steady-state loops (same n, nprocs, msg_size), zero allocations
*/
static void ensure_slot_capacity(
	ServletSlot *slot, size_t send_bytes, int ngroup,
	size_t extra_bytes, size_t temp_recv_bytes, bool use_hugepages)
{
	if (send_bytes > slot->send_buffer_capacity) {
		if (slot->send_buffer) servlet_free(slot->send_buffer);
		slot->send_buffer = (char*) servlet_malloc(send_bytes, use_hugepages);
		slot->send_buffer_capacity = send_bytes;
	}
	if (ngroup > slot->sizes_ngroup) {
		if (slot->sizes_storage) free(slot->sizes_storage); // sizes array is small, standard malloc
		slot->sizes_storage = (int*) malloc(4 * ngroup * sizeof(int));
		slot->sizes_ngroup = ngroup;
	}
	if (extra_bytes > slot->extra_buffer_capacity) {
		if (slot->extra_buffer) servlet_free(slot->extra_buffer);
		slot->extra_buffer = (char*) servlet_malloc(extra_bytes, use_hugepages);
		slot->extra_buffer_capacity = extra_bytes;
	}
	if (temp_recv_bytes > slot->temp_recv_buffer_capacity) {
		if (slot->temp_recv_buffer) servlet_free(slot->temp_recv_buffer);
		slot->temp_recv_buffer = (char*) servlet_malloc(temp_recv_bytes, use_hugepages);
		slot->temp_recv_buffer_capacity = temp_recv_bytes;
	}
}

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

	int ngroup { nprocs / n };
	int sw { static_cast<int>(ceil(log(n) / float(log(r)))) };
	int grank { rank % n };
	int gid { rank / n };
	int imax { rbruck_alltoallv_utils::pow(r, sw-1) * ngroup };
	int max_sd { (ngroup > imax) ? ngroup : imax };

	int local_max_count { 0 };
    int max_send_count { 0 };
    int id { 0 };
	int updated_sentcounts[nprocs];
    int rotate_index_array[nprocs];
    int pos_status[nprocs];
	int sent_blocks[max_sd];

	double et { MPI_Wtime() };
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i { 0 }; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	// acquire current slot, wait if previous phase 2 still in-flight
	int slot_idx { servlet_ctx->producer_idx };
	ServletSlot *slot { &servlet_ctx->slots[slot_idx] };
	wait_slot_available(slot);

	// ensure slot has enough buffer space for all workspace buffers
	size_t required_send_bytes { static_cast<size_t>(max_send_count) * typesize * nprocs };
	size_t required_extra_bytes { static_cast<size_t>(max_send_count) * typesize * nprocs };
	size_t required_recv_bytes { static_cast<size_t>(max_send_count) * typesize * max_sd };
	ensure_slot_capacity(slot, required_send_bytes, ngroup, required_extra_bytes, required_recv_bytes, servlet_ctx->config.use_hugepages);
	char *temp_send_buffer { slot->send_buffer };

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

	// workspace buffers owned by the slot — no malloc/free per call
	char *extra_buffer { slot->extra_buffer };
	char *temp_recv_buffer { slot->temp_recv_buffer };
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
	int index { 0 };
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

	// TODO: workspace buffers persist in the slot, NOT freed here


	/*
	PHASE 2: inter-node scatter via comm servlet

	compute per-node send/recv sizes and displacements (in bytes),
	fill the slot's CommDescriptor, submit, and return immediately
	the servlet thread executes the transfers asynchronously
	*/

	st = MPI_Wtime();

	// point desc sizes/displs at the slot's heap-allocated storage
	// layout: [send_sizes | send_displs | recv_sizes | recv_displs]
	int *send_sizes  { &slot->sizes_storage[0] };
	int *send_displs { &slot->sizes_storage[ngroup] };
	int *recv_sizes  { &slot->sizes_storage[2 * ngroup] };
	int *recv_displs { &slot->sizes_storage[3 * ngroup] };

	int soffset { 0 }, roffset { 0 };
	for (int i { 0 }; i < ngroup; i++) {
		int nsend_elems { 0 };
		int nrecv_elems { 0 };
		for (int j { 0 }; j < n; j++) {
			int id { i * n + j };
			int sn { updated_sentcounts[rotate_index_array[id]] };
			nsend_elems += sn;
			nrecv_elems += recvcounts[id];
		}
		send_sizes[i]  = nsend_elems * typesize;
		send_displs[i] = soffset;
		recv_sizes[i]  = nrecv_elems * typesize;
		recv_displs[i] = roffset;
		soffset += send_sizes[i];
		roffset += recv_sizes[i];
	}

	et = MPI_Wtime();
	prepSP_time = et - st;

	st = MPI_Wtime();

	// fill descriptor
	CommDescriptor *desc { &slot->desc };
	desc->send_buf   	= temp_send_buffer;
	desc->send_sizes 	= send_sizes;
	desc->send_displs	= send_displs;
	desc->recv_buf   	= recvbuf;
	desc->recv_sizes 	= recv_sizes;
	desc->recv_displs	= recv_displs;
	desc->ngroup     	= ngroup;
	desc->n          	= n;
	desc->gid        	= gid;
	desc->grank      	= grank;
	desc->bblock     	= bblock;
	desc->comm       	= comm;

	// submit to servlet and return immediately
	// the next call to ParLinNa_servlet will wait for this slot
	// when it wraps around (after NUM_SLOTS calls)
	servlet_submit(servlet_ctx);

	et = MPI_Wtime();
	SP_time = et - st;

	// NOTE: temp_send_buffer is NOT freed here
	// it lives in the slot and persists until reuse or shutdown

	return 0;
}

}
