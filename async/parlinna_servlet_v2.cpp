/*
 * parlinna_servlet_v2.cpp
 *
 * true single-call phase 1 / phase 2 pipelining via payload chunking
 *
 * splits the message payload into C chunks, then pipelines:
 *   chunk 0: phase 1 -> submit phase 2 on slot 0
 *   chunk 1: phase 1 -> submit phase 2 on slot 1 (overlaps with chunk 0 transfer)
 *   ...
 *
 * phase 2 receives into per-chunk temp buffers (contiguous layout),
 * then a final scatter copies data to the correct positions in recvbuf
 *
 *      Author: xshthkr
 */

#include "async.h"
#include "comm_servlet.h"
#include "mpi.h"
#include "utils.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#define CHECK_CALL(call) \
  do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
      char errstr[MPI_MAX_ERROR_STRING]; \
      int errlen; \
      MPI_Error_string(err, errstr, &errlen); \
      fprintf(stderr, "[%s:%d] MPI error: %s\n", __FILE__, __LINE__, errstr); \
      MPI_Abort(MPI_COMM_WORLD, err); \
    } \
  } while (0)

namespace async_rbruck_alltoallv {

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

extern void* servlet_malloc(size_t size, bool use_hugepages);
extern void servlet_free(void *ptr);

static bool wait_slot_available(ServletSlot *slot) {
	while (true) {
		int s { slot->state.load(std::memory_order_acquire) };
		if (s == static_cast<int>(ServletState::IDLE)) return false;
		if (s == static_cast<int>(ServletState::DONE)) {
			slot->state.store(static_cast<int>(ServletState::IDLE), std::memory_order_release);
			return true;
		}
	}
}

static void scatter_chunk_recv_buffer(ServletSlot *slot) {
	if (!(slot->final_recvbuf) || !(slot->final_recvcounts) || !(slot->final_rdispls)) return;

	int n { slot->desc.n };
	int ngroup { slot->desc.ngroup };
	int num_chunks { slot->num_chunks };
	int chunk_id { slot->chunk_id };
	int typesize { slot->typesize };
	char *dst_base { slot->final_recvbuf };
	int *recvcounts { slot->final_recvcounts };
	int *rdispls { slot->final_rdispls };
	char *src { slot->chunk_recv_buffer };

	int offset { 0 };
	for (int i { 0 }; i < ngroup; i++) {
		for (int j { 0 }; j < n; j++) {
			int id { i * n + j };
			int base_r { recvcounts[id] / num_chunks };
			int rem_r  { recvcounts[id] % num_chunks };
			int chunk_offset { chunk_id * base_r + std::min(chunk_id, rem_r) };
			int count { base_r + (chunk_id < rem_r ? 1 : 0) };
			if (count > 0) {
				memcpy(dst_base + (rdispls[id] + chunk_offset) * typesize,
				       &src[offset], count * typesize);
				offset += count * typesize;
			}
		}
	}

	slot->final_recvbuf = nullptr;
	slot->final_recvcounts = nullptr;
	slot->final_rdispls = nullptr;
}

static void ensure_slot_capacity(
	ServletSlot *slot, size_t send_bytes, int ngroup,
	size_t extra_bytes, size_t temp_recv_bytes,
	size_t chunk_recv_bytes, bool use_hugepages)
{
	if (send_bytes > slot->send_buffer_capacity) {
		if (slot->send_buffer) servlet_free(slot->send_buffer);
		slot->send_buffer = (char*) servlet_malloc(send_bytes, use_hugepages);
		slot->send_buffer_capacity = send_bytes;
	}
	if (ngroup > slot->sizes_ngroup) {
		if (slot->sizes_storage) free(slot->sizes_storage); // sizes is small
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
	if (chunk_recv_bytes > slot->chunk_recv_buffer_capacity) {
		if (slot->chunk_recv_buffer) servlet_free(slot->chunk_recv_buffer);
		slot->chunk_recv_buffer = (char*) servlet_malloc(chunk_recv_bytes, use_hugepages);
		slot->chunk_recv_buffer_capacity = chunk_recv_bytes;
	}
	// if (!(slot->send_buffer) || !(slot->sizes_storage) || !(slot->extra_buffer) || !(slot->temp_recv_buffer) || !(slot->chunk_recv_buffer)) {
	// 	fprintf(stderr, "ERROR: malloc failed!\n");
    // 		MPI_Abort(MPI_COMM_WORLD, 1);
	// }
}

/*
run phase 1 bruck on a chunk, pack the slot descriptor for phase 2
phase 2 receives directly into the final recvbuf using per-group byte offsets
*/
static int run_phase1_chunk(
	int n, int r, int nprocs, int typesize,
	int ngroup, int sw, int grank, int gid,
	char *sendbuf_base, int *chunk_sendcounts, int *chunk_sdispls,
	int *chunk_recvcounts, size_t chunk_recv_bytes,
	char *recvbuf, int *recvcounts, int *rdispls,
	int chunk_id, int num_chunks,
	MPI_Comm comm, MPI_Comm local_comm, int bblock, ServletSlot *slot, ServletContext *servlet_ctx)
{
	int rank;
	MPI_Comm_rank(comm, &rank);
	int imax { rbruck_alltoallv_utils::pow(r, sw-1) * ngroup };
	int max_sd { (ngroup > imax) ? ngroup : imax };

	double st { MPI_Wtime() }, et;

	st = MPI_Wtime();
	int local_max_count { 0 };
	int max_send_count { 0 };
	int id { 0 };
	int updated_sentcounts[nprocs];
	int rotate_index_array[nprocs];
	int pos_status[nprocs];
	int sent_blocks[max_sd];

	for (int i { 0 }; i < nprocs; i++) {
		if (chunk_sendcounts[i] > local_max_count)
			local_max_count = chunk_sendcounts[i];
	}
	et = MPI_Wtime();
	init_time += et - st;

	st = MPI_Wtime();
	// CHECK_CALL(MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm));
    MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time += et - st;

	size_t required_send { static_cast<size_t>(max_send_count) * typesize * nprocs };
	size_t required_extra { static_cast<size_t>(max_send_count) * typesize * nprocs };
	size_t required_recv { static_cast<size_t>(max_send_count) * typesize * max_sd };
	ensure_slot_capacity(slot, required_send, ngroup, required_extra, required_recv, chunk_recv_bytes, servlet_ctx->config.use_hugepages);
	
	char *temp_send_buffer { slot->send_buffer };

	st = MPI_Wtime();
	for (int i { 0 }; i < ngroup; i++) {
		int gsp { i * n };
		for (int j { 0 }; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time += et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs * sizeof(int));
	memcpy(updated_sentcounts, chunk_sendcounts, nprocs * sizeof(int));

	char *extra_buffer { slot->extra_buffer };
	char *temp_recv_buffer { slot->temp_recv_buffer };
	et = MPI_Wtime();
	alcCopy_time += et - st;

	// PHASE 1: intra-node bruck
	int spoint { 1 }, distance { 1 }, next_distance { r }, di { 0 };

	for (int x { 0 }; x < sw; x++) {
		for (int z { 1 }; z < r; z++) {
			di = 0;
			spoint = z * distance;
			if (spoint > n - 1) break;

			st = MPI_Wtime();
			for (int g { 0 }; g < ngroup; g++) {
				for (int i { spoint }; i < n; i += next_distance) {
					for (int j { i }; j < (i + distance); j++) {
						if (j > n - 1) break;
						int id { g * n + (j + grank) % n };
						sent_blocks[di++] = id;
					}
				}
			}
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			int metadata_send[di];
			int sendCount { 0 }, offset { 0 };
			for (int i { 0 }; i < di; i++) {
				int send_index { rotate_index_array[sent_blocks[i]] };
				metadata_send[i] = updated_sentcounts[send_index];
				if (pos_status[send_index] == 0)
					memcpy(&temp_send_buffer[offset],
						   &sendbuf_base[chunk_sdispls[send_index] * typesize],
						   updated_sentcounts[send_index] * typesize);
				else
					memcpy(&temp_send_buffer[offset],
						   &extra_buffer[sent_blocks[i] * max_send_count * typesize],
						   updated_sentcounts[send_index] * typesize);
				offset += updated_sentcounts[send_index] * typesize;
			}

			int recv_proc { (grank + spoint) % n };
			int send_proc { (grank - spoint + n) % n };
			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			int metadata_recv[di];
			// CHECK_CALL(MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0,
			// 			 metadata_recv, di, MPI_INT, recv_proc, 0,
			// 			 comm, MPI_STATUS_IGNORE));
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0,
						 metadata_recv, di, MPI_INT, recv_proc, 0,
						 local_comm, MPI_STATUS_IGNORE);
			for (int i { 0 }; i < di; i++) sendCount += metadata_recv[i];
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// CHECK_CALL(MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1,
			// 			 temp_recv_buffer, sendCount * typesize, MPI_CHAR, recv_proc, 1,
			// 			 comm, MPI_STATUS_IGNORE));
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1,
						 temp_recv_buffer, sendCount * typesize, MPI_CHAR, recv_proc, 1,
						 local_comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();
			offset = 0;
			for (int i { 0 }; i < di; i++) {
				int send_index { rotate_index_array[sent_blocks[i]] };
				memcpy(&extra_buffer[sent_blocks[i] * max_send_count * typesize],
					   &temp_recv_buffer[offset], metadata_recv[i] * typesize);
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
		if (grank == (i % n))
			memcpy(&temp_send_buffer[index],
				   &sendbuf_base[chunk_sdispls[i] * typesize], d);
		else
			memcpy(&temp_send_buffer[index],
				   &extra_buffer[i * max_send_count * typesize], d);
		index += d;
	}
	et = MPI_Wtime();
	orgData_time += et - st;

	st = MPI_Wtime();
	// compute per-node sizes/displs into slot storage (final recvbuf layout)
	int *send_sizes  { &slot->sizes_storage[0] };
	int *send_displs { &slot->sizes_storage[ngroup] };
	int *recv_sizes  { &slot->sizes_storage[2 * ngroup] };
	int *recv_displs { &slot->sizes_storage[3 * ngroup] };

	int soff { 0 };
	int roffset { 0 };
	for (int i { 0 }; i < ngroup; i++) {
		int nsend { 0 }, nrecv { 0 };
		for (int j { 0 }; j < n; j++) {
			int id { i * n + j };
			nsend += updated_sentcounts[rotate_index_array[id]];
			nrecv += chunk_recvcounts[id];
		}
		send_sizes[i]  = nsend * typesize;
		send_displs[i] = soff;
		recv_sizes[i]  = nrecv * typesize;

		int first_rank { i * n };
		int group_offset { 0 };
		for (int j { 0 }; j < n; j++) {
			int id { first_rank + j };
			int base_r { recvcounts[id] / num_chunks };
			int rem_r  { recvcounts[id] % num_chunks };
			int chunk_offset { chunk_id * base_r + std::min(chunk_id, rem_r) };
			group_offset += chunk_offset;
		}
		if (num_chunks == 1) {
			recv_displs[i] = (rdispls[first_rank] + group_offset) * typesize;
		} else {
			recv_displs[i] = roffset;
		}
		soff += send_sizes[i];
		roffset += recv_sizes[i];
	}
	et = MPI_Wtime();
	prepSP_time += et - st;

	st = MPI_Wtime();
	CommDescriptor *desc { &slot->desc };
	desc->send_buf    = temp_send_buffer;
	desc->send_sizes  = send_sizes;
	desc->send_displs = send_displs;
	desc->recv_sizes  = recv_sizes;
	if (num_chunks == 1) {
		desc->recv_buf    = recvbuf;
		desc->recv_displs = recv_displs;
	} else {
		desc->recv_buf    = slot->chunk_recv_buffer;
		desc->recv_displs = recv_displs;
		slot->final_recvbuf = recvbuf;
		slot->final_recvcounts = recvcounts;
		slot->final_rdispls = rdispls;
		slot->chunk_id = chunk_id;
		slot->num_chunks = num_chunks;
		slot->typesize = typesize;
	}
	desc->ngroup      = ngroup;
	desc->n           = n;
	desc->gid         = gid;
	desc->grank       = grank;
	desc->bblock      = bblock;
	desc->comm        = comm;
	et = MPI_Wtime();
	SP_time += et - st;

	return 0;
}


int ParLinNa_servlet_v2(
	int n, int r, int bblock, int num_chunks,
	char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
	char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
	MPI_Comm comm, ServletContext *servlet_ctx)
{
	double st { MPI_Wtime() }, et;
	if (r < 2) return -1;
	if (num_chunks < 1) num_chunks = 1;

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	if (r > n) r = n;

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	MPI_Comm local_comm;
	MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
	int local_rank { 0 };
	int local_size { 0 };
	MPI_Comm_rank(local_comm, &local_rank);
	MPI_Comm_size(local_comm, &local_size);

	if (local_size != n) {
		if (rank == 0) {
			std::cerr << "ERROR: detected " << local_size
				<< " ranks on this node but ncores=" << n << " was requested" << std::endl;
		}
		MPI_Abort(comm, 1);
	}

	int grank { local_rank };
	int ngroup { nprocs / n };
	int sw { static_cast<int>(ceil(log(n) / float(log(r)))) };

	MPI_Comm node_leader_comm = MPI_COMM_NULL;
	MPI_Comm_split(comm, (local_rank == 0) ? 0 : MPI_UNDEFINED, rank, &node_leader_comm);
	int node_id { 0 };
	if (local_rank == 0) {
		MPI_Comm_rank(node_leader_comm, &node_id);
	}
	MPI_Bcast(&node_id, 1, MPI_INT, 0, local_comm);
	int gid { node_id };
	if (node_leader_comm != MPI_COMM_NULL) {
		MPI_Comm_free(&node_leader_comm);
	}

	et = MPI_Wtime();
	init_time += et - st;

	// single chunk: skip chunking overhead, write directly to recvbuf
	if (num_chunks == 1) {
		int slot_idx { servlet_ctx->producer_idx };
		ServletSlot *slot { &servlet_ctx->slots[slot_idx] };
		wait_slot_available(slot);

		// for single chunk, chunk counts == full counts
		int *chunk_recvcounts { new int[nprocs] };
		memcpy(chunk_recvcounts, recvcounts, nprocs * sizeof(int));

		// compute total recv bytes for temp buffer
		size_t total_recv { 0 };
		for (int i { 0 }; i < nprocs; i++) total_recv += recvcounts[i];

		run_phase1_chunk(n, r, nprocs, typesize, ngroup, sw, grank, gid,
						 sendbuf, sendcounts, sdispls, chunk_recvcounts,
						 total_recv * typesize, recvbuf, recvcounts, rdispls, 0, 1, comm, local_comm, bblock, slot, servlet_ctx);
		st = MPI_Wtime();
		servlet_submit(servlet_ctx);
		et = MPI_Wtime();
		SP_time += et - st;
		servlet_wait(servlet_ctx);

		delete[] chunk_recvcounts;
		MPI_Comm_free(&local_comm);
		return 0;
	}

	int *chunk_sendcounts { new int[nprocs] };
	int *chunk_sdispls    { new int[nprocs] };
	int *chunk_recvcounts { new int[nprocs] };

	for (int c { 0 }; c < num_chunks; c++) {
		// compute this chunk's counts and displacements
		size_t chunk_total_recv { 0 };
		for (int i { 0 }; i < nprocs; i++) {
			int base_s { sendcounts[i] / num_chunks };
			int rem_s  { sendcounts[i] % num_chunks };
			chunk_sendcounts[i] = base_s + (c < rem_s ? 1 : 0);
			chunk_sdispls[i]    = sdispls[i] + c * base_s + std::min(c, rem_s);

			int base_r { recvcounts[i] / num_chunks };
			int rem_r  { recvcounts[i] % num_chunks };
			chunk_recvcounts[i] = base_r + (c < rem_r ? 1 : 0);
			chunk_total_recv += chunk_recvcounts[i];
		}

		int slot_idx { servlet_ctx->producer_idx };
		ServletSlot *slot { &servlet_ctx->slots[slot_idx] };
		if (wait_slot_available(slot)) {
			scatter_chunk_recv_buffer(slot);
		}

		run_phase1_chunk(n, r, nprocs, typesize, ngroup, sw, grank, gid,
				sendbuf, chunk_sendcounts, chunk_sdispls,
				chunk_recvcounts, chunk_total_recv * typesize,
				recvbuf, recvcounts, rdispls, c, num_chunks,
				comm, local_comm, bblock, slot, servlet_ctx);
		if (n == 2 && bblock == 2 && num_chunks == 8) {
			fprintf(stderr, "[DBG ParLinNa] submit chunk=%d slot=%d\n", c, slot_idx);
		}
		st = MPI_Wtime();
		servlet_submit(servlet_ctx);
		et = MPI_Wtime();
		SP_time += et - st;
	}

	// wait for all in-flight chunks to complete
	servlet_wait(servlet_ctx);
	if (n == 2 && bblock == 2 && num_chunks == 8) {
		fprintf(stderr, "[DBG ParLinNa] final servlet_wait complete\n");
	}
	for (int s { 0 }; s < NUM_SLOTS; s++) {
		scatter_chunk_recv_buffer(&servlet_ctx->slots[s]);
	}

	delete[] chunk_sendcounts;
	delete[] chunk_sdispls;
	delete[] chunk_recvcounts;

	MPI_Comm_free(&local_comm);
	return 0;
}

}
