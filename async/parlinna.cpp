/*
 * parlinna.cpp
 *
 *      From ParLinNa (TuNA(l,g)) originally implemented by kokofan
 */

#include "async.h"
#include "utils.h"

#include <iostream>
#include <math.h>
#include <memory.h>

double init_time        { 0 };
double findMax_time     { 0 };
double rotateIndex_time { 0 };
double alcCopy_time     { 0 };
double getBlock_time    { 0 };
double prepData_time    { 0 };
double excgMeta_time    { 0 };
double excgData_time    { 0 };
double replace_time     { 0 };
double orgData_time     { 0 };
double prepSP_time      { 0 };
double SP_time          { 0 };

namespace async_rbruck_alltoallv {

/*
Parameters:
- number of processes per hierarchical group n (ranks per physical node)
- radix of intra-group communication tree r
- number of blocks to exchange per super-packet bblock

Exploits hierarchy of HPC clusters where intra-node communication is
much faster than inter-node communication. Processes within the same node
use a localized version of ParLogNa. Data designated for other nodes are
consolidated. Direct inter-node communication exchanges consolidated data
with other nodes.
*/
int ParLinNa_coalesced(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm)
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
	int mpi_errno { MPI_SUCCESS };

	ngroup = nprocs / float(n); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = rbruck_alltoallv_utils::pow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	// if (rank == 0) {
    //     std::cout << "Math -- TTPL: " << nprocs << " " << r << " " << sw << " " <<  imax << " " << max_sd << std::endl;
	// }

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

	// Intra-Bruck
	getBlock_time = 0;
    prepData_time = 0;
    excgMeta_time = 0;
    excgData_time = 0;
    replace_time = 0;

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
	// organize data
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


	if (bblock <= 0 || bblock > ngroup) bblock = ngroup;

	st = MPI_Wtime();
	MPI_Request* req = (MPI_Request*)malloc(2*bblock*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*bblock*sizeof(MPI_Status));
	int req_cnt { 0 };
    int ss { 0 };

	for (int ii { 0 }; ii < ngroup; ii += bblock) {
		req_cnt = 0;
		ss = ngroup - ii < bblock ? ngroup - ii : bblock;

		for (int i { 0 }; i < ss; i++) {
			int nsrc { (gid + i + ii) % ngroup };
			int src { nsrc * n + grank }; // avoid always to reach first master node

			mpi_errno = MPI_Irecv(&recvbuf[nrdisp[nsrc]], nrecv[nsrc] * typesize, MPI_CHAR, src, 0, comm, &req[req_cnt++]);
			if (mpi_errno != MPI_SUCCESS) {return -1;}

		}

		for (int i { 0 }; i < ss; i++) {
			int ndst { (gid - i - ii + ngroup) % ngroup };
			int dst { ndst * n + grank };

			mpi_errno = MPI_Isend(&temp_send_buffer[nsdisp[ndst]], nsend[ndst] * typesize, MPI_CHAR, dst, 0, comm, &req[req_cnt++]);
			if (mpi_errno != MPI_SUCCESS) {return -1;}
		}

		mpi_errno = MPI_Waitall(req_cnt, req, stat);
		if (mpi_errno != MPI_SUCCESS) {return -1;}
	}

	free(req);
	free(stat);

	free(temp_send_buffer);
	et = MPI_Wtime();
	SP_time = et - st;

	return 0;
}

int ParLinNa_staggered(
    int n, int r, int bblock, 
    char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, 
    char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, 
    MPI_Comm comm)
{
	double st { MPI_Wtime() };
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
	int mpi_errno { MPI_SUCCESS };

	ngroup = nprocs / float(n); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = rbruck_alltoallv_utils::pow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	// if (rank == 0) {
	// 	std::cout << "Math -- TTPL: " << nprocs << " " << r << " " << sw << " " <<  imax << " " << max_sd << std::endl;
	// }

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
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcounts, sendcounts, nprocs * sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count * typesize * max_sd);
	extra_buffer = (char*) malloc(max_send_count * typesize * nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count * typesize * max_sd);
	memcpy(&recvbuf[rdispls[rank] * typesize], &sendbuf[sdispls[rank] * typesize], recvcounts[rank] * typesize);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	// Intra-Bruck
	getBlock_time = 0;
    prepData_time = 0;
    excgMeta_time = 0;
    excgData_time = 0;
    replace_time = 0;

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
					for (int j { i }; j < (i+distance); j++) {
						if (j > n - 1 ) { break; }
						int id { g * n + (j + grank) % n };
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

			int recv_proc { gid * n + (grank + spoint) % n }; // receive data from rank + 2^step process
			int send_proc { gid * n + (grank - spoint + n) % n }; // send data from rank - 2^k process

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
				int send_index { rotate_index_array[sent_blocks[i]] };

				int origin_index { (sent_blocks[i] % n - grank + n) % n };
				if (sent_blocks[i] / n  == gid && origin_index < next_distance) {
					memcpy(&recvbuf[rdispls[sent_blocks[i]] * typesize], &temp_recv_buffer[offset], metadata_recv[i] * typesize);
				}
				else {
					memcpy(&extra_buffer[sent_blocks[i] * max_send_count * typesize], &temp_recv_buffer[offset], metadata_recv[i] * typesize);
				}

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
	free(temp_recv_buffer);
	free(temp_send_buffer);

    if (bblock <= 0 || bblock > nprocs) bblock = nprocs;

    MPI_Request* reqarray = (MPI_Request *)malloc(2 * bblock * sizeof(MPI_Request));
    MPI_Status* starray = (MPI_Status *)malloc(2 * bblock * sizeof(MPI_Status));
    int req_cnt { 0 };
    int ss { 0 };

	/* post only bblock isends/irecvs at a time as suggested by Tony Ladd */
	for (int ii { 0 }; ii < nprocs; ii += bblock) {
		req_cnt = 0;
		ss = nprocs - ii < bblock ? nprocs - ii : bblock;

		/* do the communication -- post ss sends and receives: */
		for (int i { 0 }; i < ss; i++) {
			int gi { (ii + i) / n };
			int gr { (ii + i) % n };
			int nsrc { (gid + gi) % ngroup };
			if (nsrc == gid) { continue; }
			id = nsrc * n + gr;

			int src { nsrc * n + grank };
			mpi_errno =  MPI_Irecv(&recvbuf[rdispls[id] * typesize], recvcounts[id] * typesize, MPI_CHAR, src, gr, comm, &reqarray[req_cnt++]);
			if (mpi_errno != MPI_SUCCESS) {return -1;}
		}

		for (int i { 0 }; i < ss; i++) {
			int gi { (ii + i) / n };
			int gr { (ii + i) % n };
			int ndst { (gid - gi + ngroup) % ngroup };
			if (ndst == gid) { continue; }
			int dst { ndst * n + grank };
			id = ndst * n + gr;

			int ds { updated_sentcounts[rotate_index_array[id]] * typesize };
			if (gr == grank) {
				mpi_errno = MPI_Isend(&sendbuf[sdispls[id] * typesize], ds, MPI_CHAR, dst, gr, comm, &reqarray[req_cnt++]);
			}
			else {
				mpi_errno = MPI_Isend(&extra_buffer[(id) * max_send_count * typesize], ds, MPI_CHAR, dst, gr, comm, &reqarray[req_cnt++]);
			}
			if (mpi_errno != MPI_SUCCESS) {return -1;}
		}

		mpi_errno = MPI_Waitall(req_cnt, reqarray, starray);
		if (mpi_errno != MPI_SUCCESS) {return -1;}
	}

	free(reqarray);
	free(starray);

	free(extra_buffer);
	et = MPI_Wtime();
	SP_time = et - st;

	return 0;
}

}