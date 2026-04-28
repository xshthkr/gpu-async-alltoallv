/*
 * twophase_tunable_rbruckv.cpp
 *
 *  Created on: Jan 4, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int tuna2_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { r = 2; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	if ( r > nprocs - 1 ) { r = nprocs - 1; }
	if (b <= 0 || b > nprocs) b = nprocs;

	int w, max_rank, nlpow, d, K, i, num_reqs;
	int local_max_count=0, max_send_count=0;
	int rotate_index_array[nprocs];
	w = 0, nlpow = 1, max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }   // number of bits required of r representation
    for (i = 0; i < w - 1; i++) { nlpow *= r; }   // maximum send number of elements
	d = (nlpow*r - nprocs) / nlpow; // calculate the number of highest digits
	K = w * (r - 1) - d; // the total number of communication rounds

	int sendNcopy[nprocs];
	char *extra_buffer, *temp_recv_buffer, *temp_send_buffer;
	int spoint = 1, distance = 1, next_distance = distance*r, di = 0;

	if (K < nprocs - 1) {
		// 1. Find max send count
		for (i = 0; i < nprocs; i++) {
			if (sendcounts[i] > local_max_count) { local_max_count = sendcounts[i]; }
		}
		MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

		// 2. create local index array after rotation
		for (i = 0; i < nprocs; i++) { rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs; }

		// 3. exchange data with log(P) steps
		extra_buffer = (char*) malloc(max_send_count * typesize * nprocs);
		temp_recv_buffer = (char*) malloc(max_send_count * nprocs * typesize);
	    if (extra_buffer == nullptr || temp_recv_buffer == nullptr) {
	        std::cerr << "extra_buffer or temp_recv_buffer allocation failed!" << std::endl;
	        return 1; // Exit program with error
	    }
		temp_send_buffer = (char*) malloc(max_send_count * nprocs * typesize);
	    if (temp_send_buffer == nullptr) {
	        std::cerr << "temp_send_buffer allocation failed!" << std::endl;
	        return 1; // Exit program with error
	    }



	}

	// copy data that need to be sent to each rank itself
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);


	int sent_blocks[r-1][nlpow];
	int metadata_recv[r-1][nlpow];
	int nc, rem, ns, ze, ss;
	spoint = 1, distance = 1, next_distance = distance*r;

//	MPI_Request* reqs = (MPI_Request *) malloc(2 * b * sizeof(MPI_Request));
	MPI_Request* reqs = (MPI_Request *) malloc(2 * r * sizeof(MPI_Request));
    if (reqs == nullptr) {
        std::cerr << "MPI_Requests allocation failed!" << std::endl;
        return 1; // Exit program with error
    }

    MPI_Status* stats = (MPI_Status *) malloc(2 * r * sizeof(MPI_Status));
    if (stats == nullptr) {
        std::cerr << "MPI_Status allocation failed!" << std::endl;
        return 1; // Exit program with error
    }

	int metadata_send[nlpow];

    int comm_size[r-1];
	for (int x = 0; x < w; x++) {
		ze = (x == w - 1)? r - d: r;
		int zoffset = 0, zoffset_send = 0, zc = ze-1;
		int zns[zc];

		for (int k = 1; k < ze; k += b) {
			ss = ze - k < b ? ze - k : b;
			num_reqs = 0;

			for (int s = 0; s < ss; s++) {

				int z = k + s;

//				if (rank == 0) {
//					std::cout << k << " " << s << " " << z << std::endl;
//				}

				spoint = z * distance;
				nc = nprocs / next_distance * distance, rem = nprocs % next_distance - spoint;
				if (rem < 0) { rem = 0; }
				ns = (rem > distance)? (nc + distance) : (nc + rem);
				zns[z-1] = ns;

				int recvrank = (rank + spoint) % nprocs;
				int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process


				if (ns == 1) {

					MPI_Irecv(&recvbuf[rdispls[recvrank]*typesize], recvcounts[recvrank]*typesize, MPI_CHAR, recvrank, 1, comm, &reqs[num_reqs++]);

					MPI_Isend(&sendbuf[sdispls[sendrank]*typesize], sendcounts[sendrank]*typesize, MPI_CHAR, sendrank, 1, comm, &reqs[num_reqs++]);
				}
				else {
					di = 0;
					for (int i = spoint; i < nprocs; i += next_distance) {
						int j_end = (i+distance > nprocs)? nprocs: i+distance;
						for (int j = i; j < j_end; j++) {
							int id = (j + rank) % nprocs;
							sent_blocks[z-1][di++] = id;
						}
					}

					// 2) prepare metadata
					int sendCount = 0, offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];

						if (i % distance == 0) {
							metadata_send[i] = sendcounts[send_index];
						}
						else {
							metadata_send[i] = sendNcopy[sent_blocks[z-1][i]];
						}
						offset += metadata_send[i] * typesize;
					}

					MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv[z-1], di,
							MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

					for(int i = 0; i < di; i++) { sendCount += metadata_recv[z-1][i]; }
					comm_size[z-1] = sendCount; // total exchanged data per round

					// prepare send data
					offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int size = 0;

						if (i % distance == 0) {
							size = sendcounts[send_index]*typesize;
							memcpy(&temp_send_buffer[zoffset_send + offset], &sendbuf[sdispls[send_index]*typesize], size);
						}
						else {
							size = sendNcopy[sent_blocks[z-1][i]]*typesize;
							memcpy(&temp_send_buffer[zoffset_send + offset], &extra_buffer[sent_blocks[z-1][i]*max_send_count*typesize], size);
						}
						offset += size;
					}

					MPI_Irecv(&temp_recv_buffer[zoffset], comm_size[z-1]*typesize, MPI_CHAR, recvrank, recvrank+z, comm, &reqs[num_reqs++]);
					MPI_Isend(&temp_send_buffer[zoffset_send], offset, MPI_CHAR, sendrank, rank+z, comm, &reqs[num_reqs++]);

					zoffset += comm_size[z-1]*typesize;
					zoffset_send += offset;
				}

			}

			MPI_Waitall(num_reqs, reqs, stats);
			for (int i = 0; i < num_reqs; i++) {
			    if (stats[i].MPI_ERROR != MPI_SUCCESS) {
			        printf("Request %d encountered an error: %d\n", i, stats[i].MPI_ERROR);
			    }
			}


		}

		if (K < nprocs - 1) {
			// replaces
			int offset = 0;
			for (int i = 0; i < zc; i++) {
				for (int j = 0; j < zns[i]; j++){

					if (zns[i] > 1){
						int size = metadata_recv[i][j]*typesize;

						if (j < distance) {
							memcpy(&recvbuf[rdispls[sent_blocks[i][j]]*typesize], &temp_recv_buffer[offset], size);
						}
						else {
							memcpy(&extra_buffer[sent_blocks[i][j]*max_send_count*typesize], &temp_recv_buffer[offset], size);
							sendNcopy[sent_blocks[i][j]] = metadata_recv[i][j];
						}
						offset += size;
					}
				}
			}
		}

		distance *= r;
		next_distance *= r;

	}
	if (K < nprocs - 1) {
		free(extra_buffer);
		free(temp_recv_buffer);
		free(temp_send_buffer);
	}
	free(reqs);
	free(stats);

	return 0;
}

