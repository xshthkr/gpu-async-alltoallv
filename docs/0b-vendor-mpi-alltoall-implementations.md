`MPI_Alltoall` is a collective comms operations where each process sends distinct data to evey process. Data type and count of data items is the same. P processses each send P messages, one to each rank, and received P messages (1 message is sent to self). 

MPI vendors (OpenMPI, MPICH, Intel MPI, etc) use multiple algorithms to implement this comm operation. Performance is heavily dependant on number of processes P and message size M. The algorithm type is dynamically chosen between linear or logarithmic implementaions.

Latency: fixed startup time to initiate message (software overhead + network propagation)
Bandwidth: time to transmit the data bits over the wire
Total time = (steps * latency) + (total data * bandwidth)

For small messages, latency dominates. We want to minimize the number of communication steps.

The larger messages, bandwidth and network contention dominate. We need to maximize link utilization and avoid congestion.

Bruck's algorithm is the logarithmic approach. Designed to minimize latency. Based on the concept or recursive doubling.
- Processes are arranged logically. in step k, a process exchanges data with parter at distance 2^k
- in every step the amount of data help that is "correctly placed" doubles or the distance of data exchanges doubles
- after logP (base 2) steps every process has received all required data from all other processes

Very few comms steps (logP vs P) and excellent for small messages where startup latency is the bottleneck. Network contention in early steps as many processes communicate over long logical distances. Can reuduce effective bandwidth. Requires complex data indexing and memory copying to rearrange data blocks logically before and after exchanges.

Pairwise exchange is the linear approach. Designed to maximize bandwitch and minimize contention.
- broken into P-1 phases
- in phase k evey process i exchanges data specifically with process i+k mod P
- each step every process sends exactly the chunk of data intended for that specific partnet and received the chunk that that partner intended for it
- after P-1 steps every process has exchanged data with every other process exactly once

The traffic is predictable as each process sends exactly one message each step. Low network contention due to simple communication pattern. Less complex memory manipulation compared to bruck's. But it has a high latency cost and required P-1 startup events. Terrible for small messages.

MPI libraries use a cross-over threshold to decide which algorithm to use at runtime. decision tree. based on message size per destination.
