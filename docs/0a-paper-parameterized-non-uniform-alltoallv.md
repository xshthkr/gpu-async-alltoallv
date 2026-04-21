The paper "Parameterized Algorithms for Non-uniform All-to-all" addresses the `MPI_alltoallv` collective. The collective is used for a pattern of communication where each process sends data of potentially different sizes to every other process. The paper proposes 2 novel algorithms to optimize the algorithms implemented by MPI vendors in HPC.

The standard `MPI_alltoall` (*UNIFORM* alltoall) vendor implementations have linear and logarithmic time algorithms. *NON-UNIFORM* alltoall is a generalizaiton of uniform alltoall and the vendor MPI implementations often fall back to the slower linear algorithms.

2 key improvements that can be made to the non-uniform alltoall algorithm:
- They dont leverage faster logarithmic algorithsm that work well for some data sizes
- They ignore the hierarchy of the system (intra-node comms are faster than inter-node comms)

The authors introduced 2 parameterized algorithms that can be tuned to suit the system and problem.
- ParLogNa (parameterized logarithmic non-uniform alltoall) minimizes the number of communicaiotn steps (latency). This is significant for small message sizes.
- ParLinNa (parameterized linear non-uniform alltoall) optimizes bandwidth usage and distinguishes between comms that are intra-node and inter-node.

Tuning the algorithms is tuning the tradeoff between minimizing latency and maximizing bandwidth.


