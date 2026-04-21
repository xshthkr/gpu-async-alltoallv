2 phases
- intra-node
- inter-node

The first phase uses ParLogNa (blocking, but negligible cost to communicate with processes within the same node. Bottleneck: latency-bound. Few processess, fast transfers.

The second phase uses lienar-time scattered algorithm (non-blocking) with a `batch_size` param that blocks between batches.

"ParLinNa blocks for all requests in one batch to be completed (in a non-blocking way) before moving on to the next batch"

ParLinNa uses non-blocking calls within a batch but synchronizes between batches using `MPI_Waitall` before starting the next batch.

`batch_size` essentially controls network congestion. When CPU waits for `MPI_Waitall`, it sits idle. We can overlap computation (packing of data for next batch) with the `MPI_Waitall` that completes the current batch.

MPI library must itself be able to make progress on the non-blocking sends in the background, perhaps with a child process or thread, while the current packing parent process or thread is running. We need to explicitly tell MPI library to do the communicaiton work on a separate progress thread or core. Or a simple polling loop with `MPI_Test` to manually force progress. This could be CPU inefficient.

Double buffering system into ParLinNa's batching structure. Pre pack batch in one thread, communicate in another thread, swap buffers.

Progress thread. Communications Servlet for ParLinNa. Allow network comms to happen in the background while main application threads continue computing. Non-blocking MPI calls often dont make actual progress until a blocking function is called. The master thread is the main application thread. its job is to perform application's core computation. The servlet thread is thhe dedicated comms servlet. its job is to handle MPI calls. it will receive tasks from master thread and sit in a loop and notify the master thread with a flag or a semaphore when a comms task is complete. need to pin threads to specific cores (`sched_setaffinity()` or `numactl`).
 
proactive instad of reactive. parlinna is reactive as it calls `MPI_Wait` on a batch and blocks the GPU until it completes. multi threaded progress for parlinna will be proactive. comms thread runs continuously on a dedicated core, overlapping the comms of one batch with packing/computation of next batch. 
