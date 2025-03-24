//
// Created by 杜建璋 on 2025/1/19.
//

#ifndef CONF_H
#define CONF_H
#include <thread>

class Conf {
public:
    // pool type
    enum PoolT {
        CTPL_POOL,
        TBB_POOL
    };

    // blocking queue type
    enum QueueT {
        CAS_QUEUE,
        LOCK_QUEUE
    };

    // comm type
    enum CommT {
        MPI
    };

    // bmt generation
    enum BmtT {
        BMT_BACKGROUND,
        BMT_JIT,
        BMT_FIXED,
        BMT_BATCH_BACKGROUND
     };
public:
    // ---------------Settings for bmts---------------
    // If intermediate data produced in background (DO NOT FORGET TO ENABLE MULTI-THREAD)
    constexpr static BmtT BMT_METHOD = BMT_BACKGROUND;
    // Bmt max num in queue (INVALID when BMT_BACKGROUND is false)
    constexpr static int MAX_BMTS = 10000;
    // Used times limit of one bmt (INVALID when BMT_BACKGROUND is false)
    constexpr static int BMT_USAGE_LIMIT = 1;
    // Blocking Bmt Queue (INVALID when background bmt is disabled)
    constexpr static QueueT BMT_QUEUE_TYPE = CAS_QUEUE;

    // ---------------Settings for threads---------------
    // Task tag bits
    constexpr static int TASK_TAG_BITS = 3;
    // Enable single-thread only
    constexpr static bool DISABLE_MULTI_THREAD = false;
    // Enable multiple-thread computation in each single executor
    constexpr static bool INTRA_OPERATOR_PARALLELISM = false;
    // Sum of threads in a process
    inline static int LOCAL_THREADS = static_cast<int>(std::thread::hardware_concurrency() * 10);
    // Index of thread pool type (0 = ctpl, 1 = tbb)
    constexpr static int THREAD_POOL_TYPE = CTPL_POOL;
    // Thread pool task queue separation
    constexpr static bool JOB_QUEUE_SEPARATION = false;
    // Thread pool abort policy is CallerRunsPolicy
    constexpr static bool CALLER_RUNS_POLICY = true;

    // ---------------Settings for networks---------------
    // Communication object index (0 = OpenMpi)
    constexpr static CommT COMM_TYPE = MPI;
    // Batch communicate or execute by elements
    constexpr static bool TASK_BATCHING = true;
    // Invalid if intra parallelism or batching is false
    constexpr static int BATCH_SIZE = 10;
    // Transfer compression
    constexpr static bool ENABLE_TRANSFER_COMPRESSION = false;

    // ---------------Settings for benchmark---------------
    constexpr static bool CLASS_WISE_TIMING = false;

    // ---------------Settings for sort---------------
    // Sort in parallel
    constexpr static bool SORT_IN_PARALLEL = false;

    // ---------------Settings for acceleration---------------
    constexpr static bool ENABLE_SIMD = true;
};



#endif //CONF_H
