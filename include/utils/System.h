//
// Created by 杜建璋 on 2024/7/26.
//

#ifndef MPC_PACKAGE_SYSTEM_H
#define MPC_PACKAGE_SYSTEM_H
#include "../parallel/ctpl_stl.h"
#include "../comm/Comm.h"
#include "../conf/Conf.h"

class System {
private:
    /**
     * 0 is preserved for BmtGenerator.
     * 1 is preserved for BitwiseBmtGenerator.
     * 2 is preserved for BaseOtExecutor. (No need for preservation)
     * 3 is preserved for ABPairGenerator. (Abort)
     */
    inline static int PRESERVED_TASK_TAGS = 0;
    inline static std::atomic_int _currentTaskTag = PRESERVED_TASK_TAGS;

public:
    inline static std::atomic_bool _shutdown = false;

public:
    static void init(int argc, char **argv);

    static void finalize();

    static int nextTask();

    static int64_t currentTimeMillis();
};


#endif //MPC_PACKAGE_SYSTEM_H
