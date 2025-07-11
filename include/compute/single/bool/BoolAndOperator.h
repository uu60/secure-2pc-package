//
// Created by 杜建璋 on 2024/11/12.
//

#ifndef INTANDEXECUTOR_H
#define INTANDEXECUTOR_H
#include "./BoolOperator.h"
#include "../../../intermediate/item/BitwiseBmt.h"

class BoolAndOperator : public BoolOperator {
private:
    // std::vector<Bmt> *_bmts{};
    BitwiseBmt* _bmt{};

public:
    inline static std::atomic_int64_t _totalTime = 0;

public:
    BoolAndOperator(int64_t x, int64_t y, int l, int taskTag, int msgTagOffset,
                       int clientRank) : BoolOperator(x, y, l, taskTag, msgTagOffset, clientRank) {
    }

    BoolAndOperator *execute() override;

    [[nodiscard]] static int tagStride(int width);

    BoolAndOperator *setBmt(BitwiseBmt *bmt);
};


#endif //INTANDEXECUTOR_H
