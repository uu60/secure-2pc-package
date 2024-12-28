//
// Created by 杜建璋 on 2024/12/1.
//

#include "compute/arith/ArithToBoolExecutor.h"

#include "comm/IComm.h"
#include "compute/bool/BoolAndExecutor.h"
#include "intermediate/IntermediateDataSupport.h"
#include "utils/Log.h"
#include "utils/Math.h"

ArithToBoolExecutor *ArithToBoolExecutor::execute() {
    _currentMsgTag = _startMsgTag;
    if (IComm::impl->isServer()) {
        // bitwise separate xi
        // xi is xored into xi_i and xi_o
        int64_t xi_i = Math::randInt();
        int64_t xi_o = xi_i ^ _xi;
        bool carry_i = false;

        for (int i = 0; i < _l; i++) {
            bool ai, ao, bi, bo;
            bool *self_i = IComm::impl->rank() == 0 ? &ai : &bi;
            bool *self_o = IComm::impl->rank() == 0 ? &ao : &bo;
            bool *other_i = IComm::impl->rank() == 0 ? &bi : &ai;
            *self_i = (xi_i >> i) & 1;
            *self_o = (xi_o >> i) & 1;
            IComm::impl->serverExchange(self_o, other_i, buildTag(_currentMsgTag));
            this->_zi += ((ai ^ bi) ^ carry_i) << i;

            // Compute carry
            if (i < _l - 1) {
                bool propagate_i = ai ^ bi;

                auto vec0 = _bmts == nullptr ? IntermediateDataSupport::pollBmts(1) : std::vector<Bmt>{(*_bmts)[i * 2]};
                auto vec1 = _bmts == nullptr
                                ? IntermediateDataSupport::pollBmts(1)
                                : std::vector<Bmt>{(*_bmts)[i * 2 + 1]};

                auto f0 = System::_threadPool.push([ai, bi, this, &vec0](int _) {
                    return BoolAndExecutor(ai, bi, 1, _objTag, _currentMsgTag, -1).setBmts(&vec0)->execute()->_zi;
                });
                auto f1 = System::_threadPool.push([propagate_i, carry_i, this, &vec1](int _) {
                    return BoolAndExecutor(propagate_i, carry_i, 1, _objTag,
                                           static_cast<int16_t>(_currentMsgTag + BoolAndExecutor::neededMsgTags(1)),
                                           -1).setBmts(&vec1)->execute()->_zi;
                });
                bool generate_i = f0.get();
                bool tempCarry_i = f1.get();
                bool sum_i = generate_i ^ tempCarry_i;
                bool and_i = BoolAndExecutor(generate_i, tempCarry_i, 1, _objTag, _currentMsgTag, -1).execute()->_zi;

                carry_i = sum_i ^ and_i;
            }
        }

        _zi = ring(_zi);
    }

    return this;
}

std::string ArithToBoolExecutor::className() const {
    return "ArithToBoolExecutor";
}

int16_t ArithToBoolExecutor::neededMsgTags() {
    return static_cast<int16_t>(2 * BoolAndExecutor::neededMsgTags(1));
}

ArithToBoolExecutor *ArithToBoolExecutor::setBmts(std::vector<Bmt> *bmts) {
    _bmts = bmts;
    return this;
}
