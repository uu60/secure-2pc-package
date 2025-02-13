//
// Created by 杜建璋 on 2024/12/1.
//

#include "compute/single/arith/ArithToBoolExecutor.h"

#include "comm/Comm.h"
#include "compute/single/bool/BoolAndExecutor.h"
#include "intermediate/BitwiseBmtGenerator.h"
#include "intermediate/BmtGenerator.h"
#include "intermediate/IntermediateDataSupport.h"
#include "utils/Log.h"
#include "utils/Math.h"

void ArithToBoolExecutor::prepareBmts(BitwiseBmt &b0, BitwiseBmt &b1, BitwiseBmt &b2) const {
    if (_bmts != nullptr) {
        b0 = _bmts->at(0);
        b1 = _bmts->at(1);
        b2 = _bmts->at(2);
    } else if (Conf::BMT_BACKGROUND) {
        auto bs = IntermediateDataSupport::pollBitwiseBmts(3, _width);
        b0 = bs[0];
        b1 = bs[1];
        b2 = bs[2];
    } else {
        if (!Conf::INTRA_OPERATOR_PARALLELISM) {
            b0 = BitwiseBmtGenerator(_width, _taskTag, _currentMsgTag).execute()->_bmt;
            b1 = BitwiseBmtGenerator(_width, _taskTag, _currentMsgTag).execute()->_bmt;
            b2 = BitwiseBmtGenerator(_width, _taskTag, _currentMsgTag).execute()->_bmt;
        } else {
            auto f0 = System::_threadPool.push([&](int) {
                return BitwiseBmtGenerator(_width, _taskTag, _currentMsgTag).execute()->_bmt;
            });
            auto f1 = System::_threadPool.push([&](int) {
                return BitwiseBmtGenerator(_width, _taskTag,
                                           static_cast<int16_t>(
                                               _currentMsgTag + BitwiseBmtGenerator::msgTagCount(1))).execute()->_bmt;
            });
            b2 = BitwiseBmtGenerator(_width, _taskTag,
                                     static_cast<int16_t>(
                                         _currentMsgTag + 2 * BitwiseBmtGenerator::msgTagCount(1))).execute()->_bmt;
            b0 = f0.get();
            b1 = f1.get();
        }
    }
}

ArithToBoolExecutor *ArithToBoolExecutor::execute() {
    _currentMsgTag = _startMsgTag;
    if (Comm::isServer()) {
        // bitwise separate xi
        // xi is xored into xi_i and xi_o
        int64_t xi_i = Math::randInt();
        int64_t xi_o = xi_i ^ _xi;
        bool carry_i = false;

        BitwiseBmt b0, b1, b2;
        prepareBmts(b0, b1, b2);

        for (int i = 0; i < _width; i++) {
            bool ai, ao, bi, bo;
            bool *self_i = Comm::rank() == 0 ? &ai : &bi;
            bool *self_o = Comm::rank() == 0 ? &ao : &bo;
            bool *other_i = Comm::rank() == 0 ? &bi : &ai;
            *self_i = (xi_i >> i) & 1;
            *self_o = (xi_o >> i) & 1;
            std::vector self_ov = {static_cast<int64_t>(*self_o)};
            std::vector<int64_t> other_iv;
            Comm::serverSend(self_ov, _width, buildTag(_currentMsgTag));
            Comm::serverReceive(other_iv, _width, buildTag(_currentMsgTag));
            *other_i = other_iv[0];
            this->_zi += static_cast<int64_t>((ai ^ bi) ^ carry_i) << i;

            // Compute carry
            if (i < _width - 1) {
                bool propagate_i = ai ^ bi;

                int16_t cm = _currentMsgTag;
                std::future<int64_t> f;
                bool generate_i;

                if (Conf::INTRA_OPERATOR_PARALLELISM) {
                    f = System::_threadPool.push([&](int) {
                        auto bmt = b0.extract(i);
                        return BoolAndExecutor(ai, bi, 1, _taskTag, cm, NO_CLIENT_COMPUTE).setBmt(
                            &bmt)->execute()->_zi;
                    });
                } else {
                    auto bmt = b0.extract(i);
                    generate_i = BoolAndExecutor(ai, bi, 1, _taskTag, cm, NO_CLIENT_COMPUTE).setBmt(
                        &bmt)->execute()->_zi;
                }

                _currentMsgTag += BoolAndExecutor::msgTagCount(1);

                auto bmt = b1.extract(i);
                bool tempCarry_i = BoolAndExecutor(propagate_i, carry_i, 1, _taskTag, _currentMsgTag, -1).setBmt(
                    &bmt)->execute()->_zi;

                if (Conf::INTRA_OPERATOR_PARALLELISM) {
                    generate_i = f.get();
                }
                bool sum_i = generate_i ^ tempCarry_i;

                bmt = b2.extract(i);
                bool and_i = BoolAndExecutor(generate_i, tempCarry_i, 1, _taskTag, _currentMsgTag, NO_CLIENT_COMPUTE).
                        setBmt(&bmt)->execute()->_zi;

                carry_i = sum_i ^ and_i;
            }
        }
        _zi = ring(_zi);
    }

    return this;
}

int16_t ArithToBoolExecutor::msgTagCount(int l) {
    return static_cast<int16_t>(2 * BoolAndExecutor::msgTagCount(l));
}

ArithToBoolExecutor * ArithToBoolExecutor::setBmts(std::vector<BitwiseBmt> *bmts) {
    if (bmts != nullptr && bmts->size() != bmtCount(_width)) {
        throw std::runtime_error("Bmt size mismatch.");
    }
    _bmts = bmts;
    return this;
}

ArithToBoolExecutor *ArithToBoolExecutor::reconstruct(int clientRank) {
    _currentMsgTag = _startMsgTag;
    BoolExecutor e(_zi, _width, _taskTag, _currentMsgTag, NO_CLIENT_COMPUTE);
    e.reconstruct(clientRank);
    if (Comm::rank() == clientRank) {
        _result = e._result;
    }
    return this;
}

int ArithToBoolExecutor::bmtCount(int width) {
    return 3;
}