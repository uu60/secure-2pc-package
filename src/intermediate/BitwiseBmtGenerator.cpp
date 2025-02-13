//
// Created by 杜建璋 on 2025/2/1.
//

#include "../../include/intermediate/BitwiseBmtGenerator.h"

#include <openssl/asn1.h>

#include "../../include/ot/RandOtBatchExecutor.h"
#include "../../include/utils/Math.h"

BitwiseBmtGenerator *BitwiseBmtGenerator::execute() {
    _currentMsgTag = _startMsgTag;
    if (Comm::isServer()) {
        generateRandomAB();

        computeMix(0);
        computeMix(1);
        computeC();
    }
    return this;
}

void BitwiseBmtGenerator::generateRandomAB() {
    _bmt._a = ring(Math::randInt());
    _bmt._b = ring(Math::randInt());
}

void BitwiseBmtGenerator::computeMix(int sender) {
    // atomic integer needed for multiple-thread computation
    int64_t sum = 0;
    bool isSender = Comm::rank() == sender;

    // messages and choices are stored in int64_t
    std::vector<int64_t> ss0, ss1;
    std::vector<int> choices;

    if (isSender) {
        ss0.reserve(_width);
        ss1.reserve(_width);
        for (int i = 0; i < _width; ++i) {
            int64_t bit = Math::randInt(0, 1);
            ss0.push_back(bit);
            ss1.push_back(corr(i, bit));
        }
    } else {
        choices.reserve(_width);
        for (int i = 0; i < _width; ++i) {
            choices.push_back(Math::getBit(_bmt._b, i));
        }
    }

    RandOtBatchExecutor r(sender, &ss0, &ss1, &choices, _width, _taskTag, static_cast<int16_t>(
                              _currentMsgTag + sender * RandOtBatchExecutor::msgTagCount()));
    r.execute();

    if (isSender) {
        for (int i = 0; i < _width; ++i) {
            sum += ss0[i] << i;
        }
    } else {
        for (int i = 0; i < _width; ++i) {
            sum += r._results[i] << i;
        }
    }

    if (sender == 0) {
        _ui = ring(sum);
    } else {
        _vi = ring(sum);
    }
}

void BitwiseBmtGenerator::computeC() {
    _bmt._c = ring(_bmt._a & _bmt._b ^ _ui ^ _vi);
}

int64_t BitwiseBmtGenerator::corr(int i, int64_t x) const {
    return (Math::getBit(_bmt._a, i) - x) & 1;
}

AbstractSecureExecutor * BitwiseBmtGenerator::reconstruct(int clientRank) {
    throw std::runtime_error("Not support.");
}

int16_t BitwiseBmtGenerator::msgTagCount(int width) {
    return static_cast<int16_t>(2 * RandOtBatchExecutor::msgTagCount());
}
