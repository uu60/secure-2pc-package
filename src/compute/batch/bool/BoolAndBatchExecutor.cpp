//
// Created by 杜建璋 on 2025/2/24.
//

#include "compute/batch/bool/BoolAndBatchExecutor.h"

#include "conf/Conf.h"
#include "intermediate/BitwiseBmtBatchGenerator.h"
#include "intermediate/BitwiseBmtGenerator.h"
#include "intermediate/IntermediateDataSupport.h"

BoolAndBatchExecutor *BoolAndBatchExecutor::execute() {
    _currentMsgTag = _startMsgTag;

    if (Comm::isClient()) {
        return this;
    }

    int64_t start;
    if (Conf::CLASS_WISE_TIMING) {
        start = System::currentTimeMillis();
    }

    std::vector<BitwiseBmt> bmts;
    int num = static_cast<int>(_xis.size());
    int totalBits = num * _width;
    int bc = -1;
    if (totalBits > 64) {
        // ceil division
        bc = (num * _width + 63) / 64;
    }
    if (_bmts != nullptr) {
        bmts = std::move(*_bmts);
    } else if (Conf::BMT_METHOD == Consts::BMT_BACKGROUND) {
        if (bc == -1) {
            bmts = IntermediateDataSupport::pollBitwiseBmts(1, totalBits);
        } else {
            bmts = IntermediateDataSupport::pollBitwiseBmts(bc, 64);
        }
    } else if (Conf::BMT_METHOD == Consts::BMT_JIT) {
        if (bc == -1) {
            bmts = {BitwiseBmtGenerator(totalBits, _taskTag, _currentMsgTag).execute()->_bmt};
        } else {
            bmts = BitwiseBmtBatchGenerator(bc, 64, _taskTag, _currentMsgTag).execute()->_bmts;
        }
    }

    // The first num elements are ei, and the left num elements are fi
    std::vector<int64_t> efi(num * 2);

    int bmtBitIdx = 0;
    for (int i = 0; i < num; i++) {
        if constexpr (Conf::BMT_METHOD == Consts::BMT_FIXED) {
            efi[i] = _xis[i] ^ IntermediateDataSupport::_fixedBitwiseBmt._a;
            efi[num + i] = _yis[i] ^ IntermediateDataSupport::_fixedBitwiseBmt._b;
        } else {
            if (bc == -1) {
                // Only one bmt with totalBits
                efi[i] = _xis[i] ^ bmts[0]._a;
                efi[num + i] = _yis[i] ^ bmts[0]._b;
            } else {
                // Multiple 64 bit bmts
                uint64_t mask = (1ull << _width) - 1;
                auto &bmt = bmts[i * _width / 64];
                int offset = i % 64 * _width;
                efi[i] = _xis[i] & (((bmt._a) & (mask << offset)) >> offset);
                efi[num + i] = _yis[i] & (((bmt._b) & (mask << offset)) >> offset);
            }
        }
    }

    std::vector<int64_t> efo;
    Comm::serverSend(efi, _width, buildTag(_currentMsgTag));
    Comm::serverReceive(efo, _width, buildTag(_currentMsgTag));

    std::vector<int64_t> efs(num * 2);
    for (int i = 0; i < num; i++) {
        efs[i] = efi[i] ^ efo[i];
        efs[num + i] = efi[num + i] ^ efo[num + i];
    }

    _zis.resize(num);
    int64_t extendedRank = Comm::rank() ? ring(-1ll) : 0;
    for (int i = 0; i < num; i++) {
        int64_t e = efi[i];
        int64_t f = efi[num + i];
        if constexpr (Conf::BMT_METHOD == Consts::BMT_FIXED) {
            _zis[i] = (extendedRank & e & f) ^ (
                          f & IntermediateDataSupport::_fixedBitwiseBmt._a) ^ (
                          e & IntermediateDataSupport::_fixedBitwiseBmt._b) ^
                      IntermediateDataSupport::_fixedBitwiseBmt._c;
        } else {
            if (bc == -1) {
                _zis[i] = (extendedRank & e & f) ^ (f & bmts[0]._a) ^ (e & bmts[0]._b) ^ bmts[0]._c;
            } else {
                // Multiple 64 bit bmts
                int64_t a, b, c;
                if (_width < 64) {
                    int64_t mask = (1ll << _width) - 1;
                    auto &bmt = bmts[i * _width / 64];
                    int offset = i % 64 * _width;
                    a = (bmt._a & (mask << offset)) >> offset;
                    b = (bmt._b & (mask << offset)) >> offset;
                    c = (bmt._c & (mask << offset)) >> offset;
                } else {
                    a = bmts[i]._a;
                    b = bmts[i]._b;
                    c = bmts[i]._c;
                }
                _zis[i] = (extendedRank & e & f) ^ (f & a) ^ (e & b) ^ c;
            }
        }
    }

    if (Conf::CLASS_WISE_TIMING) {
        _totalTime += System::currentTimeMillis() - start;
    }

    return this;
}

int BoolAndBatchExecutor::msgTagCount(int num, int width) {
    return BitwiseBmtBatchGenerator::msgTagCount(bmtCount(num), width);
}

BoolAndBatchExecutor *BoolAndBatchExecutor::setBmts(std::vector<BitwiseBmt> *bmts) {
    _bmts = bmts;
    return this;
}

int BoolAndBatchExecutor::bmtCount(int num) {
    return num;
}
