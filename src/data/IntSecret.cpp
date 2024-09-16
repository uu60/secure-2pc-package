//
// Created by 杜建璋 on 2024/9/12.
//

#include "data/IntSecret.h"
#include "executor/share/IntShareExecutor.h"
#include "executor/share/arithmetic/multiplication/RsaOtMultiplicationShareExecutor.h"

template<typename T>
IntSecret<T>::IntSecret(T x) {
    _data = x;
}

template<typename T>
IntSecret<T> IntSecret<T>::add(T yi) const {
    return IntSecret(_data + yi);
}

template<typename T>
IntSecret<T> IntSecret<T>::add(T xi, T yi) {
    return IntSecret(xi + yi);
}

template<typename T>
IntSecret<T> IntSecret<T>::multiply(T yi) const {
    return IntSecret(RsaOtMultiplicationShareExecutor(_data, yi, false).execute(false)->result());
}

template<typename T>
IntSecret<T> IntSecret<T>::share() const {
    return IntSecret(IntShareExecutor(_data).xi());
}

template<typename T>
IntSecret<T> IntSecret<T>::reconstruct() const {
    return IntSecret(IntShareExecutor(0).zi(_data)->reconstruct()->result());
}

template<typename T>
IntSecret<T> IntSecret<T>::share(T x) {
    return IntSecret(IntShareExecutor(x).xi());
}

template<typename T>
IntSecret<T> IntSecret<T>::multiply(T xi, T yi) {
    return IntSecret(RsaOtMultiplicationShareExecutor(xi, yi, false).execute(false)->result());
}

template<typename T>
T IntSecret<T>::get() const {
    return _data;
}

template<typename T>
IntSecret<T> IntSecret<T>::sum(const std::vector<T> &xis) {
    IntSecret<T> ret(0);
    for (T x: xis) {
        ret = ret.add(x);
    }
    return ret;
}

template<typename T>
IntSecret<T> IntSecret<T>::add(IntSecret<T> yi) const {
    return add(yi.get());
}

template<typename T>
IntSecret<T> IntSecret<T>::multiply(IntSecret<T> yi) const {
    return multiply(yi.get());
}

template<typename T>
IntSecret<T> IntSecret<T>::sum(const std::vector<T> &xis, const std::vector<T> &yis) {
    IntSecret<T> ret(0);
    for (int i = 0; i < xis.size(); i++) {
        ret = ret.add(xis[i]).add(yis[i]);
    }
    return ret;
}

template<typename T>
IntSecret<T> IntSecret<T>::share(IntSecret<T> x) {
    return share(x.get());
}


template<typename T>
IntSecret<T> IntSecret<T>::add(IntSecret<T> xi, IntSecret<T> yi) {
    return add(xi.get(), yi.get());
}

template<typename T>
IntSecret<T> IntSecret<T>::multiply(IntSecret<T> xi, IntSecret<T> yi) {
    return multiply(xi.get(), yi.get());
}

template<typename T>
IntSecret<T> IntSecret<T>::sum(const std::vector<IntSecret<T>> &xis, bool dummy) {
    std::vector<T> temp;
    temp.reserve(xis.size());
    for (IntSecret<T> x: xis) {
        temp.push_back(x.get());
    }
    return sum(temp);
}

template<typename T>
IntSecret<T> IntSecret<T>::sum(const std::vector<IntSecret<T>> &xis, const std::vector<IntSecret<T>> &yis, bool dummy) {
    std::vector<T> tempX(xis.size());
    for (IntSecret<T> x: xis) {
        tempX.push_back(x.get());
    }
    std::vector<T> tempY(yis.size());
    for (IntSecret<T> y: yis) {
        tempY.push_back(y.get());
    }
    return sum(tempX, tempY);
}

template
class IntSecret<bool>;

template
class IntSecret<int8_t>;

template
class IntSecret<int16_t>;

template
class IntSecret<int32_t>;

template
class IntSecret<int64_t>;




