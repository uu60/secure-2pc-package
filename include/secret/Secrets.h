//
// Created by 杜建璋 on 2025/2/12.
//

#ifndef SECRETS_H
#define SECRETS_H
#include <vector>

#include "./item/ArithSecret.h"
#include "./item/BoolSecret.h"


/**
 * A utility method class for secrets.
 */
class Secrets {
public:
    static void sort(std::vector<ArithSecret> &secrets, bool asc, int taskTag);

    static void sort(std::vector<BoolSecret> &secrets, bool asc, int taskTag);
};


#endif //SECRETS_H
