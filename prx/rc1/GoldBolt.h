//
// Created by bordplate on 7/20/2023.
//

#ifndef RAC1_MULTIPLAYER_GOLDBOLT_H
#define RAC1_MULTIPLAYER_GOLDBOLT_H

#include "moby.h"

struct GoldBoltVars {
    int number;
};

struct GoldBolt : public Moby {
public:
    void update();
};


#endif //RAC1_MULTIPLAYER_GOLDBOLT_H
