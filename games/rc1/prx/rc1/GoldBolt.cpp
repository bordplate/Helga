//
// Created by bordplate on 7/20/2023.
//

#include "GoldBolt.h"
#include "Game.h"

#include <lib/shk.h>

extern "C" SHK_HOOK(void, goldBoltUpdate, Moby* moby);
void GoldBolt::update() {
    GoldBoltVars* vars = (GoldBoltVars*)this->pVars;

    SHK_CALL_HOOK(goldBoltUpdate, this);
}