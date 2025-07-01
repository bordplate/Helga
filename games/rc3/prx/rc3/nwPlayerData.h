//
// Created by bordplate on 6/23/2025.
//

#ifndef RC3_NWPLAYERDATA_H
#define RC3_NWPLAYERDATA_H

#include "Moby.h"
#include "../lib/types.h"

struct nwPlayerData {
    char pad0[0x2568];
    float health;
    char pad1[0xac];
    struct Moby *player_moby;
    char pad2[0x20];
    u8 team_id;
    char pad3[0x2043];
};

#endif //RC3_NWPLAYERDATA_H
