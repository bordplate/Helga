//
// Created by bordplate on 6/23/2025.
//

#ifndef RC3_NWPLAYERDATA_H
#define RC3_NWPLAYERDATA_H

#include "Moby.h"
#include "../lib/types.h"

struct nwPlayerData {
	/* 0x0    */ char pad000[0x1a0];
	/* 0x1a0  */ float speed;
	/* 0x1a4  */ char pad00[0x1840];
	/* 0x19e4 */ int state;
	/* 0x19e8 */ char pad0000[0xac8];
	/* 0x24b0 */ struct Vec4 camera_forward;
    /* 0x2564 */ char pad0[0xa8];
    /* 0x2608 */ float health;
    /* 0x260c */ char pad1[0xac];
    /* 0x26b8 */ struct Moby *player_moby;
    /* 0x26bc */ char pad2[0x20];
    /* 0x26dc */ u8 team_id;
    /* 0x26dd */ char pad3[0x2043];
};

#endif //RC3_NWPLAYERDATA_H
