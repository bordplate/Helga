#ifdef GAME_RC3

#ifndef RC3_H
#define RC3_H

#include <lib/shk.h>
#include <lib/types.h>

#include "Moby.h"

extern "C" {

typedef enum GameState {
    PlayerControl = 0,
    Movie = 1,
    CutScene = 2,
    Menu = 3,
    ExitRace = 4,
    Gadgetron = 5,
    PlanetLoading = 6,
    CinematicMaybe = 7,
    UnkFF = 255
} GameState;

//void unlock_item(ITEM item,bool equipped)
enum CONTROLLER_INPUT {
    L2 = 1,
    R2 = 2,
    L1 = 4,
    R1 = 8,
    Triangle = 16,
    Circle = 32,
    Cross = 64,
    Square = 128,
    Select = 256,
    L3 = 512,
    R3 = 1024,
    Start = 2048,
    Up = 4096,
    Right = 8192,
    Down = 16384,
    Left = 32768
};

struct CollOutput {
    void* grid;
    float pad1;
    float pad2;
    float pad3;
    int count;
    int damage_next;
    Moby* pMoby;
    int poly;
    Vec4 ip;
    Vec4 push;
    Vec4 normal;
    Vec4 v0;
    Vec4 v1;
    Vec4 v2;
};

void rc3_init();
void rc3_shutdown();

extern int current_level;
extern int destination_level;

extern int game_state;

extern Moby *hero_moby;

extern CollOutput coll_output;

SHK_FUNCTION_DEFINE_STATIC_5(0x956c0, int, coll_line, Vec4*, position1, Vec4*, position2, int, flags, Moby*, moby, Vec4*, unk_vec);

};

#endif // RC3_H
#endif // GAME_RC3