#ifdef GAME_RC3

#ifndef RC3_H
#define RC3_H

#include <lib/shk.h>
#include <lib/types.h>

#include "Moby.h"
#include "nwPlayerData.h"

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
extern int load_level;

extern int game_state;
extern int target_game_state;

extern Moby *hero_moby;

extern CollOutput coll_output;

extern int is_local_multiplayer;
extern int multiplayer_level;
extern int num_local_players;
extern nwPlayerData* team_data;
extern int lobby_mode;

SHK_FUNCTION_DEFINE_STATIC_5(0x956c0, int, coll_line, Vec4*, position1, Vec4*, position2, int, flags, Moby*, moby, Vec4*, unk_vec);
SHK_FUNCTION_DEFINE_STATIC_1(0x312380, void, nwConnect, void*, netConnectionInfo);
SHK_FUNCTION_DEFINE_STATIC_1(0x31ade0, void, nwSetGameSetupFlagsForGameType, int, gameType);
SHK_FUNCTION_DEFINE_STATIC_0(0x3127c0, void, nwJoin);
SHK_FUNCTION_DEFINE_STATIC_0(0x313c7c, void, nwLeaveGame);
SHK_FUNCTION_DEFINE_STATIC_0(0x318370, void, nwResetLobby);

SHK_FUNCTION_DEFINE_STATIC_1(0x31813c, void, nwSetInGame, int, inGame);
SHK_FUNCTION_DEFINE_STATIC_2(0x14aa54, void, LoadLevel, int, level, int, something);

struct tNW_GameSettings {
    char playerNames[8][32];
    char clanTags[8][16];
    short playerSkins[8];
    short playerTeams[8];
    short playerClients[8]; /* Created by retype action */
    short playerState[8];
    short playerTypes[8];
    float playerRank[8];
    float playerRankDeviation[8];
    int accountIds[8];
    int gameStartTime;
    int gameLoadStartTime;
    int numPlayers;
    short level; /* Created by retype action */
    u16 unk1;
    int gameType;
    int unk2;
    bool vehiclesAllowed;
    byte unk3[6];
    byte nodesOn;
    int whatthis;
    byte baseDefensesOn;
    byte hmmha;
    char altGameType; /* Attrition/Chaos/etc */
    byte fragLimit;
    byte ctfCap;
    byte unk4;
    bool startWithChargeboots;
    bool unlimitedAmmo;
    bool startWithAllWeapons;
    bool shizzolate;
    bool voiceEnabled;
    bool showPlayerNames;
    int netObjectIndex;
};



extern tNW_GameSettings* game_settings;

extern void* player1_controller_ptr;
extern void* player2_controller_ptr;
extern void* player3_controller_ptr;
extern void* player4_controller_ptr;

extern Moby* moby_ptr;
extern Moby* moby_ptr_end;

};

#endif // RC3_H
#endif // GAME_RC3