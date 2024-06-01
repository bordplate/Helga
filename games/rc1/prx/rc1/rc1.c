#include "rc1.h"
#include "common.h"

#include <lib/memory.h>
#include <netex/net.h>
#include <sysutil/sysutil_gamecontent.h>

#include <cell/cell_fs.h>
#include <cell/pad.h>

#include <rc1/Game.h>

#include "bridging.h"

extern "C" {
void game_tick() {
    _c_game_tick();
}

SHK_HOOK(void, game_loop_start);
void game_loop_start_hook() {
    if (current_planet != 0 || ratchet_moby != 0) {
        game_tick();
    }

    SHK_CALL_HOOK(game_loop_start);
}

SHK_HOOK(void, game_loop_intro_start);
void game_loop_intro_start_hook() {
    game_tick();

    SHK_CALL_HOOK(game_loop_intro_start);
}

SHK_HOOK(void, on_respawn);
void on_respawn_hook() {
    SHK_CALL_HOOK(on_respawn);
    _c_on_respawn();
}

// Hook to avoid some consoles getting a "game is corrupted, restart the game" on game start
// I think maybe it makes trophies not work?
SHK_HOOK(void, authenticate_game);
void authenticate_game_hook() {
    MULTI_LOG("Game totally authenticated\n");
}

SHK_HOOK(void, FUN_000784e8);
void FUN_000784e8_hook() {
    _c_game_render();

    SHK_CALL_HOOK(FUN_000784e8);
}

SHK_HOOK(void, wrench_update_func, Moby *);
void wrench_update_func_hook(Moby *moby) {
    // Clear the collision out ptr before calling original wrench function
    coll_moby_out = 0;

    SHK_CALL_HOOK(wrench_update_func, moby);

    // If coll_moby_out has a value, the wrench has "attacked" something
    if (!coll_moby_out) {
        return;
    }

    // Figure out what moby we hit and if we need to tell the server about it
    Moby* hit = coll_moby_out;
    if (!hit->pVars) {
        // If we don't have pVars, this isn't something the server needs to know about
        return;
    }

    MPMobyVars* vars = (MPMobyVars*)hit->pVars;

    // If this moby has UUID vars
    if (vars->uuid && vars->sig == 0x4542) {

    }
}

SHK_HOOK(int, cellGameBootCheck, unsigned int*, unsigned int*, CellGameContentSize*, char*);
int cellGameBootCheckHook(unsigned int* type, unsigned int* attributes, CellGameContentSize* size, char* dirName) {
    MULTI_LOG("Type: %p, attr: %p, size: %p, dirName: %p\n", type, attributes, size, dirName);

    *type = 2;
    *attributes = 0;
    size->hddFreeSizeKB = 10000;
    size->sizeKB = -1;
    size->sysSizeKB = 4;

    int fd;
    const char* src;
    // Manually copying the string
    // Check if digital version exists and use that. Otherwise fall back to disc. If no disc then we just crash
    CellFsErrno ebootStat = cellFsOpendir("/dev_hdd0/game/NPEA00385/", &fd);
    if (ebootStat == CELL_FS_ENOENT) {
        src = "BCES01503";
    } else {
        src = "NPEA00385";
    }
    while (*src) {
        *dirName = *src;
        dirName++;
        src++;
    }
    *dirName = '\0';  // Null terminate the string

    MULTI_LOG("Done the thing\n");

    return 0;
}

SHK_HOOK(int, cellGameContentPermit, char*, char*);
int cellGameContentPermitHook(char* contentInfoPath, char* usrdirPath) {
    MULTI_LOG("contentInfoPath: %p, usrdirPath: %p\n", contentInfoPath, usrdirPath);

    int fd;
    const char *src;

    // Check if digital version exists and use that. Otherwise fall back to disc. If no disc then we just crash
    CellFsErrno ebootStat = cellFsOpendir("/dev_hdd0/game/NPEA00385/", &fd);
    if (ebootStat == CELL_FS_ENOENT) {
        src = "/dev_bdvd/PS3_GAME";
    } else {
        src = "/dev_hdd0/game/NPEA00385";
    }
    while (*src) {
        *contentInfoPath = *src;
        contentInfoPath++;
        src++;
    }
    *contentInfoPath = '\0';  // Null terminate the string

    if (ebootStat == CELL_FS_ENOENT) {
        src = "/dev_bdvd/PS3_GAME/USRDIR";
    } else {
        src = "/dev_hdd0/game/NPEA00385/USRDIR";
    }
    while (*src) {
        *usrdirPath = *src;
        usrdirPath++;
        src++;
    }
    *usrdirPath = '\0';  // Null terminate the string

    MULTI_LOG("Done the thing\n");

    return 0;
}

#include "GoldBolt.h"

SHK_HOOK(void, goldBoltUpdate, Moby* moby);
void goldBoltUpdateHook(Moby* moby) {
    ((GoldBolt*)moby)->update();
}

// Hook the item_unlock function
SHK_HOOK(void, _unlock_item, int, uint8_t);
void _unlock_item_hook(int item_id, uint8_t equip) {

}

// Make original unlock_item available to our code
void unlock_item(int item_id, uint8_t equip) {
    SHK_CALL_HOOK(_unlock_item, item_id, equip);
}

SHK_HOOK(void, _unlock_level, int);
void _unlock_level_hook(int level) {

}

void unlock_level(int level) {
    SHK_CALL_HOOK(_unlock_level, level);
}

SHK_HOOK(void, some_rendering_func);
void some_rendering_func_hook() {
    if (should_render == 0) {
        return;
    }

    SHK_CALL_HOOK(some_rendering_func);
}

#define remote_pressed_buttons *((int*)0xB00008)
#define last_remote_pressed_buttons *((int*)0xB0000C)
#define remote_joysticks *((int*)0xB00010)

SHK_HOOK(int32_t, cellPadGetDataRedirect, uint32_t, CellPadData*);
int32_t cellPadGetDataRedirectHook(uint32_t port_no, CellPadData *data) {
    int32_t ret = cellPadGetData(port_no, data);

    int32_t len = data->len;

    memset(data, 0, 16);

    data->len = len;

//    if (data->len != 0) {
//        MULTI_LOG("Port_no: %d; Data len: %d. inputs: %.4x. Ret: %d\n", port_no, data->len,
//                  (data->button[2] << 8) + data->button[3], ret);
//    }

    if (current_planet != 0) {
        if (data->len == 0 && (remote_pressed_buttons != last_remote_pressed_buttons)) {
            data->len = 24;
        }

        data->button[2] |= (remote_pressed_buttons & 0xff00) >> 8;
        data->button[3] |= remote_pressed_buttons & 0x00ff;
        data->button[4] = (remote_joysticks & 0x000000ff);
        data->button[5] = (remote_joysticks & 0x0000ff00) >> 8;
        data->button[6] = (remote_joysticks & 0x00ff0000) >> 16;
        data->button[7] = (remote_joysticks & 0xff000000) >> 24;
    }

    last_remote_pressed_buttons = remote_pressed_buttons;

    return ret;
}

void rc1_init() {
    MULTI_LOG("Multiplayer initializing.\n");

    init_memory_allocator(memory_area, sizeof(memory_area));

    MULTI_LOG("Initialized memory allocator. Binding hooks\n");

    SHK_BIND_HOOK(game_loop_start, game_loop_start_hook);
    SHK_BIND_HOOK(game_loop_intro_start, game_loop_intro_start_hook);
    SHK_BIND_HOOK(wrench_update_func, wrench_update_func_hook);
    SHK_BIND_HOOK(authenticate_game, authenticate_game_hook);
    SHK_BIND_HOOK(FUN_000784e8, FUN_000784e8_hook);
    SHK_BIND_HOOK(on_respawn, on_respawn_hook);
    SHK_BIND_HOOK(cellGameBootCheck, cellGameBootCheckHook);
    SHK_BIND_HOOK(cellGameContentPermit, cellGameContentPermitHook);
    SHK_BIND_HOOK(goldBoltUpdate, goldBoltUpdateHook);
    SHK_BIND_HOOK(_unlock_item, _unlock_item_hook);
    SHK_BIND_HOOK(_unlock_level, _unlock_level_hook);
    SHK_BIND_HOOK(cellPadGetDataRedirect, cellPadGetDataRedirectHook);
    SHK_BIND_HOOK(some_rendering_func, some_rendering_func_hook);

    MULTI_LOG("Bound hooks\n");
}

void rc1_shutdown() {
    sys_net_finalize_network();
}

};