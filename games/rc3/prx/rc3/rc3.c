#include "rc3.h"
#include "common.h"

#include "bridging.h"

#include <sysutil/sysutil_gamecontent.h>

#include <cell/cell_fs.h>
#include <cell/pad.h>

extern "C" {

SHK_HOOK(int, cellGameBootCheck, unsigned int*, unsigned int*, CellGameContentSize*, char*);
int cellGameBootCheckHook(unsigned int* type, unsigned int* attributes, CellGameContentSize* size, char* dirName) {
    MULTI_LOG("Type: %p, attr: %p, size: %p, dirName: %p\n", type, attributes, size, dirName);

    *type = 2;
    *attributes = 0;
    size->hddFreeSizeKB = 100000;
    size->sizeKB = -1;
    size->sysSizeKB = 4;

    int fd;
    const char* src;
    // Manually copying the string
    // Check if digital version exists and use that. Otherwise fall back to disc. If no disc then we just crash
    CellFsErrno ebootStat = cellFsOpendir("/dev_hdd0/game/NPEA00387/", &fd);
    if (ebootStat == CELL_FS_ENOENT) {
        src = "BCES01503";
    } else {
        src = "NPEA00387";
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
    CellFsErrno ebootStat = cellFsOpendir("/dev_hdd0/game/NPEA00387/", &fd);
    if (ebootStat == CELL_FS_ENOENT) {
        src = "/dev_bdvd/PS3_GAME";
    } else {
        src = "/dev_hdd0/game/NPEA00387";
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
        src = "/dev_hdd0/game/NPEA00387/USRDIR";
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

SHK_HOOK(void, pre_game_loop, void);
void pre_game_loop_hook() {
    _c_game_tick();
}

#define headless *((int*)0x1B00200)

SHK_HOOK(void, render, int);
void render_hook(int a1) {
    if (headless == 0) {
        SHK_CALL_HOOK(render, 0);
    }
}

#define remote_pressed_buttons *((int*)0x1B00008)
#define last_remote_pressed_buttons *((int*)0x1B0000C)

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

    if (current_level > 0) {
        if (data->len == 0 && (remote_pressed_buttons != last_remote_pressed_buttons)) {
            data->len = 24;
        }

        data->button[2] |= (remote_pressed_buttons & 0xff00) >> 8;
        data->button[3] |= remote_pressed_buttons & 0x00ff;
        data->button[4] = 0x7f;
        data->button[5] = 0x7f;
        data->button[6] = 0x7f;
        data->button[7] = 0x7f;
    }

    last_remote_pressed_buttons = remote_pressed_buttons;

    return ret;
}

void rc3_init() {
    MULTI_LOG("Multiplayer initializing.\n");

    init_memory_allocator(memory_area, sizeof(memory_area));

    SHK_BIND_HOOK(cellGameBootCheck, cellGameBootCheckHook);
    SHK_BIND_HOOK(cellGameContentPermit, cellGameContentPermitHook);

    SHK_BIND_HOOK(pre_game_loop, pre_game_loop_hook);
    SHK_BIND_HOOK(render, render_hook);

    SHK_BIND_HOOK(cellPadGetDataRedirect, cellPadGetDataRedirectHook);

    MULTI_LOG("Initialized memory allocator. Binding hooks\n");

    MULTI_LOG("Bound hooks\n");
}

void rc3_shutdown() {

}

};