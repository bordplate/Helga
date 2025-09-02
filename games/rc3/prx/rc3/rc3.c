#include "rc3.h"
#include "common.h"

#include "bridging.h"

#include <sysutil/sysutil_gamecontent.h>
#include "Moby.h"

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

SHK_HOOK(void, STUB_008acc94);
void game_start() {
    _c_on_game_start();
}

SHK_HOOK(void, pre_game_loop, void);
void pre_game_loop_hook() {
    _c_game_tick();
}

SHK_HOOK(void, update_mobys_func);
void update_mobys_func_hook() {
    SHK_CALL_HOOK(update_mobys_func);

    _c_game_tick();
}

SHK_HOOK(Moby*, spawn_moby, int, int);
Moby* spawn_moby_hook(int o_class, int unk2) {
    SHK_CALL_HOOK(spawn_moby, o_class, unk2);
}

#define last_state *((int*)0xcc518c)

#define blue_flag_position *((Vec3*)0xcc5190)

SHK_HOOK(void, ctf_flag_update_func, Moby*);
void ctf_flag_update_func_hook(Moby* moby) {
    SHK_CALL_HOOK(ctf_flag_update_func, moby);

    _c_on_flag_update(moby);
}

//#define headless *((int*)0xcc5200)

//SHK_HOOK(void, render, int, char*);
//void render_hook(int a1, char* pass) {
////    return;
////    if (strcmp(pass, "Shrub") == 0 ||
////        strcmp(pass, "Frame") == 0) {
////        MULTI_LOG("Ignoring %s\n", pass);
////        return;
////    }
//
//    SHK_CALL_HOOK(render, a1, pass);
//}

#define remote_pressed_buttons 0xcc5200
#define last_remote_pressed_buttons 0xcc5300
#define remote_joysticks 0xcc5204

SHK_HOOK(int32_t, cellPadGetDataRedirect, uint32_t, CellPadData*);
int32_t cellPadGetDataRedirectHook(uint32_t port_no, CellPadData *data) {
    int32_t ret = cellPadGetData(port_no, data);

	int pressed_buttons;
	int joysticks;
	int last_pressed_buttons;
	memcpy(&pressed_buttons, (void*)(remote_pressed_buttons + 0x8 * port_no), 4);
	memcpy(&joysticks, (void*)(remote_joysticks + 0x8 * port_no), 4);
	memcpy(&last_pressed_buttons, (void*)(last_remote_pressed_buttons + 0x8 * port_no), 4);

	//if (port_no == 0) {
	//	return ret;
	//}

    //if (port_no == 0) {
        int32_t len = data->len;

        memset(data, 0, 16);

        data->len = len;

//    if (data->len != 0) {
//        MULTI_LOG("Port_no: %d; Data len: %d. inputs: %.4x. Ret: %d\n", port_no, data->len,
//                  (data->button[2] << 8) + data->button[3], ret);
//    }

        if (current_level != 0) {
            if (data->len == 0 && ((remote_pressed_buttons + 0x8 * port_no) != (last_remote_pressed_buttons + 0x8 * port_no))) {
            //if (data->len == 0) {
                data->len = 24;
            }

            data->button[2] |= (pressed_buttons & 0xff00) >> 8;
            data->button[3] |= pressed_buttons & 0x00ff;
            data->button[4] = (joysticks & 0x000000ff);
            data->button[5] = (joysticks & 0x0000ff00) >> 8;
            data->button[6] = (joysticks & 0x00ff0000) >> 16;
            data->button[7] = (joysticks & 0xff000000) >> 24;

			//*(int*)(last_remote_pressed_buttons + 0x8 * port_no) = (remote_pressed_buttons + 0x8 * port_no);
			memcpy((void*)(last_remote_pressed_buttons + 0x8 * port_no), &pressed_buttons, 4);
        }
    //}

    return ret;
}

void rc3_init() {
    MULTI_LOG("Multiplayer initializing.\n");

    init_memory_allocator(memory_area, sizeof(memory_area));

    SHK_BIND_HOOK(cellGameBootCheck, cellGameBootCheckHook);
    SHK_BIND_HOOK(cellGameContentPermit, cellGameContentPermitHook);

    SHK_BIND_HOOK(STUB_008acc94, game_start);

    SHK_BIND_HOOK(update_mobys_func, update_mobys_func_hook);
    SHK_BIND_HOOK(pre_game_loop, pre_game_loop_hook);
    //SHK_BIND_HOOK(render, render_hook);

    SHK_BIND_HOOK(spawn_moby, spawn_moby_hook);
    SHK_BIND_HOOK(ctf_flag_update_func, ctf_flag_update_func_hook);

    SHK_BIND_HOOK(cellPadGetDataRedirect, cellPadGetDataRedirectHook);

    MULTI_LOG("Initialized memory allocator. Binding hooks\n");

    MULTI_LOG("Bound hooks\n");
}

void rc3_shutdown() {

}

};