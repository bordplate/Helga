//
// Created by Vetle Hjelle on 20/12/2022.
//

#include "Game.h"

#include <rc3/rc3.h>

#include <lib/vector.h>
#include <lib/logger.h>
#include <lib/memory.h>

#include <sys/random_number.h>
#include <sysutil/sysutil_msgdialog.h>

#include "PersistentStorage.h"

#define custom_frame_count *((int*)0xcc5180)
#define progress_frame_count *((int*)0xcc5184)
#define game_reset *((int*)0xcc5188)

struct RaycastInfo {
    float distances[64];
    int classes[64];
    float normals_x[64];
    float normals_y[64];
    float normals_z[64];
};

struct PlayerInfo {
    /* 0x0  */ Vec3 position;
    /* 0xc  */ Vec3 rotation;
    /* 0x18 */ float health;
    /* 0x1c */ int team_id;
	/* 0x20 */ int moby_state;
};

struct TeamInfo {
    Vec3 flag_position;
    int flag_state;
    float team_health;
    int flag_holder;
    int score;
};

#define BASE_ADDRESS 0xcd0000

//#define raycast_info ((struct RaycastInfo*)(BASE_ADDRESS + (0x100 * 4 * 0)))
#define PLAYER_INFO ((struct PlayerInfo*)(BASE_ADDRESS + (0x100 * 4 * 0)))
#define TEAM_INFO ((struct TeamInfo*)(0xce0000))

Moby *test_moby[8];
float sqrt_ppc(float number) {
    float result;
    asm("fsqrts %0, %1" : "=f"(result) : "f"(number));
    return result;
}

float distance(const Vec4& a, const Vec4& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrt_ppc(dx * dx + dy * dy + dz * dz);
}

// Function to raise a number to an integer power
long double power(double base, int exponent) {
    long double result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

// Factorial function
long double factorial(int n) {
    long double result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Approximate sine function
long double sin_approx(double x) {
    long double result = 0;
    for (int n = 0; n < 10; ++n) {
        long double term = power(x, 2 * n + 1) / factorial(2 * n + 1);
        result += (n % 2 == 0) ? term : -term;
    }
    return result;
}

// Approximate cosine function
long double cos_approx(double x) {
    long double result = 0;
    for (int n = 0; n < 10; ++n) {
        long double term = power(x, 2 * n) / factorial(2 * n);
        result += (n % 2 == 0) ? term : -term;
    }
    return result;
}

// For whatever dumb reason I can't get the compiler to include
//  .cpp files under /lib/, so it's defined here.
// Maybe better to just have it here anyway. Idk.
LogLevel Logger::log_level_ = Debug;

const double M_PI = 3.14159265358979323846;

void Game::start() {

}

void Game::on_game_start() {
    // Loads into multiplayer
    destination_level = 0x27;
    load_level = 1;

	resets = 0;
}

void Game::flag_update(Moby* flag_moby) {
    int team_id = flag_moby->o_class == 7217 ? 0 : 1;

    TeamInfo* team_info = &TEAM_INFO[team_id];

    team_info->flag_holder = *((int*)((u32)flag_moby->pVars + 0x10));
    team_info->flag_state = flag_moby->state;

    team_info->flag_position.x = flag_moby->pos.x;
    team_info->flag_position.y = flag_moby->pos.y;
    team_info->flag_position.z = flag_moby->pos.z;
}

void Game::reset_game() {
	multiplayer_level = 6;
    is_local_multiplayer = 1;
    num_local_players = 4;

    nwSetGameSetupFlagsForGameType(1);

    nwConnect(nullptr);
    nwJoin();

    player1_controller_ptr = (void*)0xd992c0;
    player2_controller_ptr = (void*)0xd99824;
    player3_controller_ptr = (void*)0xd99d88;
    player4_controller_ptr = (void*)0xd9a2ec;

    game_settings->gameType = 1;
    game_settings->altGameType = 0;
    game_settings->level = 46;
    game_settings->unk2 = 4;
    game_settings->numPlayers = 4;
    game_settings->ctfCap = 255;
    game_settings->showPlayerNames = true;
    game_settings->startWithChargeboots = true;
    game_settings->shizzolate = true;

    game_settings->playerTeams[0] = 0;
    game_settings->playerTeams[1] = 0;
    game_settings->playerTeams[2] = 1;
    game_settings->playerTeams[3] = 1;

	game_settings->playerSkins[0] = 2;
	game_settings->playerSkins[1] = 3;
	game_settings->playerSkins[2] = 4;
	game_settings->playerSkins[3] = 5;

    sprintf((char*)&game_settings->playerNames[0], "Ben");
    sprintf((char*)&game_settings->playerNames[1], "Jen");
    sprintf((char*)&game_settings->playerNames[2], "Ken");
    sprintf((char*)&game_settings->playerNames[3], "Len");

    destination_level = 46;
    load_level = 1;
    lobby_mode = 2;

	resets = game_reset;
}

#define who_knows *((int*)0xee933c)
#define widget_layers *((int*)0xcaa090)

void Game::on_tick() {
	if (game_reset != resets) {
		resets = game_reset;

		// Not setting either of these makes the game not reload at all
		lobby_mode = 5;
		who_knows = 0;

		widget_layers = 0;  // Causes a crash after 10 resets if not set to 0

		LoadLevel(0x27, 1);

		return;
	}

    if (current_level < 30) {
        return;
    }

    if (current_level != 46) {
		this->reset_game();

		return;
    }

    TEAM_INFO[0].team_health = 0;
    TEAM_INFO[1].team_health = 0;

    TEAM_INFO[0].score = (*(int**)(0x13dd374))[2];
    TEAM_INFO[1].score = (*(int**)(0x13dd374))[3];

    for (int player_id = 0; player_id < num_local_players; player_id++) {
        Moby* moby = team_data[player_id].player_moby;
        PlayerInfo* player_info = &PLAYER_INFO[player_id];

        player_info->position.x = moby->pos.x;
        player_info->position.y = moby->pos.y;
        player_info->position.z = moby->pos.z;

        player_info->rotation.x = moby->rot.x;
        player_info->rotation.y = moby->rot.y;
        player_info->rotation.z = moby->rot.z;

        player_info->health = team_data[player_id].health;
        player_info->team_id = game_settings->playerTeams[player_id];

		player_info->moby_state = (int)team_data[player_id].state;

        TEAM_INFO[player_info->team_id].team_health += player_info->health;
    }

    while (progress_frame_count < custom_frame_count && current_level != 0) {}
    custom_frame_count += 1;
}

void Game::before_player_spawn() {

}

void Game::on_render() {

}

void Game::transition_to(View *view) {
    Logger::trace("Starting transition to a new view");

    if (current_view) {
        current_view->on_unload();
        delete current_view;
    }

    current_view = view;
    current_view->on_load();

    Logger::trace("Done transitioning");
}

void Game::alert(String& message) {
    cellMsgDialogOpen2(CELL_MSGDIALOG_TYPE_SE_TYPE_NORMAL, message.c_str(), nullptr, nullptr, nullptr);
}

extern "C" void _c_game_tick() {
    Game::shared().on_tick();
}

extern "C" void _c_game_render() {
    Game::shared().on_render();
}

extern "C" void _c_game_start() {
    Game::shared().start();
}

extern "C" void _c_game_quit() {

}

extern "C" void _c_on_respawn() {
    Game::shared().before_player_spawn();
}

extern "C" void _c_on_game_start() {
    Game::shared().on_game_start();
}

extern "C" void _c_on_flag_update(Moby* flag_moby) {
    Game::shared().flag_update(flag_moby);
}