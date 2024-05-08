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

#define custom_frame_count *((int*)0x1B00000)
#define progress_frame_count *((int*)0x1B00004)

#define collision_info ((float*)0x1B00010)
#define collision_info_types ((int*)0x1B00014)

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

void Game::on_tick() {
    if (current_level < 30) {
        return;
    }

    // Get forward and up vector
    Vec4 forward = hero_moby->forward;
    Vec4 up = hero_moby->up;
    Vec4 position = hero_moby->position;
    position.z += 2.0f;

    // Make 8 rays around the player in a circle with a radius of 10, without using rotate function
    Vec4 rays[16];
    for (int i = 0; i < 16; i++) {
        float angle = (i / 16.0f) * 2 * M_PI;
        rays[i] = Vec4((float)(position.x + 50 * cos_approx(angle)), (float)position.y, (float)(position.z + 50 * sin_approx(angle)), 0.0f);

        // Get collision info for the ray
        int collision = coll_line(&hero_moby->position, &rays[i], 0x0, hero_moby, nullptr);

        float dist = -10.0f;

        if (collision > 0) {
            // Get distance
            dist = distance(position, coll_output.ip);
        } else {
            dist = -10.0f;
        }

        collision_info[i*2] = dist;
        if (coll_output.pMoby != nullptr) {
            collision_info_types[i * 2] = (int)coll_output.pMoby->o_class;
        } else {
            collision_info_types[i * 2] = 0;
        }
    }

    custom_frame_count += 1;

    while (progress_frame_count != custom_frame_count) {}
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